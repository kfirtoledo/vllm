# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
import torch
from pathlib import Path


from vllm.logger import init_logger
from vllm.v1.offloading.worker.worker import OffloadingHandler, TransferSpec, TransferResult

logger = init_logger(__name__)
import time
import storage_offload_ext

import psutil
def get_current_cpu() -> int:
    """Return the current CPU/core the thread is running on."""
    if hasattr(os, "sched_getcpu"):
        try:
            return os.sched_getcpu()
        except Exception:
            pass
    try:
        return psutil.Process().cpu_num()
    except Exception:
        return -1  # Unknown

# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_PINNED_MEMORY_GB = 100
DEFAULT_MAX_CONCURRENCY = 64

class StorageOffloadingHandler(OffloadingHandler):
    """Base handler with common helpers for Storage offloading."""

    def __init__(self,
                 model_name: str,
                 tp_size: int,
                 tp_rank: int,
                 dtype: torch.dtype,
                 gpu_blocks_per_file: int,
                 max_concurrency: int = None,
                 max_pinned_memory_gb: int = DEFAULT_MAX_PINNED_MEMORY_GB,  # in GB
                 root_dir: str = "/tmp/shared-kv"):
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.base_path = self.get_kv_cache_base_path(model_name, tp_size, tp_rank, dtype, root_dir)
        self.max_concurrency = min(int(max_concurrency or (os.cpu_count())), DEFAULT_MAX_CONCURRENCY)
        self.max_pinned_memory_gb = max_pinned_memory_gb
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

    # ----------------------------
    # Shared path helpers
    # ----------------------------
    @staticmethod
    def get_kv_cache_base_path(model_name, tp_size, tp_rank, dtype, root_dir: str) -> Path:
        dtype_str = str(dtype).replace("torch.", "")
        base_path = Path(f"{root_dir}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}")
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    @staticmethod
    def get_file_name(base_path: Path, block_hash: int) -> Path:
        block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"
        subfolder1, subfolder2 = block_hash_hex[:8], block_hash_hex[8:16]
        full_path = base_path / subfolder1 / subfolder2 / f"{block_hash_hex}.bin"
        os.makedirs(full_path.parent, exist_ok=True)
        return full_path

    def compute_pinned_mb(self,dst_tensors, gpu_blocks_per_file, safety=1.0, min_mb=64, max_mb=None):
        ref = dst_tensors[0]
        block_elems = ref[:, 0].numel()                                  # (2, H, B, D)
        total_elems = block_elems * len(dst_tensors) * gpu_blocks_per_file
        total_bytes = total_elems * ref.element_size()
        mb = math.ceil((total_bytes / (1024 * 1024)) * safety)
        if min_mb: mb = max(mb, min_mb)
        if max_mb: mb = min(mb, max_mb)
        return mb

# ----------------------------
# Futures management
# ----------------------------
    def get_finished(self) -> list[TransferResult]:
        return storage_offload_ext.get_finished_ext()
    def __del__(self):
        storage_offload_ext.cleanup_performance_resources()
# ----------------------------------------------------------------------
# GPU → Storage (PUT)
# ----------------------------------------------------------------------

class GPUStorageOffloadingHandler(StorageOffloadingHandler):
    def __init__(self, model_name, tp_size, tp_rank, src_tensors,
                 gpu_blocks_per_file, dtype, max_concurrency=None,
                 max_pinned_memory_gb = DEFAULT_MAX_PINNED_MEMORY_GB, root_dir="/tmp/shared-kv"):
        super().__init__(model_name, tp_size, tp_rank, dtype,
                         gpu_blocks_per_file, max_concurrency, max_pinned_memory_gb, root_dir)

        self.src_tensors = src_tensors
        self.buffer_size_mb = self.compute_pinned_mb(src_tensors, gpu_blocks_per_file)

        storage_offload_ext.init_performance_resources(
            io_threads=self.max_concurrency,
            pinned_buffer_size_mb=self.buffer_size_mb,
            max_pinned_memory_gb=self.max_pinned_memory_gb,  # TODO: separate pools for PUT/GET
        )

        logger.info(f"GPUStorageOffloadingHandler: max_concurrency={self.max_concurrency} pinned_buffer_size_mb={self.buffer_size_mb}")

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        time_start = time.time()
        src_specs, dst_specs = spec
        if not dst_specs:
            return True

        target_files, all_src_tensors, all_block_ids = [], [], []
        for i, dst_spec in enumerate(dst_specs):
            start = i * self.gpu_blocks_per_file
            end = min((i + 1) * self.gpu_blocks_per_file, len(src_specs))
            if start >= len(src_specs):
                break
            block_ids = [src_specs[j].block_id for j in range(start, end)]
            target_file = str(self.get_file_name(self.base_path, dst_spec.block_hash))
            target_files.append(target_file)
            all_src_tensors.append(self.src_tensors)
            all_block_ids.append(block_ids)
            #print(f"[DEBUG PUT] dst_spec {i}: len block_ids={len(block_ids)} block_ids={block_ids}")
        stream = self.h2d_stream
        with torch.cuda.stream(stream):
            storage_offload_ext.transfer_async_put_ext(
                job_id, target_files, all_src_tensors, all_block_ids )
        #print(f"Total PUT job {job_id} setup time: {time.time()-time_start:.6f} sec")

        return True

# ----------------------------------------------------------------------
# Storage → GPU (GET)
# ----------------------------------------------------------------------
class StorageGPUOffloadingHandler(StorageOffloadingHandler):
    """Handler for reading KV blocks from shared storage back into GPU."""

    def __init__(self, model_name, tp_size, tp_rank, dtype,
                 gpu_blocks_per_file, dst_tensors,
                 max_concurrency=None, max_pinned_memory_gb = DEFAULT_MAX_PINNED_MEMORY_GB, root_dir="/tmp/shared-kv"):
        super().__init__(model_name, tp_size, tp_rank, dtype,
                         gpu_blocks_per_file, max_concurrency, max_pinned_memory_gb, root_dir)
        self.dst_tensors = dst_tensors

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_specs, dst_specs = spec
        if not src_specs:
            return True

        source_files = []
        all_block_ids = []
        first_len = len(dst_specs) % self.gpu_blocks_per_file or self.gpu_blocks_per_file
        start = 0
        for i, src_spec in enumerate(src_specs):
            if i == 0:
                size = first_len
            else:
                size = self.gpu_blocks_per_file

            end = min(start + size, len(dst_specs))
            block_ids = [dst.block_id for dst in dst_specs[start:end]]
            #print(f"[DEBUG GET] src_spec {i}: len block_ids={len(block_ids)} block_ids={block_ids}")
            source_files.append(str(self.get_file_name(self.base_path, src_spec.block_hash)))
            all_block_ids.append(block_ids)
            start += size
        stream = self.d2h_stream

        with torch.cuda.stream(stream):
            storage_offload_ext.transfer_async_get_ext(
                job_id,
                source_files,
                all_block_ids,
                self.dst_tensors,
                self.gpu_blocks_per_file,
            )
        return True

