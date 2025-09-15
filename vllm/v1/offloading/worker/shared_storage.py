# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import math
import mmap
import torch
import numpy as np
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import threading
from vllm.logger import init_logger
from vllm.v1.offloading.worker.worker import OffloadingHandler, TransferSpec, TransferResult

logger = init_logger(__name__)
import time

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

class StorageOffloadingHandler(OffloadingHandler):
    """Base handler with common helpers for Storage offloading."""

    def __init__(self,
                 model_name: str,
                 tp_size: int,
                 tp_rank: int,
                 dtype: torch.dtype,
                 gpu_blocks_per_file: int,
                 max_concurrency: int = None,
                 root_dir: str = "/tmp/shared-kv"):
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.base_path = self.get_kv_cache_base_path(model_name, tp_size, tp_rank, dtype, root_dir)
        self.max_concurrency = max_concurrency or os.cpu_count()
        self.pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
        self.futures: dict[int, Future] = {}
        self.futs_files = {}


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


# ----------------------------
# Futures management
# ----------------------------
    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []

        # job_id -> list of futures
        for job_id, fut_list in list(self.futs_files.items()):
            # ensure we always have a list
            if not isinstance(fut_list, list):
                fut_list = [fut_list]

            # check if all futures for this job are done
            all_done = all(fut.done() for fut in fut_list if fut is not None)

             # if all done, check if any failed
            if all_done:
                results.append((job_id, True))
                # cleanup after finishing
                self.futs_files.pop(job_id, None)
                self.futures.pop(job_id, None)

        return results

# ----------------------------------------------------------------------
# GPU → Storage (PUT)
# ----------------------------------------------------------------------

class GPUStorageOffloadingHandler(StorageOffloadingHandler):
    """Handler for writing KV blocks from GPU to shared storage."""

    def __init__(self,
                    model_name: str,
                    tp_size: int,
                    tp_rank: int,
                    src_tensors: List[torch.Tensor],
                    gpu_blocks_per_file: int,
                    dtype: torch.dtype,
                    max_concurrency: int = None,
                    root_dir: str = "/tmp/shared-kv"):
        super().__init__(model_name, tp_size, tp_rank, dtype,
                            gpu_blocks_per_file, max_concurrency, root_dir)

        self.src_tensors = src_tensors
        self.write_pool = ThreadPoolExecutor(max_workers=self.max_concurrency)

    def copy_gpu_tensors_to_buffer(
        self,
        src_tensors: List[torch.Tensor],
        block_ids_list: List[int],
    ) -> memoryview:
        """Serialize selected [K,V] blocks from all layers to a contiguous CPU buffer."""
        blocks = []
        # Collect [K,V] slices for every (layer, block_id) into a list
        for block_id in block_ids_list:
            for tensor in src_tensors:
                blocks.append(tensor[:, block_id])  # [2, H, B, D]


        # Concatenate all selected [K,V] block slices into one big tensor along dim=0 → [2 * num_layers * num_blocks, H, B, D]
        flat = torch.cat(blocks, dim=0)
        flat_cpu = flat.contiguous().detach().cpu()  # Ensure contiguous

        # Non-blocking copy to pinned CPU memory
        #flat_cpu = flat.to("cpu")#, non_blocking=True)

        # If you need the buffer right away for writing → sync once
        #torch.cuda.current_stream().synchronize()

        if flat_cpu.dtype == torch.bfloat16:  # If storing bf16, write as uint16 payload
            flat_cpu = flat_cpu.view(torch.uint16)

        # Zero-copy: expose tensor bytes as a NumPy view wrapped in memoryview for fast I/O
        return memoryview(flat_cpu.numpy())

    def write_buffer_to_file(self, target_file: Path, buffer: memoryview) -> None:
        tmp_file_path = target_file.with_suffix(".tmp")
        with open(tmp_file_path, "wb") as f:
            f.write(buffer)
        os.replace(tmp_file_path, target_file)


    def _write_one(self,dst_spec, block_ids: List[int]) -> bool:
        cpu_id = get_current_cpu()
        print(f"[DEBUG] _write_one: handling block {dst_spec.block_hash} on CPU {cpu_id}")
        block_hash = dst_spec.block_hash
        target_file = self.get_file_name(self.base_path, block_hash)
        if os.path.exists(target_file):
            return True
        try:
            #start_time= time.time()
            buf = self.copy_gpu_tensors_to_buffer(self.src_tensors, block_ids)
            self.write_buffer_to_file(target_file, buf)
            #print(f"PUT {block_ids} in {time.time()-start_time:.6f} sec")
            return True
        except Exception as e:
            logger.warning("PUT failed for %s: %r", target_file, e)
            try:
                if os.path.exists(target_file):
                    os.remove(target_file)
            except Exception:
                pass
            return False

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        start_time= time.time()
        src_specs, dst_specs = spec
        if not dst_specs:
            return True

        def _req():
            ok = True
            futs = []
            for i, dst_spec in enumerate(dst_specs):
                start = i * self.gpu_blocks_per_file
                end = min((i + 1) * self.gpu_blocks_per_file, len(src_specs))
                if start >= len(src_specs):
                    break
                block_ids = [src_specs[j].block_id for j in range(start, end)]
                futs.append(self.write_pool.submit(self._write_one, dst_spec, block_ids))

            print("len fut_list:", len(futs))
            self.futs_files[job_id] = futs
            return ok

        self.futures[job_id] = self.pool.submit(_req)
        print(f"Scheduled PUT job {job_id} with {len(dst_specs)} files in {time.time()-start_time:.6f} sec")
        return True

# ----------------------------------------------------------------------
# Storage → GPU (GET)
# ----------------------------------------------------------------------

class StorageGPUOffloadingHandler(StorageOffloadingHandler):
    """Handler for reading KV blocks from shared storage back into GPU."""

    def __init__(self, model_name, tp_size, tp_rank, dtype,
                 gpu_blocks_per_file, attn_backend, dst_tensors,
                 max_concurrency=None, root_dir="/tmp/shared-kv"):
        super().__init__(model_name, tp_size, tp_rank, dtype,
                         gpu_blocks_per_file, max_concurrency, root_dir)
        self.attn_backend = attn_backend
        self.dst_tensors = dst_tensors
        self.reads_pool = ThreadPoolExecutor(max_workers=self.max_concurrency)

    def read_file_to_bytes(self, path: Path):
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return mm, f

    def copy_buffer_to_gpu_tensors(self, buf, block_ids_list):
        # Wrap raw buffer in a memoryview (zero-copy access to bytes)
        buf_mv = memoryview(buf)

        # Shape of one KV block (K and V): [2, num_heads, block_size, head_dim]
        block_shape = self.dst_tensors[0][:, 0].shape # (2, H, B, D)
        elems_per_block = np.prod(block_shape)  # number of elements in one block

        num_layers = len(self.dst_tensors)          # how many layers total
        num_blocks = self.gpu_blocks_per_file       # how many block IDs
        expected_size = elems_per_block * num_layers * num_blocks  # total elems in file

      # Map torch dtype → numpy dtype (e.g., float16 → np.float16)
        np_dtype = self.torch_dtype_to_numpy(self.dst_tensors[0].dtype)
        # Interpret raw buffer as 1D NumPy array (zero-copy view)
        np_arr = np.frombuffer(buf_mv, dtype=np_dtype)

        # Sanity check: file size must match what we expect
        full_block = len(block_ids_list) % self.gpu_blocks_per_file == 0
        if np_arr.size != expected_size and full_block:
            raise ValueError(
                f"File has {np_arr.size} elements but expected {expected_size}"
            )
        start_time= time.time()
        # Reshape into [num_blocks, num_layers, 2, H, B, D] for easy indexing
        np_arr = np_arr.reshape(num_blocks, num_layers, *block_shape)
        torch_arr = torch.from_numpy(np_arr)
        # Special case: if data was stored as uint16 (for bf16), reinterpret back to bf16
        if self.dst_tensors[0].dtype == torch.bfloat16 and torch_arr.dtype != torch.bfloat16:
            torch_arr = torch_arr.view(torch.bfloat16)
        print(f"Loaded {len(block_ids_list)} blocks for {num_layers} layers in {time.time()-start_time:.6f} sec")
        start_time= time.time()
        # Calculate first block offset
        offset = 0
        if not full_block:
            offset = self.gpu_blocks_per_file - len(block_ids_list)  # If not full blocks, start at the offset
        # Build list of per-layer tensors, each shaped [2, num_blocks, H, B, D]
        src_tensors: list[torch.Tensor] = []
        for i in range(num_layers):
            blocks_for_layer = []
            for b in range(offset,num_blocks):
                block_tensor = torch_arr[b, i]  # one block for this layer [2, H, B, D]
                blocks_for_layer.append(block_tensor)
            # Stack all blocks along dim=1 → [2, num_blocks, H, B, D]
            layer_tensor = torch.stack(blocks_for_layer, dim=1).contiguous()
            src_tensors.append(layer_tensor)
        print(f"Prepared {len(block_ids_list)} blocks for {num_layers} layers in {time.time()-start_time:.6f} sec")
        # Mapping: (source block index, destination block index)

        src_to_dst = torch.tensor(
            [(i, bid) for i, bid in enumerate(block_ids_list)],
            dtype=torch.int64,
            device="cpu",
        )

        # Perform optimized block transfer (GPU kernel if available, else CPU)
        time_start = time.time()
        if torch.cuda.is_available():
            with torch.cuda.stream(torch.cuda.Stream()):  # async CUDA stream
                self.attn_backend.swap_blocks_multi_layer(src_tensors, self.dst_tensors, src_to_dst)
                #print(f"Swapped {block_ids_list} in {time.time()-time_start:.6f} sec")
        else:
            self.attn_backend.swap_blocks_multi_layer(src_tensors, self.dst_tensors, src_to_dst)

    def torch_dtype_to_numpy(self,dtype) -> np.dtype:
        mapping = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.bfloat16: np.uint16,  # stored as uint16 on disk
        }
        return mapping[dtype]



    def _read_one_file(self, src_spec, block_ids: List[int]) -> bool:
        path = self.get_file_name(self.base_path, src_spec.block_hash)

        try:
            #start_put = time.time()
            buf, f = self.read_file_to_bytes(path)
            #end_put = time.time()- start_put
            #print(f"Time taken to read file {path}: {end_put} seconds")
        except Exception as e:
            logger.warning("GET read failed for %s: %r", path, e)
            return False
        try:

            copy_time_start = time.time()
            self.copy_buffer_to_gpu_tensors(buf, block_ids)
            print(f"copy_buffer_to_gpu_tensors taken: {time.time()-copy_time_start:.6f} sec")
            return True
        except Exception as e:
            logger.warning("GET failed for %s: %r", path, e)
            return False
        finally:
            buf.close()
            f.close()


    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        cpu_id = get_current_cpu()
        print(f"[DEBUG] Main read transfer_async: job_id {job_id} on CPU {cpu_id}")
        time_start = time.time()
        src_specs, dst_specs = spec
        if not src_specs:
            return True  # no-op

        expected_src = math.ceil(len(dst_specs) / self.gpu_blocks_per_file) if dst_specs else 0
        assert len(src_specs) == expected_src, (
            f"Mismatch source spec {len(src_specs)} vs dst {len(dst_specs)}"
        )

        def _req():
            ok = True
            first_len = len(dst_specs) % self.gpu_blocks_per_file or self.gpu_blocks_per_file
            start = 0
            fut=[]
            for i, src_spec in enumerate(src_specs):
                size = first_len if i == 0 else self.gpu_blocks_per_file
                end = min(start + size, len(dst_specs))
                if start >= len(dst_specs):
                    break
                block_ids = [dst_specs[j].block_id for j in range(start, end)]
                fut.append(self.reads_pool.submit(self._read_one_file, src_spec, block_ids))

                print(f"appended future for job {job_id}, future {i}")
                start += size

            print("len fut_list:", len(fut))
            self.futs_files[job_id] = fut
            return ok


        self.futures[job_id] = self.pool.submit(_req)
        print(f"Total GET job {job_id} setup time: {time.time()-time_start:.6f} sec")
        return True
