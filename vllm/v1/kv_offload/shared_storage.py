# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from collections.abc import Iterator
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec, SharedStorageLoadStoreSpec
from vllm.v1.kv_offload.shared_storage_manager import SharedStorageOffloadingManager
from vllm.v1.kv_offload.spec import OffloadingSpec

from vllm.v1.kv_offload.worker.shared_storage import (
    GPUStorageOffloadingHandler,
    StorageGPUOffloadingHandler,
    DEFAULT_MAX_PINNED_MEMORY_GB,
    DEFAULT_MAX_THREADS_PER_GPU
)

from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class SharedStorageOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self._num_blocks: Optional[int] = None
        self._manager: Optional[OffloadingManager] = None

        self.threads_per_gpu = int(self.extra_config.get("threads_per_gpu", DEFAULT_MAX_THREADS_PER_GPU))
        self.shared_storage_path   = self.extra_config.get("shared_storage_path", "/tmp/shared-kv")
        self.max_pinned_memory_gb        = self.extra_config.get("max_pinned_memory_gb", DEFAULT_MAX_PINNED_MEMORY_GB) # Max pinned CPU buffer in GB

        self.gpu_blocks_per_file   = int(self.offloaded_block_size / self.gpu_block_size)
        assert self.offloaded_block_size % self.gpu_block_size == 0, "offloaded_block_size must be a multiple of gpu_block_size"

        self._manager: Optional[OffloadingManager] = None
        self._gpu_to_storage: Optional[OffloadingHandler] = None
        self._storage_to_gpu: Optional[OffloadingHandler] = None

    @property
    def num_blocks(self) -> int:
        if self._num_blocks is None:
            self._num_blocks = self.extra_config.get(
                "num_shared_blocks", self.vllm_config.cache_config.num_gpu_blocks)
        return self._num_blocks

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = SharedStorageOffloadingManager(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                root_dir=self.shared_storage_path,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:


        if not self._gpu_to_storage or not self._storage_to_gpu:
                self._gpu_to_storage = GPUStorageOffloadingHandler(
                    model_name           = self.vllm_config.model_config.model,
                    tp_size              = self.vllm_config.parallel_config.tensor_parallel_size,
                    tp_rank              = self.vllm_config.parallel_config.rank,
                    src_tensors          = list(kv_caches.values()),
                    gpu_blocks_per_file  = self.gpu_blocks_per_file,
                    dtype                = self.vllm_config.cache_config.cache_dtype,
                    threads_per_gpu      = self.threads_per_gpu,
                    max_pinned_memory_gb = self.max_pinned_memory_gb,
                    root_dir             = self.shared_storage_path,
                )

                self._storage_to_gpu = StorageGPUOffloadingHandler(
                    model_name           = self.vllm_config.model_config.model,
                    tp_size              = self.vllm_config.parallel_config.tensor_parallel_size,
                    tp_rank              = self.vllm_config.parallel_config.rank,
                    dtype                = self.vllm_config.cache_config.cache_dtype,
                    gpu_blocks_per_file  = self.gpu_blocks_per_file,
                    dst_tensors          = list(kv_caches.values()),
                    root_dir             = self.shared_storage_path,
                    threads_per_gpu      = self.threads_per_gpu,
                    max_pinned_memory_gb = self.max_pinned_memory_gb,
                )

        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._gpu_to_storage
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._storage_to_gpu
