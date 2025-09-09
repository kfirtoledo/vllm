# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from collections.abc import Iterator
from typing import Optional

from vllm.attention import get_attn_backend
from vllm.config import VllmConfig
from vllm.v1.offloading.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.offloading.mediums import GPULoadStoreSpec, SharedStorageLoadStoreSpec
from vllm.v1.offloading.shared_storage_manager import SharedStorageOffloadingManager
from vllm.v1.offloading.spec import OffloadingSpec

from vllm.v1.offloading.worker.shared_storage import (
    GPUStorageOffloadingHandler,
    StorageGPUOffloadingHandler,
)

from vllm.v1.offloading.worker.worker import OffloadingHandler


class SharedStorageOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self._num_blocks: Optional[int] = None
        self._manager: Optional[OffloadingManager] = None

        self.shared_storage_path   = self.extra_config.get("shared_storage_path", "/tmp/shared-kv")
        self.threads_per_request   = self.extra_config.get("threads_per_request", 16) # Threads used for processing a single request
        # self.max_parallel_requests = self.extra_config.get("max_parallel_requests", 8) # Limit for concurrent requests across the whole system
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
                attn_backend = get_attn_backend(
                    self.vllm_config.model_config.get_head_size(),
                    self.vllm_config.model_config.dtype,
                    self.vllm_config.cache_config.cache_dtype,
                    self.gpu_block_size,
                    self.vllm_config.model_config.is_attention_free,
                    use_mla=self.vllm_config.model_config.use_mla,
                )

                # Worker-side handlers
                self._gpu_to_storage = GPUStorageOffloadingHandler(
                    model_name=self.vllm_config.model_config.model,
                    tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                    tp_rank=self.vllm_config.parallel_config.rank,
                    src_tensors=list(kv_caches.values()),
                    gpu_blocks_per_file=self.gpu_blocks_per_file,
                    dtype=self.vllm_config.cache_config.cache_dtype,
                    root_dir=self.shared_storage_path,
                )

                self._storage_to_gpu = StorageGPUOffloadingHandler(
                    model_name=self.vllm_config.model_config.model,
                    tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                    tp_rank=self.vllm_config.parallel_config.rank,
                    dtype=self.vllm_config.cache_config.cache_dtype,
                    gpu_blocks_per_file=self.gpu_blocks_per_file,
                    attn_backend=attn_backend,
                    dst_tensors=list(kv_caches.values()),
                    root_dir=self.shared_storage_path,
                )

        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._gpu_to_storage
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._storage_to_gpu
