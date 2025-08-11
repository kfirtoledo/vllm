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
from vllm.v1.offloading.worker.cpu import (
    create_cpu_tensors,
)
from vllm.v1.offloading.worker.shared_storage import (
    generate_put_transfer_function,
    generate_get_transfer_function,
)
from vllm.v1.offloading.worker.worker import TransferFunction


class SharedStorageOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for shared storage backend (e.g., mounted NFS, PVC).
    """
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self._num_blocks: Optional[int] = None
        self._manager: Optional[OffloadingManager] = None

        self._gpu_to_shared_func: Optional[TransferFunction] = None
        self._shared_to_gpu_func: Optional[TransferFunction] = None
        self.shared_storage_path   = self.extra_config.get("shared_storage_path", "/mnt/shared-kv")
        self.threads_per_request   = self.extra_config.get("threads_per_request", 16) # Threads used for processing a single request
        self.max_parallel_requests = self.extra_config.get("max_parallel_requests", 8) # Limit for concurrent requests across the whole system
        self.gpu_blocks_per_file   = int(self.offloaded_block_size / self.gpu_block_size)
        assert self.offloaded_block_size % self.gpu_block_size == 0, "offloaded_block_size must be a multiple of gpu_block_size"

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
            )
        return self._manager

    def get_transfer_functions(
        self,
        kv_caches: dict[str, torch.Tensor]
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], TransferFunction]]:

        if not self._gpu_to_shared_func or not self._shared_to_gpu_func:

            gpu_caches, _ = create_cpu_tensors(
                kv_caches,
                self.gpu_block_size,
                self.offloaded_block_size,
                self.num_blocks
            )

            self._gpu_to_shared_func = generate_put_transfer_function(
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                src_tensors=gpu_caches,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                dtype=self.vllm_config.cache_config.cache_dtype,
                root_dir=self.shared_storage_path,
                max_concurrency=self.threads_per_request,
            )

            self._shared_to_gpu_func = generate_get_transfer_function(
                dst_tensors=gpu_caches,
                gpu_blocks_per_file=self.gpu_blocks_per_file,
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                root_dir=self.shared_storage_path,
                max_concurrency=self.threads_per_request,
            )

        assert self._gpu_to_shared_func is not None
        assert self._shared_to_gpu_func is not None
        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._gpu_to_shared_func, self.max_parallel_requests
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._shared_to_gpu_func, self.max_parallel_requests
