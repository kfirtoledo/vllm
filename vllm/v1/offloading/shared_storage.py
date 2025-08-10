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
                root_dir=self.extra_config.get("shared_kv_root", "/mnt/shared-kv")
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
                src_block_size=self.offloaded_block_size,
                dtype=self.vllm_config.cache_config.cache_dtype,
                root_dir=self.extra_config.get("shared_kv_root", "/mnt/shared-kv")
            )

            self._shared_to_gpu_func = generate_get_transfer_function(
                dst_tensors=gpu_caches,
                dst_block_size=self.offloaded_block_size,
                model_name=self.vllm_config.model_config.model,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                tp_rank=self.vllm_config.parallel_config.rank,
                dtype=self.vllm_config.cache_config.cache_dtype,
                root_dir=self.extra_config.get("shared_kv_root", "/mnt/shared-kv")
            )


        assert self._gpu_to_shared_func is not None
        assert self._shared_to_gpu_func is not None
        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._gpu_to_shared_func
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._shared_to_gpu_func
