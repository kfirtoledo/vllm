# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
import torch
from pathlib import Path
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from typing import Optional
from vllm.v1.offloading.mediums import SharedStorageLoadStoreSpec
from vllm.v1.offloading.abstract import (LoadStoreSpec, OffloadingManager,
                                         PrepareStoreOutput,)
from vllm.v1.offloading.worker.shared_storage import StorageOffloadingHandler
from vllm.v1.offloading.lru_manager import BlockStatus
from vllm.logger import init_logger

logger = init_logger(__name__)

class SharedStorageOffloadingManager(OffloadingManager):
    """
    An SharedStorageOffloadingManager for managing offloading to shared storage.
    """
    def __init__(self, model_name: str, tp_size: int, tp_rank: int, dtype: torch.dtype, root_dir: str = "/tmp/shared-kv") -> None:
        self.model_name = model_name
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dtype = dtype

        self.base_path: Path = StorageOffloadingHandler.get_kv_cache_base_path(
            dtype=dtype,
            model_name=model_name,
            tp_size=tp_size,
            tp_rank=tp_rank,
            root_dir=root_dir,
        )

    def lookup(self, block_hashes: list[int]) -> int:
        hit_count = 0
        for block_hash in block_hashes:

            file_path = StorageOffloadingHandler.get_file_name(self.base_path, block_hash)
            if not os.path.exists(file_path):
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: list[int]) -> list[LoadStoreSpec]:
        """
        For shared storage, loading is stateless — return specs pointing to files.
        """
        specs: list[LoadStoreSpec] = []

        for block_hash in block_hashes:
            spec = SharedStorageLoadStoreSpec(block_hash=block_hash)
            specs.append(spec)

        return specs


    def touch(self, block_ids: list[str]):
        """
        Update the access time of the files corresponding to the given block IDs.
        """
        now = time.time()
        for block_id in block_ids:
            path = StorageOffloadingHandler.get_file_name(self.base_path, block_id)
            try:
                os.utime(path, (now, -1))  # Set atime to the current time (reading, opening, or executing the file).
            except FileNotFoundError:
                pass

    def complete_load(self, block_hashes: list[int]):
        """
        For shared storage, loading is stateless — no action needed.
        """
        pass


    def prepare_store(self, block_hashes: list[int]) -> Optional[PrepareStoreOutput]:
        """
        In shared storage, you can always store new blocks. No eviction required.
        """
        store_specs = []

        for block_hash in block_hashes:
            store_specs.append(
                SharedStorageLoadStoreSpec(
                    block_hash=block_hash
                )
            )

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes,
            store_specs=store_specs,
            block_hashes_evicted=[],  # no eviction needed
        )

    def complete_store(self, block_hashes: list[int], is_success: bool = True):
        """
        For shared storage, storing is stateless — no action needed.
        """
        if not is_success: # TODO- Check if this is needed
            # If storing failed, need to clean up the files
            for block_hash in block_hashes:
                path = StorageOffloadingHandler.get_file_name(self.base_path, block_hash)
                if os.path.exists(path):
                    os.remove(path)
        # Otherwise, files are already saved and no further action is needed.
