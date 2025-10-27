# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
import torch
from pathlib import Path
from collections.abc import Iterable
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.mediums import SharedStorageLoadStoreSpec
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.worker.shared_storage import StorageOffloadingHandler
from vllm.logger import init_logger

logger = init_logger(__name__)


class SharedStorageOffloadingManager(OffloadingManager):
    """
    SharedStorageOffloadingManager manages KV offloading to a shared storage medium.
    """

    def __init__(
        self,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        root_dir: str = "/tmp/shared-kv",
    ) -> None:
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

    # ----------------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------------
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        """Return how many consecutive blocks from the start are already offloaded."""
        hit_count = 0
        for block_hash in block_hashes:
            file_path = StorageOffloadingHandler.get_file_name(self.base_path, block_hash)
            if not os.path.exists(file_path):
                break
            hit_count += 1
        return hit_count

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """For shared storage, loading is stateless - return specs pointing to files."""
        return SharedStorageLoadStoreSpec(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        """Update access time for given block hashes."""
        now = time.time()
        for block_hash in block_hashes:
            path = StorageOffloadingHandler.get_file_name(self.base_path, block_hash)
            try:
                os.utime(path, (now, -1))
            except FileNotFoundError:
                pass

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        """Stateless load - no post-load action needed."""
        return

    # ----------------------------------------------------------------------
    # Store
    # ----------------------------------------------------------------------
    def prepare_store(self, block_hashes: Iterable[BlockHash]) -> Optional[PrepareStoreOutput]:
        """
        In shared storage, you can always store new blocks.
        No eviction required.
        """
        block_hashes_to_store: list[BlockHash] = []
        for block_hash in block_hashes:
            file_path = StorageOffloadingHandler.get_file_name(self.base_path, block_hash)
            if os.path.exists(file_path):
                continue  # already stored
            block_hashes_to_store.append(block_hash)

        # Set up store spec
        store_spec = SharedStorageLoadStoreSpec(block_hashes_to_store)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=[],  # no eviction needed
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        """
        For shared storage, storing is stateless - no action needed.
        If storing failed, clean up partial files.
        """
        if not success:
            for block_hash in block_hashes:
                path = StorageOffloadingHandler.get_file_name(self.base_path, block_hash)
                if os.path.exists(path):
                    os.remove(path)
