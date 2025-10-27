# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

import numpy as np
from typing import Iterable
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"
class SharedStorageLoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing KV blocks on shared storage.

    Accepts a collection of BlockHash values (ints or np.uint64).
    Stores them internally as np.ndarray(dtype=np.uint64).
    """

    def __init__(self, block_hashes: Iterable[BlockHash]):
        # Validate all items are bytes (BlockHash)
        block_hashes = list(block_hashes)
        for h in block_hashes:
            if not isinstance(h, (bytes, bytearray)):
                raise TypeError(f"Expected BlockHash (bytes), got {type(h)}")

        # Store directly as object array of bytes
        self.block_hashes = np.array(block_hashes, dtype=object)

    def __repr__(self) -> str:
        return repr(self.block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"