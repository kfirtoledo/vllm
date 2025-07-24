import asyncio
import os
import pytest
import torch
import time
import xxhash

from vllm.v1.offloading.worker.shared_storage import (
    generate_put_transfer_function,
    generate_get_transfer_function,
    get_file_name,
    get_kv_cache_base_path,
    HASH_NAME_INDEX
)
from vllm.v1.offloading.mediums import SharedStorageLoadStoreSpec, GPULoadStoreSpec

def create_dummy_kv_tensors(num_layers: int, num_blocks: int, block_size: int, num_heads: int, head_size: int, dtype: torch.dtype,seed=42):
    torch.manual_seed(seed) # Use a fixed seed for reproducibility
    shape = (2, num_blocks, num_heads, block_size, head_size)  # 2 for K and V
    return [torch.rand(shape, dtype=dtype) for _ in range(num_layers)]

def get_prefix_hash(token_ids: list[int]) -> int:
    return xxhash.xxh64(bytes(token_ids)).intdigest()


@pytest.mark.asyncio
async def test_shared_storage_roundtrip():
    model_name = "llama3-70b"
    tp_size = 2
    tp_rank = 0
    dtype = torch.float16
    root_dir = "/tmp/shared-kv-test"
    num_blocks = 4

    # Approximate config for llama3-70b
    num_layers = 80                 # LLaMA 70B has 80 transformer layers
    block_size = 16                 # 16 tokens per block (typical)
    num_heads = 64                  # Total attention heads
    head_size = 128                 # Hidden size = 8192 → head_size = 8192 / 64

    # Create original tensors
    original_tensors = create_dummy_kv_tensors(num_layers, num_blocks, block_size, num_heads, head_size, dtype)
    restored_tensors = [torch.zeros_like(t) for t in original_tensors]

    block_ids = [1, 2]
    gpu_specs = [GPULoadStoreSpec(block_id=b) for b in block_ids]

    block_hashs = [get_prefix_hash(range(100, 117)), get_prefix_hash(range(200, 217))]  # Generate hash IDs for each block ID
    storage_specs = [SharedStorageLoadStoreSpec(block_hash=h) for h in block_hashs]

    block_count = len(block_ids)
    block_size_mb =  num_layers * 2 * num_heads * block_size * head_size * torch.tensor([], dtype=dtype).element_size()/ (1024 * 1024)
    print(f"[INFO] block size: {block_size_mb:.2f} MB and total size for blocks {block_ids}: {block_count * block_size_mb:.2f} MB")
    file_name = get_file_name(get_kv_cache_base_path(model_name,tp_size, tp_rank, dtype, root_dir), block_hashs[HASH_NAME_INDEX])
    os.remove(file_name) if os.path.exists(file_name) else None  # Clean up any existing file

    # Generate and run save function
    put_specs = (gpu_specs,storage_specs)
    put_fn = generate_put_transfer_function(model_name, tp_size, tp_rank, original_tensors, dtype, root_dir)
    start_time = time.time()
    success_put = put_fn(put_specs)
    elapsed_put = time.time() - start_time
    print(f"[INFO] Put operation took {elapsed_put:.4f} seconds (throughput: {block_count * block_size_mb / (elapsed_put * 1024):.2f} GB/s)")
    assert success_put, "Put operation failed."
    print(f"[INFO] Successfully wrote to files: {file_name}")
    # Generate and run load function
    get_fn = generate_get_transfer_function(restored_tensors, model_name, tp_size, tp_rank, dtype, root_dir)
    get_specs = (storage_specs,gpu_specs)
    start_time = time.time()
    success_get = get_fn(get_specs)
    elapsed_get = time.time() - start_time
    print(f"[INFO] Get operation took {elapsed_get:.4f} seconds (throughput: {block_count * block_size_mb / (elapsed_get * 1024):.2f} GB/s)")
    assert success_get, "Get operation failed."

    # Check values
    for orig, restored in zip(original_tensors, restored_tensors):
        for block_id in block_ids:
            torch.testing.assert_close(orig[:, int(block_id)], restored[:, int(block_id)])
    print("Shared storage roundtrip test passed.")

if __name__ == "__main__":
    asyncio.run(test_shared_storage_roundtrip())
