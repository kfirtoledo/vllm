# tests/v1/offloading/test_shared_storage.py
import math
import os
import time
import struct
import xxhash
import torch
import pytest

from vllm.v1.offloading.worker.shared_storage import (
    generate_put_transfer_function,
    generate_get_transfer_function,
    get_file_name,
    get_kv_cache_base_path,
)

from vllm.v1.offloading.mediums import SharedStorageLoadStoreSpec, GPULoadStoreSpec

# ----------------------------
# Helpers functions
# ----------------------------

def create_dummy_kv_tensors(num_layers: int, num_blocks: int, block_size: int, num_heads: int, head_size: int, dtype: torch.dtype, seed: int = 42):
    """Create dummy KV cache tensors [K, V] for all layers with shape (2, num_blocks, num_heads, block_size, head_size)."""
    torch.manual_seed(seed)
    shape = (2, num_blocks, num_heads, block_size, head_size)
    return [torch.rand(shape, dtype=dtype) for _ in range(num_layers)]

def get_prefix_hash(token_ids):
    """Generate a stable 64-bit hash for a list of token IDs by packing each as uint32."""
    buf = bytearray()
    for t in token_ids:
        buf += struct.pack("<I", int(t) & 0xFFFFFFFF)
    return xxhash.xxh64(buf).intdigest()

def make_gpu_specs(block_ids):
    """Create GPULoadStoreSpec objects for the given block IDs."""
    return [GPULoadStoreSpec(block_id=int(b)) for b in block_ids]

def make_storage_specs(num_files: int):
    """Create SharedStorageLoadStoreSpec objects and their hashes for a given number of files."""
    ranges = [(100 + i * 100, 117 + i * 100) for i in range(num_files)]
    hashes = [get_prefix_hash(range(a, b)) for (a, b) in ranges]
    return [SharedStorageLoadStoreSpec(block_hash=h) for h in hashes], hashes

def cleanup_files(model_name, tp_size, tp_rank, dtype, root_dir, block_hashes):
    """Remove existing files for the provided block hashes."""
    base_path = get_kv_cache_base_path(model_name, tp_size, tp_rank, dtype, root_dir)
    for h in block_hashes:
        path = get_file_name(base_path, h)
        if os.path.exists(path):
            os.remove(path)
    return base_path

def throughput_gbps(total_mb: float, seconds: float) -> float:
    """Calculate throughput in GB/s given MB transferred and elapsed seconds."""
    return float("inf") if seconds <= 0 else (total_mb / 1024.0) / seconds

def assert_blocks_equal(original_tensors, restored_tensors, block_ids):
    """Assert that restored blocks match the original blocks for the given block IDs."""
    for orig, restored in zip(original_tensors, restored_tensors):
        for b in block_ids:
            torch.testing.assert_close(orig[:, int(b)], restored[:, int(b)])

def total_block_size_mb(num_layers, num_heads, block_size, head_size, dtype, num_blocks):
    """Compute total block size in MB for the given model dimensions and number of blocks."""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    per_block_bytes = num_layers * 2 * num_heads * block_size * head_size * bytes_per_elem
    return (per_block_bytes * num_blocks) / (1024 * 1024)

def log_file_info(base_path, block_hashes):
    """Log information about the files corresponding to the given block hashes."""
    file_sizes = []
    for h in block_hashes:
        path = get_file_name(base_path, h)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            file_sizes.append(size_mb)
    num_files = len(file_sizes)
    return num_files, file_sizes

def roundtrip_once(*, model_name: str, tp_size: int, tp_rank: int, dtype: torch.dtype, root_dir: str, num_layers: int, num_blocks: int, block_size: int, num_heads: int, head_size: int, block_ids: list[int], group_size: int):
    """Perform a PUT and GET roundtrip for the specified configuration and validate results."""

    original = create_dummy_kv_tensors(num_layers, num_blocks, block_size, num_heads, head_size, dtype)
    restored = [torch.zeros_like(t) for t in original]
    gpu_specs = make_gpu_specs(block_ids)
    num_files = math.ceil(len(block_ids) / group_size)
    storage_specs, block_hashes = make_storage_specs(num_files)
    base_path = cleanup_files(model_name, tp_size, tp_rank, dtype, root_dir, block_hashes)

    # PUT phase: write KV blocks to shared storage
    put_fn = generate_put_transfer_function(model_name=model_name, tp_size=tp_size, tp_rank=tp_rank, src_tensors=original, src_block_size=group_size, dtype=dtype, root_dir=root_dir)
    start_put = time.time()
    ok_put = put_fn((gpu_specs, storage_specs))
    dur_put = time.time() - start_put
    assert ok_put, "PUT failed"
    for h in block_hashes:
        assert os.path.exists(get_file_name(base_path, h)), "missing file after PUT"

    # GET phase: load KV blocks back from shared storage
    get_fn = generate_get_transfer_function(dst_tensors=restored, dst_block_size=group_size, model_name=model_name, tp_size=tp_size, tp_rank=tp_rank, dtype=dtype, root_dir=root_dir)
    start_get = time.time()
    ok_get = get_fn((storage_specs, gpu_specs))
    dur_get = time.time() - start_get
    assert ok_get, "GET failed"
    assert_blocks_equal(original, restored, block_ids)

    # Print results
    total_mb = total_block_size_mb(num_layers, num_heads, block_size, head_size, dtype, len(block_ids))
    file_size_mb = os.path.getsize(get_file_name(base_path, block_hashes[0])) / (1024 * 1024)
    num_files = len(block_hashes)
    print(
        f"[INFO] group={group_size} "
        f"PUT {dur_put:.4f}s ({throughput_gbps(total_mb, dur_put):.2f} GB/s), "
        f"GET {dur_get:.4f}s ({throughput_gbps(total_mb, dur_get):.2f} GB/s), "
        f"files={num_files}, sizes(MB)={file_size_mb:.2f} "
    )
# ----------------------------
# Test
# ----------------------------

@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_shared_storage_roundtrip_param(group_size: int, tmp_path):
    """Test roundtrip save/load for multiple group sizes using model-like dimensions."""
    model_name = "llama3-70b"
    tp_size = 2
    tp_rank = 0
    dtype = torch.float16
    root_dir = str(tmp_path)
    num_layers = 80
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 8
    block_ids = list(range(num_blocks))
    roundtrip_once(model_name=model_name, tp_size=tp_size, tp_rank=tp_rank, dtype=dtype, root_dir=root_dir, num_layers=num_layers, num_blocks=num_blocks, block_size=block_size, num_heads=num_heads, head_size=head_size, block_ids=block_ids, group_size=group_size)
