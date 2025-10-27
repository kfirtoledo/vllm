# tests/v1/offloading/test_shared_storage.py
import math
import os
import time
import struct
import xxhash
import torch
import pytest
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm.v1.kv_offload.worker.shared_storage import (
    StorageOffloadingHandler,
    GPUStorageOffloadingHandler,
    StorageGPUOffloadingHandler,
)
from vllm.v1.kv_offload.mediums import SharedStorageLoadStoreSpec, GPULoadStoreSpec

TMP_DIR = "/tmp/shared-kv-test"

# ----------------------------
# Helpers functions
# ----------------------------

def create_dummy_kv_tensors(num_layers: int, num_blocks: int, block_size: int, num_heads: int, head_size: int, dtype: torch.dtype, seed: int = 42):
    """Create dummy KV cache tensors [K, V] for all layers with shape (2, num_blocks, num_heads, block_size, head_size)."""
    torch.manual_seed(seed)
    shape = (2, num_blocks, num_heads, block_size, head_size)
    return [torch.rand(shape, dtype=dtype, device="cuda") for _ in range(num_layers)]

def get_prefix_hash(token_ids):
    """Generate a stable 64-bit hash for a list of token IDs by packing each as uint32."""
    buf = bytearray()
    for t in token_ids:
        buf += struct.pack("<I", int(t) & 0xFFFFFFFF)
    digest_int = xxhash.xxh64(buf).intdigest()
    # Convert 64-bit int to 8-byte little-endian representation
    return BlockHash((digest_int & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))

def make_gpu_specs(block_ids):
    """Create GPULoadStoreSpec objects for the given block IDs."""
    return GPULoadStoreSpec(block_ids)

def make_storage_specs(num_files: int):
    """Create SharedStorageLoadStoreSpec objects and their hashes for a given number of files."""
    ranges = [(100 + i * 100, 117 + i * 100) for i in range(num_files)]
    hashes = [get_prefix_hash(range(a, b)) for (a, b) in ranges]
    return SharedStorageLoadStoreSpec(hashes), hashes

def cleanup_files(model_name, tp_size, tp_rank, dtype, root_dir, block_hashes):
    """Remove existing files for the provided block hashes."""
    base_path = StorageOffloadingHandler.get_kv_cache_base_path(model_name, tp_size, tp_rank, dtype, root_dir)
    for h in block_hashes:
        path = StorageOffloadingHandler.get_file_name(base_path, h)
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
        path = StorageOffloadingHandler.get_file_name(base_path, h)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            file_sizes.append(size_mb)
    num_files = len(file_sizes)
    return num_files, file_sizes


def wait_for(handler, job_id: int, timeout: float = 2.0):
    """Wait for a specific job in handler.get_finished() up to timeout seconds."""
    start = time.time()
    while time.time() - start < timeout:
        finished = handler.get_finished()
        for jid, ok in finished:
            if jid == job_id:
                return ok
        time.sleep(0.01)  # avoid busy-spin
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")

def roundtrip_once(*, model_name: str, tp_size: int, tp_rank: int, dtype: torch.dtype,
                   root_dir: str, num_layers: int, num_blocks: int,
                   block_size: int, num_heads: int, head_size: int,
                   read_block_ids: list[int], write_block_ids: list[int],
                   group_size: int, threads_per_gpu: int):

    original = create_dummy_kv_tensors(num_layers, num_blocks, block_size, num_heads, head_size, dtype)
    restored = [torch.zeros_like(t) for t in original]

    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / group_size)
    put_storage_specs, block_hashes = make_storage_specs(put_num_files)
    base_path = cleanup_files(model_name, tp_size, tp_rank, dtype, root_dir, block_hashes)

    # PUT phase
    put_handler = GPUStorageOffloadingHandler(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        src_tensors=original,
        gpu_blocks_per_file=group_size,
        dtype=dtype,
        threads_per_gpu=threads_per_gpu,
        root_dir=root_dir,
    )
    start_put = time.time()
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    ok_put = wait_for(put_handler, job_id=1, timeout=2.0)
    assert ok_put, "PUT failed"
    dur_put = time.time() - start_put
    for h in block_hashes:
        assert os.path.exists(StorageOffloadingHandler.get_file_name(base_path, h)), "missing file after PUT"

    # GET phase
    get_handler = StorageGPUOffloadingHandler(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        gpu_blocks_per_file=group_size,
        dst_tensors=restored,
        threads_per_gpu=threads_per_gpu,
        root_dir=root_dir,
    )
    start_get = time.time()
    get_gpu_specs = make_gpu_specs(read_block_ids)
    get_num_files = math.ceil(len(read_block_ids) / group_size)
    start_index = len(put_storage_specs.block_hashes) - get_num_files
    get_storage_spec = SharedStorageLoadStoreSpec(put_storage_specs.block_hashes[start_index:])
    get_handler.transfer_async(job_id=2, spec=(get_storage_spec, get_gpu_specs))
    ok_get = wait_for(get_handler, job_id=2, timeout=2.0)
    dur_get = time.time() - start_get
    assert ok_get, "GET failed"
    assert_blocks_equal(original, restored, read_block_ids)

    # Report
    write_total_mb = total_block_size_mb(num_layers, num_heads, block_size, head_size, dtype, len(write_block_ids))
    read_total_mb = total_block_size_mb(num_layers, num_heads, block_size, head_size, dtype, len(read_block_ids))
    file_size_mb = os.path.getsize(StorageOffloadingHandler.get_file_name(base_path, block_hashes[0])) / (1024 * 1024)
    num_files = len(block_hashes)
    print(
        f"[INFO] group={group_size} write blocks len: {len(write_block_ids)} read blocks len: {len(read_block_ids)} "
        f"PUT {dur_put:.4f}s ({throughput_gbps(write_total_mb, dur_put):.2f} GB/s), "
        f"GET {dur_get:.4f}s ({throughput_gbps(read_total_mb, dur_get):.2f} GB/s), "
        f"files={num_files}, sizes(MB)={file_size_mb:.2f} "
    )
# ----------------------------
# Test
# ----------------------------

@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("start_idx", [0, 3])  # 0 = full from start, 3 = partial first group (e.g., 3..7)
def test_shared_storage_roundtrip_param(group_size: int, start_idx: int):
    """Test roundtrip save/load for multiple group sizes using model-like dimensions."""
    model_name = "llama3-70b"
    tp_size = 2
    tp_rank = 1
    dtype = torch.float16
    root_dir = TMP_DIR
    num_layers = 80
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 8
    write_block_ids = list(range(num_blocks))
    read_block_ids = list(range(start_idx, num_blocks))
    threads_per_gpu = 8
    roundtrip_once(model_name=model_name, tp_size=tp_size, tp_rank=tp_rank, dtype=dtype, root_dir=root_dir, num_layers=num_layers,
                   num_blocks=num_blocks, block_size=block_size, num_heads=num_heads, head_size=head_size, read_block_ids=read_block_ids,
                   write_block_ids=write_block_ids, group_size=group_size,threads_per_gpu=threads_per_gpu)
