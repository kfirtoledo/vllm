import os
import time
import math
import mmap
import numpy as np
import torch

from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm.v1.offloading.worker.worker import TransferFunction, TransferSpec
from vllm.logger import init_logger

HASH_NAME_INDEX = -1  # Use the last spec's hash ID for the file name
logger = init_logger(__name__)


# -----------------------
# Paths and sharding
# -----------------------

def get_kv_cache_base_path(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
    root_dir: str = "/tmp/shared-kv",
) -> Path:
    """Return the base directory for KV cache files and ensure it exists."""
    dtype_str = str(dtype).replace("torch.", "")
    base_path = Path(f"{root_dir}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}")
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def get_file_name(base_path: Path, block_hash: int) -> Path:
    """Build a sharded file path for a given block hash."""
    block_hash_hex = f"{block_hash & ((1 << 64) - 1):016x}"  # 16 hex digits, unsigned
    subfolder1 = block_hash_hex[:8]
    subfolder2 = block_hash_hex[8:16]
    full_path = base_path / subfolder1 / subfolder2 / f"{block_hash_hex}.bin"
    os.makedirs(full_path.parent, exist_ok=True)
    return full_path


# -----------------------
# Synchronous helpers
# -----------------------

def convert_tensors_to_bytes(
    src_tensors: List[torch.Tensor],
    block_ids_list: List[int],
) -> memoryview:
    """Serialize selected [K,V] blocks from all layers to a contiguous CPU buffer."""
    blocks = []
    # Collect [K,V] slices for every (layer, block_id) into a list
    for block_id in block_ids_list:
        for tensor in src_tensors:
            blocks.append(tensor[:, block_id])  # [2, H, B, D]

    # Concatenate all selected [K,V] block slices into one big tensor along dim=0 → [2 * num_layers * num_blocks, H, B, D]
    flat = torch.cat(blocks, dim=0)
    flat = flat.contiguous().detach().cpu() # Ensure contiguous and on CPU

    if flat.dtype == torch.bfloat16: # If storing bf16, write as uint16 payload
        flat = flat.view(torch.uint16)

    # Zero-copy: expose tensor bytes as a NumPy view wrapped in memoryview for fast I/O
    return memoryview(flat.numpy())


def torch_dtype_to_numpy(dtype) -> np.dtype:
    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bfloat16: np.uint16,  # stored as uint16 on disk
    }
    return mapping[dtype]


# -----------------------
# New tensor-first restore using swap_blocks_multi_layer
# -----------------------

def convert_file_to_tensors_with_swap(
    buf: bytes,
    dst_tensors: List[torch.Tensor],
    block_ids_list: List[int],
    gpu_blocks_per_file: int,
    attn_backend,
) -> None:
    """Restore blocks from file into destination tensors using swap_blocks_multi_layer."""

    # Wrap raw buffer in a memoryview (zero-copy access to bytes)
    buf_mv = memoryview(buf)

    # Shape of one KV block (K and V): [2, num_heads, block_size, head_dim]
    block_shape = dst_tensors[0][:, 0].shape # (2, H, B, D)
    elems_per_block = np.prod(block_shape)  # number of elements in one block

    num_layers = len(dst_tensors)          # how many layers total
    num_blocks = gpu_blocks_per_file       # how many block IDs
    expected_size = elems_per_block * num_layers * num_blocks  # total elems in file

    # Map torch dtype → numpy dtype (e.g., float16 → np.float16)
    np_dtype = torch_dtype_to_numpy(dst_tensors[0].dtype)
    # Interpret raw buffer as 1D NumPy array (zero-copy view)
    np_arr = np.frombuffer(buf_mv, dtype=np_dtype)

    # Sanity check: file size must match what we expect
    full_block = len(block_ids_list) % gpu_blocks_per_file == 0
    assert len(block_ids_list) <= gpu_blocks_per_file # length should be smaller
    if np_arr.size != expected_size and full_block:
        raise ValueError(
            f"File has {np_arr.size} elements but expected {expected_size} "
            f"({num_blocks} blocks × {num_layers} layers × {elems_per_block} elems/block)"
        )

    # Reshape into [num_blocks, num_layers, 2, H, B, D] for easy indexing
    np_arr = np_arr.reshape(num_blocks, num_layers, *block_shape)
    torch_arr = torch.from_numpy(np_arr)  # Wrap as Torch tensor (still zero-copy)

    # Special case: if data was stored as uint16 (for bf16), reinterpret back to bf16
    if dst_tensors[0].dtype == torch.bfloat16 and torch_arr.dtype != torch.bfloat16:
        torch_arr = torch_arr.view(torch.bfloat16)

    # Calculate first block offset
    offset = 0
    if not full_block:
        offset = gpu_blocks_per_file - len(block_ids_list) # If not full blocks, start at the offset

    # Build list of per-layer tensors, each shaped [2, num_blocks, H, B, D]
    src_tensors: list[torch.Tensor] = []
    for i in range(num_layers):
        blocks_for_layer = []
        for b in range(offset,num_blocks):
            block_tensor = torch_arr[b, i]  # one block for this layer [2, H, B, D]
            blocks_for_layer.append(block_tensor)
        # Stack all blocks along dim=1 → [2, num_blocks, H, B, D]
        layer_tensor = torch.stack(blocks_for_layer, dim=1).contiguous()
        src_tensors.append(layer_tensor)

    # Mapping: (source block index, destination block index)

    src_to_dst = torch.tensor(
        [(i, bid) for i, bid in enumerate(block_ids_list)],
        dtype=torch.int64,
        device="cpu",
    )

    # Perform optimized block transfer (GPU kernel if available, else CPU)
    if torch.cuda.is_available():
        with torch.cuda.stream(torch.cuda.Stream()):  # async CUDA stream
            attn_backend.swap_blocks_multi_layer(src_tensors, dst_tensors, src_to_dst)
    else:
        attn_backend.swap_blocks_multi_layer(src_tensors, dst_tensors, src_to_dst)


# -----------------------
# File I/O helpers
# -----------------------
def write_buffer_to_file(target_file: Path, buffer: memoryview) -> None:
    tmp_file_path = target_file.with_suffix(".tmp")
    with open(tmp_file_path, "wb") as f:
        f.write(buffer)
        f.flush()             # Flush Python's internal I/O buffers to the OS
        os.fsync(f.fileno())  # Force the OS to flush its write cache to disk for durability
    os.replace(tmp_file_path, target_file)


def read_file_to_bytes(path: Path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return mm, f


# -----------------------------
# Flexible PUT (thread pool)
# -----------------------------

def generate_put_transfer_function(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    src_tensors: List[torch.Tensor],
    gpu_blocks_per_file: int,
    dtype: torch.dtype,
    root_dir: str,
    max_concurrency: int,
) -> TransferFunction:
    """Write blocks to shared storage using a thread pool."""
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    def _write_group_sync(dst_spec, block_ids: List[int]) -> bool:
        block_hash = dst_spec.block_hash
        target_file = get_file_name(base_path, block_hash)
        if os.path.exists(target_file):
            return True
        try:
            buf = convert_tensors_to_bytes(src_tensors, block_ids)
            write_buffer_to_file(target_file, buf)
            return True
        except Exception as e:
            logger.warning("PUT failed for %s: %r", target_file, e)
            try:
                if os.path.exists(target_file):
                    os.remove(target_file)
            except Exception:
                pass
            return False

    def transfer_function(spec: TransferSpec) -> bool:
        src_specs, dst_specs = spec
        if not dst_specs:
            return True

        futs = []
        with ThreadPoolExecutor(max_workers=max_concurrency or os.cpu_count()) as pool:
            for i, dst_spec in enumerate(dst_specs):
                start = i * gpu_blocks_per_file
                end = min((i + 1) * gpu_blocks_per_file, len(src_specs))
                if start >= len(src_specs):
                    break
                block_ids = [src_specs[j].block_id for j in range(start, end)]
                futs.append(pool.submit(_write_group_sync, dst_spec, block_ids))

            ok = True
            for f in as_completed(futs):
                ok = ok and bool(f.result())
            return ok

    return transfer_function


# -----------------------------
# Flexible GET (thread pool) using swap
# -----------------------------

def generate_get_transfer_function(
    dst_tensors: List[torch.Tensor],
    gpu_blocks_per_file: int,
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
    root_dir: str,
    max_concurrency: int,
    attn_backend=None,   ### CHANGED: pass backend in
) -> TransferFunction:
    """Read blocks from shared storage using a thread pool and swap_blocks_multi_layer."""
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    def _read_group_sync(src_spec, block_ids: List[int]) -> bool:
        path = get_file_name(base_path, src_spec.block_hash)

        try:
            #start_put = time.time()
            buf, f = read_file_to_bytes(path)
            #end_put = time.time()- start_put
            #print(f"Time taken to read file {path}: {end_put} seconds")
        except Exception as e:
            logger.warning("GET read failed for %s: %r", path, e)
            return False

        try:
            #t0 = time.time()
            convert_file_to_tensors_with_swap(buf, dst_tensors, block_ids, gpu_blocks_per_file, attn_backend)
            #print(f"Restored {len(block_ids)} blocks from {path} in {time.time()-t0:.4f}s")
            #end_convert = time.time() - start_convert
            #print(f"Time taken to convert bytes to tensors for {path}: {end_convert} seconds")
            return True
        except Exception as e:
             logger.warning("GET convert failed for %s: %r", path, e)
             return False
        finally:
            buf.close()
            f.close()


    def transfer_function(spec: TransferSpec) -> bool:
        src_specs, dst_specs = spec
        expected_src = math.ceil(len(dst_specs) / gpu_blocks_per_file) if dst_specs else 0
        assert len(src_specs) == expected_src, (
            f"Mismatch source spec {len(src_specs)} vs dst {len(dst_specs)}"
        )
        if not src_specs:
            return True

        futs = []
        first_len = len(dst_specs) % gpu_blocks_per_file or gpu_blocks_per_file
        start = 0
        with ThreadPoolExecutor(max_workers=max_concurrency or os.cpu_count()) as pool:
            for i, src_spec in enumerate(src_specs):
                size = first_len if i == 0 else gpu_blocks_per_file
                end = min(start + size, len(dst_specs))
                if start >= len(dst_specs):
                    break
                block_ids = [dst_specs[j].block_id for j in range(start, end)]
                futs.append(pool.submit(_read_group_sync, src_spec, block_ids))
                start += size

            ok = True
            for f in as_completed(futs):
                ok = ok and bool(f.result())
            return ok

    return transfer_function
