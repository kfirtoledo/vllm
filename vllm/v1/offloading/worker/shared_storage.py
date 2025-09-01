import os
import time
import math
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
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
    root_dir: str = "/mnt/shared-kv",
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

def get_block_offset(dst_tensors: list[torch.Tensor]) -> int:
    """Get the byte offset for a given block tensor."""
    block_size_bytes = 0
    for tensor in dst_tensors:
        block = tensor[:, 0]  # Use block 0 to get the shape
        block_size_bytes += block.numel() * block.element_size()
    return block_size_bytes


def convert_tensors_to_bytes(
    src_tensors: List[torch.Tensor],
    block_ids_list: List[int],
) -> memoryview:
    """Serialize selected [K,V] blocks from all layers to a contiguous CPU buffer.

    Layout: for each block_id in order, for each layer tensor in order, write [2, heads, block, head_size].
    """
    blocks = []
    for block_id in block_ids_list:
        for tensor in src_tensors:
            blocks.append(tensor[:, block_id])  # [2, H, B, D]

    flat = torch.cat(blocks, dim=0)  # concat along first dim
    # Ensure contiguous and on CPU
    flat = flat.contiguous().detach().cpu()

    # If storing bf16, write as uint16 payload
    if flat.dtype == torch.bfloat16:
        flat = flat.view(torch.uint16)

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



def convert_bytes_to_tensors(
    buffer: bytes,
    dst_tensors: List[torch.Tensor],
    block_ids_list: List[int],
    block_offset: int,
    gpu_blocks_per_file: int
) -> None:
    """Restore bytes into destination tensors by block ids.

    Mirrors convert_tensors_to_bytes layout exactly. No offsets or padding.
    """
    reshaped_blocks: list[list[torch.Tensor]] = []  # [tensor_index][block_index]

    mv = memoryview(buffer)
    offset = 0
    assert len(block_ids_list) <= gpu_blocks_per_file # length should be smaller
    if len(block_ids_list) % gpu_blocks_per_file != 0:
        offset = block_offset *(gpu_blocks_per_file - len(block_ids_list)) # If not full blocks, start at the offset

    # Step 1: Parse and reshape blocks from buffer
    #start_time = time.time()
    for block_id in block_ids_list:
        tensor_blocks = []
        for tensor in dst_tensors:
            block = tensor[:, block_id]
            num_bytes = block.numel() * block.element_size()
            mv_block = mv[offset:offset + num_bytes]
            offset += num_bytes

            np_dtype = torch_dtype_to_numpy(block.dtype)
            np_arr = np.frombuffer(mv_block, dtype=np_dtype).reshape(block.shape)
            src = torch.from_numpy(np_arr)

            if block.dtype is torch.bfloat16 and src.dtype != torch.bfloat16:
                src = src.view(torch.bfloat16)

            tensor_blocks.append(src)
        reshaped_blocks.append(tensor_blocks)
    #end_time = time.time() - start_time
    #print(f"convert_bytes_to_tensors: step 1 - parse and reshape blocks: {end_time} seconds")
    # Step 2: Copy into destination tensors
    #start_time = time.time()
    for i, block_id in enumerate(block_ids_list):
        for j, tensor in enumerate(dst_tensors):
            block = tensor[:, block_id]
            src = reshaped_blocks[i][j]

            if src.dtype != block.dtype or src.device != block.device:
                src = src.to(device=block.device, dtype=block.dtype, non_blocking=True)

            block.copy_(src)
    #end_time = time.time() - start_time
    #print(f"convert_bytes_to_tensors: step 2 - copy reshaped blocks into destination tensors: {end_time} seconds")

def write_buffer_to_file(target_file: Path, buffer: memoryview) -> None:
    """Atomically write a buffer to target_file on POSIX systems."""
    tmp_file_path = target_file.with_suffix(".tmp")
    with open(tmp_file_path, "wb") as f:
        f.write(buffer)
        f.flush()               # Flush Python's internal I/O buffers to the OS
        os.fsync(f.fileno())    # Force the OS to flush its write cache to disk for durability
    os.replace(tmp_file_path, target_file)

import mmap
def read_file_to_bytes(path: Path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # return both so caller can mm.close(); f.close() after conversion
    return mm, f

# -----------------------------
# Flexible PUT (thread pool)
# -----------------------------

def generate_put_transfer_function(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    src_tensors: List[torch.Tensor],
    gpu_blocks_per_file: int,  # number of GPU blocks per file
    dtype: torch.dtype,
    root_dir: str,
    max_concurrency: int,
) -> TransferFunction:
    """Create a TransferFunction that writes blocks to shared storage using a thread pool."""
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
            #start_put = time.time()
            buf = convert_tensors_to_bytes(src_tensors, block_ids)
            #end_put = time.time() - start_put
            #print(f"Time taken to convert tensors to bytes for {target_file}: {end_put} seconds")

            #start_write = time.time()
            write_buffer_to_file(target_file, buf)
            #end_write = time.time() - start_write
            #print(f"Time taken to write file {target_file}: {end_write} seconds")
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
        """Entry point used by the worker to perform PUT."""
        src_specs, dst_specs = spec
        if not dst_specs:
            return True

        # Pack gpu_blocks_per_file source specs into each destination spec, in order.
        futs = []
        #start_transfer = time.time()
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
            #end_transfer = time.time() - start_transfer
            #print(f"Time taken to put transfer {end_transfer} seconds")
            return ok

    return transfer_function


# -----------------------------
# Flexible GET (thread pool)
# -----------------------------

def generate_get_transfer_function(
    dst_tensors: List[torch.Tensor],
    gpu_blocks_per_file: int,  # number of destination blocks per file
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
    root_dir: str,
    max_concurrency: int,
) -> TransferFunction:
    """Create a TransferFunction that reads blocks from shared storage using a thread pool."""
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    block_offset = get_block_offset(dst_tensors)


    def _read_group_sync(src_spec, block_ids: List[int]) -> bool:
        block_hash = src_spec.block_hash
        path = get_file_name(base_path, block_hash)

        try:
            start_put = time.time()
            buf, f = read_file_to_bytes(path)
            end_put = time.time()- start_put
            print(f"Time taken to read file {path}: {end_put} seconds")
        except Exception as e:
            logger.warning("GET read failed for %s: %r", path, e)
            return False

        try:
            #start_convert = time.time()
            convert_bytes_to_tensors(buf, dst_tensors, block_ids, block_offset, gpu_blocks_per_file)
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
        """Entry point used by the worker to perform GET."""
        src_specs, dst_specs = spec
        # Each src file corresponds to up to gpu_blocks_per_file destination blocks
        expected_src = math.ceil(len(dst_specs) / gpu_blocks_per_file) if dst_specs else 0
        assert len(src_specs) == expected_src, (
            f"Mismatch sizes of source spec {len(src_specs)} and destination specs {len(dst_specs)}, "
            f"gpu_blocks_per_file {gpu_blocks_per_file}"
        )

        if not src_specs:
            return True

        futs = []
        # The first group may be partial, so calculate its length
        first_len = len(dst_specs) % gpu_blocks_per_file or gpu_blocks_per_file
        start=0
        #start_transfer = time.time()
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
            #end_transfer = time.time() - start_transfer
            #print(f"Time taken to get transfer {end_transfer} seconds")
            return ok

    return transfer_function
