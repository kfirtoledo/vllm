import aiofiles
import asyncio
import numpy as np
import torch
import os
from typing import List, Optional
from pathlib import Path
from vllm.v1.offloading.worker.worker import TransferFunction, TransferSpec

from vllm.logger import init_logger

HASH_NAME_INDEX = -1  # Use the last spec's hash ID for the file name
MAX_CONCURRENCY = 100  # Default max concurrency for file operations
logger = init_logger(__name__)


def get_kv_cache_base_path(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
    root_dir: str = "/mnt/shared-kv",
) -> Path:
    """Return the base directory for KV cache files and ensure it exists.

    Args:
        model_name: Model identifier used to namespace files.
        tp_size: Total tensor parallel world size.
        tp_rank: Current tensor parallel rank.
        dtype: Torch dtype for the tensors saved under this path.
        root_dir: Root folder for shared storage.

    Returns:
        Path to a per-model, per-rank, per-dtype directory.
    """
    dtype_str = str(dtype).replace("torch.", "")
    base_path = Path(
        f"{root_dir}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}"
    )
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def get_file_name(base_path: Path, block_hash: int) -> Path:
    """Build a sharded file path for a given block hash.

    Args:
        base_path: Directory returned by get_kv_cache_base_path.
        block_hash: 64-bit integer that keys the block content.

    Returns:
        Full path to a file like base/aaaaaaaa/bbbbbbbb/aaaaaaaa bbbbbbbb.bin
        Two-level sharding keeps directories small.
    """
    block_hash_hex = f"{block_hash:x}"
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
    """Serialize selected [K, V] blocks from all layers to bytes.

    Args:
        src_tensors: List of per-layer tensors with shape
            [2, num_blocks, num_heads, block_size, head_size].
        block_ids_list: Block indices to extract in order.

    Returns:
        A memoryview over a NumPy buffer backed by CPU memory.
        Order is by block id outer loop, then by layer.
    """
    blocks = []
    for block_id in block_ids_list:
        for tensor in src_tensors:
            # Select the full [K, V] slice for this block across heads
            block = tensor[:, block_id]
            blocks.append(block)
    # Flatten across layers and K/V
    flat = torch.cat(blocks, dim=0)
    flat = flat.contiguous().detach().cpu()
    return memoryview(flat.numpy())


def write_buffer_to_file(target_file: Path, buffer: memoryview) -> None:
    """Atomically write a buffer to target_file on POSIX systems.

    Args:
        target_file: Destination path to write.
        buffer: Bytes-like memoryview to persist.

    Notes:
        Writes to a .tmp file then renames to avoid partial files on crash.
    """
    tmp_file_path = target_file.with_suffix(".tmp")
    with open(tmp_file_path, "wb") as f:
        f.write(buffer)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file_path, target_file)



def convert_bytes_to_tensors(
    buffer: bytes,
    dst_tensors: List[torch.Tensor],
    block_ids_list: List[int],
) -> None:
    """Restore bytes into the given destination tensors by block id."""
    mv = memoryview(buffer)  # zero-copy view over the whole file
    offset = 0

    for block_id in block_ids_list:
        for tensor in dst_tensors:
            block = tensor[:, block_id]
            num_bytes = block.numel() * block.element_size()

            # slice the memoryview without copying
            mv_block = mv[offset:offset + num_bytes]
            offset += num_bytes

            # interpret bytes as numpy with the matching dtype
            np_dtype = torch_dtype_to_numpy(block.dtype)
            np_arr = np.frombuffer(mv_block, dtype=np_dtype).reshape(block.shape)

            # make a torch tensor from the numpy view (CPU)
            src = torch.from_numpy(np_arr)

            # handle bf16 if you store it as uint16 on disk
            if block.dtype is torch.bfloat16 and src.dtype != torch.bfloat16:
                src = src.view(torch.bfloat16)

            # move to the right dtype and device, then copy into place
            if src.dtype != block.dtype or src.device != block.device:
                src = src.to(device=block.device, dtype=block.dtype, non_blocking=True)

            block.copy_(src)

def torch_dtype_to_numpy(dtype) -> np.dtype:
    mapping = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int8:    np.int8,
        torch.int16:   np.int16,
        torch.int32:   np.int32,
        torch.int64:   np.int64,
        # If you support bf16, store as uint16 on disk, then view to bfloat16 after
        torch.bfloat16: np.uint16,
    }
    return mapping[dtype]

# ----------------------------------------
# Async wrappers and loop runner
# ----------------------------------------

def _run_async_in_thread(coro) -> bool:
    """Run an async coroutine in a new event loop inside a background thread.

    Args:
        coro: Awaitable object to run to completion.

    Returns:
        True on success. False only if inner code returns False.

    Raises:
        Propagates any exception raised inside the coroutine.
    """
    from threading import Thread

    result = {"ok": False, "exc": None}

    def _runner():
        loop = asyncio.new_event_loop()  #Create a brand-new asyncio event loop for this background thread
        try:
            asyncio.set_event_loop(loop)  # Set the new loop as the current event loop for this thread
            result["ok"] = loop.run_until_complete(coro)
        except Exception as e:
            result["exc"] = e
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = Thread(target=_runner, daemon=True) # Create a background daemon thread that runs _runner()
    t.start() # Start the thread — the coroutine begins executing in the new event loop
    t.join() # Wait until the background thread finishes

    if result["exc"]:
        raise result["exc"]

    return result["ok"]


async def _to_bytes_async(src_tensors: List[torch.Tensor], block_ids: List[int]) -> memoryview:
    """Serialize tensors off the event loop using a thread pool.

    Args:
        src_tensors: Source tensors to serialize.
        block_ids: Block ids to extract.

    Returns:
        Memoryview with serialized bytes.
    """
    loop = asyncio.get_running_loop()  # Get the currently running asyncio event loop for this task
    return await loop.run_in_executor( # Schedule a blocking CPU-bound function to run in a background thread
        None, lambda: convert_tensors_to_bytes(src_tensors, block_ids)
    )


async def _from_bytes_async(
    buffer: bytes, dst_tensors: List[torch.Tensor], block_ids: List[int]
) -> None:
    """Deserialize bytes into destination tensors off the event loop.

    Args:
        buffer: Bytes returned by file read.
        dst_tensors: Destination tensors to write.
        block_ids: Block ids to fill.
    """
    loop = asyncio.get_running_loop()  # Get the currently running asyncio event loop for this task
    return await loop.run_in_executor( # Schedule a blocking CPU-bound function to run in a background thread
        None, lambda: convert_bytes_to_tensors(buffer, dst_tensors, block_ids)
    )


async def _write_file_async(target: Path, buf: memoryview) -> None:
    """Write buf to target atomically using aiofiles - short and clear."""
    tmp = target.with_suffix(".tmp")
    async with aiofiles.open(tmp, "wb") as f:
        await f.write(buf)
    os.replace(tmp, target)

async def _read_file_async(path: Path) -> bytes:
    """Read a file using aiofiles to allow concurrency.

    Args:
        path: File path to read.

    Returns:
        File content as bytes.
    """
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


# -----------------------------
# Flexible PUT (supports grouping)
# -----------------------------

def generate_put_transfer_function(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    src_tensors: List[torch.Tensor],
    src_block_size: int,  # number of GPU blocks per file
    dtype: torch.dtype = torch.float16,
    root_dir: str = "/mnt/shared-kv",
    *,
    max_concurrency: int = MAX_CONCURRENCY,
) -> TransferFunction:
    """Create a TransferFunction that writes blocks to shared storage.

    Args:
        model_name: Model identifier for path namespacing.
        tp_size: Tensor parallel world size.
        tp_rank: Current tensor parallel rank.
        src_tensors: Per-layer KV tensors to pull blocks from.
        src_block_size: Number of source blocks to group into one file.
        dtype: Data type used to choose the directory.
        root_dir: Root folder for shared storage.
        max_concurrency: Max concurrent file tasks.

    Returns:
        A callable that matches TransferFunction(spec) -> bool.
    """
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    async def _write_group(dst_spec, block_ids: List[int], sem: asyncio.Semaphore) -> bool:
        """Write one grouped file for a destination spec.

        Args:
            dst_spec: Storage spec that holds the block_hash.
            block_ids: Source block ids to group into this file.
            sem: Concurrency limiter.
        """
        block_hash = dst_spec.block_hash
        target_file = get_file_name(base_path, block_hash)

        # Idempotent: if file exists, skip work and return success.
        if os.path.exists(target_file):
            return True

        async with sem:
            try:
                buf = await _to_bytes_async(src_tensors, block_ids)
                await _write_file_async(target_file, buf)
                return True
            except Exception as e:
                logger.warning("PUT failed for %s: %r", target_file, e)
                # Best effort cleanup of partial file.
                try:
                    if os.path.exists(target_file):
                        os.remove(target_file)
                except Exception:
                    pass
                return False

    async def _main_put(src_specs, dst_specs) -> bool:
        """Plan and execute grouped writes in parallel.

        Args:
            src_specs: List of GPU specs with .block_id fields.
            dst_specs: List of storage specs with .block_hash fields.
        """
        sem = asyncio.Semaphore(max_concurrency)
        tasks = []

        # Pack src_block_size source specs into each destination spec.
        for i, dst_spec in enumerate(dst_specs):
            start = i * src_block_size
            end = min((i + 1) * src_block_size, len(src_specs))
            if start >= len(src_specs):
                break
            block_ids = [src_specs[j].block_id for j in range(start, end)]
            tasks.append(_write_group(dst_spec, block_ids, sem))

        if not tasks:
            return True

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results)

    def transfer_function(spec: TransferSpec) -> bool:
        """Entry point used by the worker to perform PUT.

        Args:
            spec: Tuple of (src_specs, dst_specs).

        Returns:
            True if all grouped writes succeeded.
        """
        src_specs, dst_specs = spec
        try:
            return _run_async_in_thread(_main_put(src_specs, dst_specs))
        except Exception as e:
            logger.warning("PUT transfer failed: %r", e)
            return False

    return transfer_function


# -----------------------------
# Flexible GET (supports grouping)
# -----------------------------

def generate_get_transfer_function(
    dst_tensors: List[torch.Tensor],
    dst_block_size: int,  # number of destination blocks per file
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype = torch.float16,
    root_dir: str = "/mnt/shared-kv",
    *,
    max_concurrency: int = MAX_CONCURRENCY,
) -> TransferFunction:
    """Create a TransferFunction that reads blocks from shared storage.

    Args:
        dst_tensors: Per-layer KV tensors to write restored blocks into.
        dst_block_size: Number of destination blocks each file provides.
        model_name: Model identifier for path namespacing.
        tp_size: Tensor parallel world size.
        tp_rank: Current tensor parallel rank.
        dtype: Data type used to choose the directory.
        root_dir: Root folder for shared storage.
        max_concurrency: Max concurrent file tasks.

    Returns:
        A callable that matches TransferFunction(spec) -> bool.
    """
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    async def _read_group(src_spec, block_ids: List[int], sem: asyncio.Semaphore) -> bool:
        """Read one grouped file and fill the destination blocks.

        Args:
            src_spec: Storage spec that holds the block_hash.
            block_ids: Destination block ids to fill from this file.
            sem: Concurrency limiter.
        """
        block_hash = src_spec.block_hash
        path = get_file_name(base_path, block_hash)

        async with sem:
            try:
                buf = await _read_file_async(path)
            except Exception as e:
                logger.warning("GET read failed for %s: %r", path, e)
                return False

            try:
                await _from_bytes_async(buf, dst_tensors, block_ids)
                return True
            except Exception as e:
                logger.warning("GET convert failed for %s: %r", path, e)
                return False

    async def _main_get(src_specs, dst_specs) -> bool:
        """Plan and execute grouped reads in parallel.

        Args:
            src_specs: List of storage specs with .block_hash fields.
            dst_specs: List of GPU specs with .block_id fields.
        """
        sem = asyncio.Semaphore(max_concurrency)
        tasks = []

        # Each src_spec file restores dst_block_size destination blocks.
        for i, src_spec in enumerate(src_specs):
            start = i * dst_block_size
            end = min((i + 1) * dst_block_size, len(dst_specs))
            if start >= len(dst_specs):
                break
            block_ids = [dst_specs[j].block_id for j in range(start, end)]
            tasks.append(_read_group(src_spec, block_ids, sem))

        if not tasks:
            return True

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results)

    def transfer_function(spec: TransferSpec) -> bool:
        """Entry point used by the worker to perform GET.

        Args:
            spec: Tuple of (src_specs, dst_specs).

        Returns:
            True if all grouped reads and conversions succeeded.
        """
        src_specs, dst_specs = spec
        try:
            return _run_async_in_thread(_main_get(src_specs, dst_specs))
        except Exception as e:
            logger.warning("GET transfer failed: %r", e)
            return False

    return transfer_function
