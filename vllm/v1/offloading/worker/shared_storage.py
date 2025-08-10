import torch
import aiofiles
import asyncio
import os
from typing import List, Optional
from pathlib import Path
from vllm.v1.offloading.worker.worker import TransferFunction, TransferSpec

from vllm.logger import init_logger

HASH_NAME_INDEX = -1  # Use the last spec's hash ID for the file name
logger = init_logger(__name__)

def get_kv_cache_base_path(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
    root_dir: str = "/mnt/shared-kv"
) -> Path:
    dtype_str = str(dtype).replace("torch.", "")
    base_path = Path(f"{root_dir}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}")
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def get_file_name(base_path: Path, block_hash: int) -> Path:
    block_hash_hex = f"{block_hash:x}"
    subfolder1 = block_hash_hex[:8]
    subfolder2 = block_hash_hex[8:16]
    full_path = base_path / subfolder1 / subfolder2 / f"{block_hash_hex}.bin"
    os.makedirs(full_path.parent, exist_ok=True)
    return full_path

# -----------------------
# Keep your sync helpers:
# -----------------------

def convert_tensors_to_bytes(
    src_tensors: List[torch.Tensor],
    block_ids_list: List[int],
) -> memoryview:
    blocks = []
    for block_id in block_ids_list:
        for tensor in src_tensors:
            block = tensor[:, block_id]
            blocks.append(block)
    flat = torch.cat(blocks, dim=0)
    flat = flat.contiguous().detach().cpu()
    return memoryview(flat.numpy())

def write_buffer_to_file(target_file: Path, buffer: memoryview):
    tmp_file_path = target_file.with_suffix(".tmp")
    with open(tmp_file_path, "wb") as f:
        f.write(buffer)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file_path, target_file)  # atomic on POSIX

def convert_bytes_to_tensors(
    buffer: bytes,
    dst_tensors: List[torch.Tensor],
    block_ids_list: List[int],
):
    offset = 0
    for block_id in block_ids_list:
        for tensor in dst_tensors:
            block = tensor[:, block_id]
            num_bytes = block.numel() * block.element_size()
            block_buffer = buffer[offset:offset + num_bytes]
            offset += num_bytes
            restored = torch.frombuffer(bytearray(block_buffer), dtype=block.dtype).view_as(block)
            block.copy_(restored)

# ----------------------------------------
# Async wrappers + loop runner for parallel
# ----------------------------------------

def _run_async_in_thread(coro) -> bool:
    """Run an async coroutine in a fresh event loop inside a background thread."""
    from threading import Thread
    result = {"ok": False, "exc": None}

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result["ok"] = loop.run_until_complete(coro)
        except Exception as e:
            result["exc"] = e
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if result["exc"]:
        raise result["exc"]
    return result["ok"]

async def _to_bytes_async(src_tensors: List[torch.Tensor], block_ids: List[int]) -> memoryview:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: convert_tensors_to_bytes(src_tensors, block_ids))

async def _from_bytes_async(buffer: bytes, dst_tensors: List[torch.Tensor], block_ids: List[int]) -> None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: convert_bytes_to_tensors(buffer, dst_tensors, block_ids))

async def _write_file_async(target: Path, buf: memoryview) -> None:
    # Keep your write_buffer_to_file helper, but run it off the loop for parallelism
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: write_buffer_to_file(target, buf))

async def _read_file_async(path: Path) -> bytes:
    # Use aiofiles for concurrent reads
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
    src_block_size: int,                       # grouping on the source side
    dtype: torch.dtype = torch.float16,
    root_dir: str = "/mnt/shared-kv",
    *,
    max_concurrency: int = 8,
) -> TransferFunction:
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    async def _write_group(dst_spec, block_ids: List[int], sem: asyncio.Semaphore) -> bool:
        block_hash = dst_spec.block_hash
        target_file = get_file_name(base_path, block_hash)

        if os.path.exists(target_file):
            return True

        async with sem:
            try:
                buf = await _to_bytes_async(src_tensors, block_ids)
                await _write_file_async(target_file, buf)
                return True
            except Exception as e:
                logger.warning("PUT failed for %s: %r", target_file, e)
                try:
                    if os.path.exists(target_file):
                        os.remove(target_file)
                except Exception:
                    pass
                return False

    async def _main_put(src_specs, dst_specs) -> bool:
        sem = asyncio.Semaphore(max_concurrency)
        tasks = []

        # Group src_block_size source specs into each destination spec
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
    dst_block_size: int,                      # grouping on the destination side
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype = torch.float16,
    root_dir: str = "/mnt/shared-kv",
    *,
    max_concurrency: int = 8,
) -> TransferFunction:
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    async def _read_group(src_spec, block_ids: List[int], sem: asyncio.Semaphore) -> bool:
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
        sem = asyncio.Semaphore(max_concurrency)
        tasks = []

        # Each src_spec file restores dst_block_size destination blocks
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
        src_specs, dst_specs = spec
        try:
            return _run_async_in_thread(_main_get(src_specs, dst_specs))
        except Exception as e:
            logger.warning("GET transfer failed: %r", e)
            return False

    return transfer_function