import ctypes
import torch
import aiofiles
import aiofiles.os
import os
from typing import List, Tuple, Type, Callable
from pathlib import Path
from vllm.v1.offloading.worker.worker import TransferFunction, TransferSpec

from vllm.logger import init_logger

HASH_NAME_INDEX = -1 # Use the last spec's hash ID for the file name
logger = init_logger(__name__)

def get_kv_cache_base_path(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
    root_dir: str = "/mnt/shared-kv"
) -> Path:
    """
    Returns the base directory path for saving KV cache files for a specific model,
    tensor parallel configuration, and dtype. Creates the directory if it doesn't exist.

    Args:
        model_name: Name of the model (e.g., "llama3-70b").
        tp_size: Total number of tensor parallel ranks.
        tp_rank: The current tensor parallel rank (e.g., 0, 1, ...).
        dtype: Data type of the KV cache tensors (e.g., torch.float16).
        root_dir: Root directory for KV cache storage (default: "/mnt/shared-kv").

    Returns:
        Path object to the specific directory.
    """
    dtype_str = str(dtype).replace("torch.", "")
    base_path = Path(f"{root_dir}/{model_name}/tp_{tp_size}/rank_{tp_rank}/{dtype_str}")
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def get_file_name(base_path: Path, block_hash: int) -> Path:
    """
    Returns the full file path with subfolders created based on the first 4 bytes of the block hash.

    Args:
        base_path: The base directory where the file resides.
        block_hash: The hash of the block (as an integer).

    Returns:
        Path object representing the full file path.
    """
    # Convert the block hash to hexadecimal
    block_hash_hex = f"{block_hash:x}"

    # Extract the first 4 bytes (first 8 hex characters) from the block hash
    subfolder1 = block_hash_hex[:8]
    subfolder2 = block_hash_hex[8:16]

    # Create the full path with two subfolders and the file name
    full_path = base_path / subfolder1 / subfolder2 / f"{block_hash_hex}.bin"

    # Create the directories if they don't exist
    os.makedirs(full_path.parent, exist_ok=True)

    return full_path


def convert_tensors_to_bytes(
    src_tensors: List[torch.Tensor],
    block_ids_list: List[int],
) -> bytes:
    """
    Converts the same block IDs from all layers in src_tensors into bytes.

    Args:
        src_tensors: List of per-layer tensors.
        block_ids_list: List of block indices to extract from each layer.

    Returns:
        Bytes of all (layer, block_id) slices concatenated together.
    """
    blocks = []

    for block_id in block_ids_list:
        for tensor in src_tensors:
            # Extract the [K, V] block (dim=0) for the given block ID from each tensor (layer)
            block = tensor[:, block_id]
            blocks.append(block)

    # Concatenate all blocks along the 0th dimension (K/V axis) to form a flat sequence for serialization
    flat = torch.cat(blocks, dim=0)
    # Ensure contiguous memory layout, detach from computation graph, move to CPU, and serialize to bytes
    flat = flat.contiguous().detach().cpu()
    # Convert the tensor to bytes directly from the contiguous memory
    flat = memoryview(flat.numpy())
    return flat


def write_buffer_to_file(target_file: Path, buffer: memoryview):
    tmp_file_path = target_file.with_suffix('.tmp')
    with open(tmp_file_path, "wb") as f:
            f.write(buffer)
    os.rename(tmp_file_path, target_file)

def generate_put_transfer_function(
    model_name: str,
    tp_size: int,
    tp_rank: int,
    src_tensors: List[torch.Tensor],
    dtype: torch.dtype = torch.float16,
    root_dir: str = "/mnt/shared-kv",
) -> TransferFunction:
    """
    Generate a function that transfers (saves) KV cache blocks to file.

    Args:
        model_name: Name of the model (e.g., "llama3-70b").
        tp_size: Total number of tensor parallel ranks.
        tp_rank: Current tensor parallel rank.
        src_tensors: List of tensors for each layer.
        dtype: Data type to save as.
        root_dir: Root directory for KV cache storage.

    Returns:
        A function that saves the specified KV cache blocks to file.
    """
    base_path = get_kv_cache_base_path(
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dtype=dtype,
        root_dir=root_dir,
    )

    def transfer_function(spec: TransferSpec) -> bool:
        src_specs, dst_specs = spec
        block_id_list = [spec.block_id for spec in src_specs]
        block_hash = dst_specs[HASH_NAME_INDEX].block_hash # Using the hash ID from the last spec
        if not block_hash:
            print("No hash ID provided in the specs.")
            return False
        target_file = get_file_name(base_path, block_hash)

        #Check if the target file already exists
        if os.path.exists(target_file):
            print(f"File {target_file} already exists. Skipping write operation.")
            return True

        # Convert tensors to bytes
        buffer = convert_tensors_to_bytes(src_tensors, block_id_list)
        try:
            write_buffer_to_file(target_file, buffer)
        except Exception as e:
            print(f"[Error] Failed to write file {target_file}: {e}")
            # Clean up
            if os.path.exists(target_file):
                os.remove(target_file)
            return False

        return True

    return transfer_function

def convert_bytes_to_tensors(
    buffer: bytes,
    dst_tensors: List[torch.Tensor],
    block_ids_list: List[int],
):
    """
    Copies raw bytes into the same block IDs across all layers of dst_tensors.

    Args:
        buffer: Raw byte content (as written by convert_tensors_to_bytes).
        dst_tensors: Destination tensors (per-layer) to copy into.
        block_ids_list: Block IDs that were saved and should be restored.
    """
    offset = 0
    for block_id in block_ids_list:
        for tensor in dst_tensors:
            block = tensor[:, block_id] # Get the [K, V] block at the specified block_id across both K and V (dim=0)
            num_bytes = block.numel() * block.element_size() # Calculate how many bytes this block occupies
            block_buffer = buffer[offset:offset + num_bytes]
            offset += num_bytes  # Update offset for the next block

            restored = torch.frombuffer(bytearray(block_buffer), dtype=block.dtype).view_as(block) # Convert raw bytes back into a tensor with the same shape and dtype

            block.copy_(restored) # Copy the restored values into the original block location


def generate_get_transfer_function(
    dst_tensors: List[torch.Tensor],
    model_name: str,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype = torch.float16,
    root_dir: str = "/mnt/shared-kv",
) -> TransferFunction:
    """
    Generate a function that loads KV cache blocks from shared storage.

    Args:
        dst_tensors: List of tensors to load into.
        model_name: Name of the model (used in storage path).
        tp_size: Total tensor parallel world size.
        tp_rank: Current TP rank.
        dtype: Data type of the KV tensors.
        root_dir: Base path for storage (e.g., shared volume).

    Returns:
        A transfer function that loads KV cache blocks.
    """
    base_path = get_kv_cache_base_path(
        dtype=dtype,
        model_name=model_name,
        tp_size=tp_size,
        tp_rank=tp_rank,
        root_dir=root_dir,
    )

    def transfer_function(spec: TransferSpec) -> bool:
        src_spec, dst_specs = spec
        block_id_list = [spec.block_id for spec in dst_specs]
        block_hash = src_spec[HASH_NAME_INDEX].block_hash # Using the hash ID from the last spec
        if not block_hash:
            print("No hash ID provided in the specs.")
            return False
        target_file = get_file_name(base_path, block_hash)

        try:
            with open(target_file, "rb") as f:
                buffer = f.read()
            convert_bytes_to_tensors(buffer, dst_tensors, block_id_list)
        except Exception as e:
            print(f"[Error] Transfer failed: {e}")
            return False

        return True

    return transfer_function