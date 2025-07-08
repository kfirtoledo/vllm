# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.offloading.abstract import (LoadStoreSpec, OffloadingEvent,
                                         PrepareStoreOutput)
from vllm.v1.offloading.cpu import CPUBackend
from vllm.v1.offloading.lru_manager import LRUOffloadingManager
from vllm.v1.offloading.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.offloading.worker.cpu import (create_cpu_tensors,
                                           generate_tensors_transfer_function)


@dataclass
class ExpectedPrepareStoreOutput:
    block_hashes_to_store: list[int]
    store_block_ids: list[int]
    block_hashes_evicted: list[int]


def verify_store_output(
        prepare_store_output: Optional[PrepareStoreOutput],
        expected_prepare_store_output: ExpectedPrepareStoreOutput):
    assert prepare_store_output is not None
    assert (prepare_store_output.block_hashes_to_store ==
            expected_prepare_store_output.block_hashes_to_store)
    assert (prepare_store_output.block_hashes_evicted ==
            expected_prepare_store_output.block_hashes_evicted)
    assert (len(prepare_store_output.store_specs) == len(
        expected_prepare_store_output.store_block_ids))
    for store_spec, expected_store_block_id in zip(
            prepare_store_output.store_specs,
            expected_prepare_store_output.store_block_ids):
        assert isinstance(store_spec, CPULoadStoreSpec)
        assert store_spec.block_id == expected_store_block_id


def verify_load_output(prepare_load_output: list[LoadStoreSpec],
                       expected_prepare_load_output: list[int]):
    for load_spec, expected_block_id in zip(prepare_load_output,
                                            expected_prepare_load_output):
        assert isinstance(load_spec, CPULoadStoreSpec)
        assert load_spec.block_id == expected_block_id


def verify_events(events: Generator[OffloadingEvent, None, None],
                  block_size: int,
                  expected_stores: tuple[set[int], ...] = (),
                  expected_evictions: tuple[set[int], ...] = ()):
    stores: list[set[int]] = []
    evictions: list[set[int]] = []
    for event in events:
        assert event.medium == CPULoadStoreSpec.medium()
        assert event.block_size == block_size
        if event.removed:
            evictions.append(set(event.block_hashes))
        else:
            stores.append(set(event.block_hashes))

    assert tuple(evictions) == expected_evictions
    assert tuple(stores) == expected_stores


def test_cpu_manager():
    """
    Tests LRUOffloadingManager with a CPUBackend.
    """
    # initialize a CPU backend with a capacity of 4 blocks
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    cpu_manager = LRUOffloadingManager(cpu_backend, enable_events=True)

    # prepare store [1, 2]
    prepare_store_output = cpu_manager.prepare_store([1, 2])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[1, 2],
            store_block_ids=[0, 1],
            block_hashes_evicted=[],
        ))

    # lookup [1, 2] -> not ready
    assert cpu_manager.lookup([1, 2]) == 0

    # no events so far
    assert list(cpu_manager.take_events()) == []

    # complete store [1, 2]
    cpu_manager.complete_store([1, 2])
    verify_events(cpu_manager.take_events(),
                  block_size=block_size,
                  expected_stores=({1, 2}, ))

    # lookup [1, 2]
    assert cpu_manager.lookup([1]) == 1
    assert cpu_manager.lookup([1, 2]) == 2
    assert cpu_manager.lookup([1, 2, 3]) == 2

    # prepare store [2, 3, 4, 5] -> evicts [1]
    prepare_store_output = cpu_manager.prepare_store([2, 3, 4, 5])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[3, 4, 5],
            store_block_ids=[2, 3, 0],
            block_hashes_evicted=[1],
        ))

    # verify eviction event
    verify_events(cpu_manager.take_events(),
                  block_size=block_size,
                  expected_evictions=({1}, ))

    # prepare store with no space
    assert cpu_manager.prepare_store([1, 6]) is None

    # complete store [2, 3, 4, 5]
    cpu_manager.complete_store([2, 3, 4, 5])

    # prepare load [2, 3]
    prepare_load_output = cpu_manager.prepare_load([2, 3])
    verify_load_output(prepare_load_output, [1, 2])

    # prepare store with no space ([2, 3] is being loaded)
    assert cpu_manager.prepare_store([6, 7, 8]) is None

    # complete load [2, 3]
    cpu_manager.complete_load([2, 3])

    # prepare store [6, 7, 8] -> evicts [2, 3, 4] (oldest)
    prepare_store_output = cpu_manager.prepare_store([6, 7, 8])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[6, 7, 8],
            store_block_ids=[3, 2, 1],
            block_hashes_evicted=[2, 3, 4],
        ))

    # complete store [6, 7, 8]
    cpu_manager.complete_store([6, 7, 8])

    # touch [5, 6, 7] (move to end of LRU order)
    cpu_manager.touch([5, 6, 7])

    # prepare store [7, 9] -> evicts [8] (oldest following previous touch)
    prepare_store_output = cpu_manager.prepare_store([9])
    verify_store_output(
        prepare_store_output,
        ExpectedPrepareStoreOutput(
            block_hashes_to_store=[9],
            store_block_ids=[1],
            block_hashes_evicted=[8],
        ))

    # complete store [7, 9] with failure
    cpu_manager.complete_store([7, 9], success=False)

    # assert [7] is still stored, but [9] is not
    assert cpu_manager.lookup([7]) == 1
    assert cpu_manager.lookup([9]) == 0

    verify_events(cpu_manager.take_events(),
                  block_size=block_size,
                  expected_stores=({3, 4, 5}, {6, 7, 8}),
                  expected_evictions=({2, 3, 4}, {8}))


NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_BLOCK_SIZES = [16]
GPU_BLOCKS_PER_CPU_BLOCK = [1, 3]
HEAD_SIZES = [64]
NUM_HEADS = [8]
NUM_LAYERS = [4]
DTYPES = [torch.bfloat16]
SEEDS = [0]
CUDA_DEVICES = ['cuda:0']
NUM_MAPPINGS = [3]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("gpu_block_size", GPU_BLOCK_SIZES)
@pytest.mark.parametrize("gpu_blocks_per_cpu_block", GPU_BLOCKS_PER_CPU_BLOCK)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_transfer(
    gpu_to_cpu: bool,
    num_mappings: int,
    head_size: int,
    num_heads: int,
    gpu_block_size: int,
    gpu_blocks_per_cpu_block: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_layers: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)

    # create per-layer GPU KV caches
    attn_backend = FlashAttentionBackend
    gpu_cache_shape = attn_backend.get_kv_cache_shape(num_gpu_blocks,
                                                      gpu_block_size,
                                                      num_heads, head_size)
    gpu_caches = {}
    for i in range(num_layers):
        gpu_caches[f'layer {i}'] = torch.rand(gpu_cache_shape,
                                              dtype=dtype,
                                              device=device)

    # create CPU KV caches
    cpu_block_size = gpu_blocks_per_cpu_block * gpu_block_size
    gpu_tensors, cpu_tensors = create_cpu_tensors(gpu_caches, gpu_block_size,
                                                  cpu_block_size,
                                                  num_cpu_blocks)

    # select block mappings
    gpu_blocks = random.sample(range(num_gpu_blocks),
                               num_mappings * gpu_blocks_per_cpu_block)
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # convert cpu blocks to gpu block size
    cpu_blocks_in_gpu_block_size = []
    for cpu_block in cpu_blocks:
        base_block_id = cpu_block * gpu_blocks_per_cpu_block
        for i in range(gpu_blocks_per_cpu_block):
            cpu_blocks_in_gpu_block_size.append(i + base_block_id)

    # maybe skip a GPU block to test writing to the middle of a CPU block
    if gpu_to_cpu:
        gpu_blocks = gpu_blocks[gpu_blocks_per_cpu_block - 1:]
        cpu_blocks_in_gpu_block_size = cpu_blocks_in_gpu_block_size[
            gpu_blocks_per_cpu_block - 1:]

    # set transfer direction
    if gpu_to_cpu:
        src_kv_caches = gpu_tensors
        dst_kv_caches = cpu_tensors
        src_block_size = gpu_block_size
        dst_block_size = cpu_block_size
        src_spec_class = GPULoadStoreSpec
        dst_spec_class = CPULoadStoreSpec
        src_blocks = gpu_blocks
        dst_blocks = cpu_blocks
        src_blocks_in_gpu_block_size = gpu_blocks
        dst_blocks_in_gpu_block_size = cpu_blocks_in_gpu_block_size
        dst_size_in_gpu_blocks = num_cpu_blocks * gpu_blocks_per_cpu_block
    else:
        src_kv_caches = cpu_tensors
        dst_kv_caches = gpu_tensors
        src_block_size = cpu_block_size
        dst_block_size = gpu_block_size
        src_spec_class = CPULoadStoreSpec
        dst_spec_class = GPULoadStoreSpec
        src_blocks = cpu_blocks
        dst_blocks = gpu_blocks
        src_blocks_in_gpu_block_size = cpu_blocks_in_gpu_block_size
        dst_blocks_in_gpu_block_size = gpu_blocks
        dst_size_in_gpu_blocks = num_gpu_blocks

    # build dst -> src mapping
    dst_to_src = {}
    for src_block, dst_block in zip(src_blocks_in_gpu_block_size,
                                    dst_blocks_in_gpu_block_size):
        dst_to_src[dst_block] = src_block

    # build transfer specs
    src_specs = [src_spec_class(block_id) for block_id in src_blocks]
    dst_specs = [dst_spec_class(block_id) for block_id in dst_blocks]

    # create transfer function
    transfer_func = generate_tensors_transfer_function(src_kv_caches,
                                                       dst_kv_caches,
                                                       attn_backend,
                                                       src_block_size,
                                                       dst_block_size)

    # clone src and dst tensors before transfer
    orig_src_caches = [x.clone() for x in src_kv_caches]
    orig_dst_caches = [x.clone() for x in dst_kv_caches]

    # call transfer function
    assert transfer_func((src_specs, dst_specs)) is True

    # verify src tensors did not change
    for orig_tensor, tensor in zip(orig_src_caches, src_kv_caches):
        assert torch.equal(orig_tensor, tensor)

    print(src_blocks)
    print(dst_blocks)
    print(dst_to_src)

    # verify dst tensors
    for dst_block in range(dst_size_in_gpu_blocks):
        src_block_candidate = dst_to_src.get(dst_block)
        for src_cache, dst_cache, orig_dst_cache in zip(
                src_kv_caches, dst_kv_caches, orig_dst_caches):
            # iterate over key, value
            for i in range(2):
                if src_block_candidate is not None:
                    expected_value = src_cache[i][src_block_candidate]
                else:
                    expected_value = orig_dst_cache[i][dst_block]
                torch.testing.assert_close(dst_cache[i][dst_block].cpu(),
                                           expected_value.cpu())
