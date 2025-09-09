# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.offloading.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.offloading.factory import OffloadingSpecFactory
from vllm.v1.offloading.mediums import GPULoadStoreSpec
from vllm.v1.offloading.spec import OffloadingSpec
from vllm.v1.offloading.worker.worker import (OffloadingQueueManager,
                                              TransferSpec)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
import time
ReqId = str

logger = init_logger(__name__)


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]


class OffloadingConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config, role)

        spec = OffloadingSpecFactory.create_spec(vllm_config)

        self.connector_scheduler: Optional[OffloadingConnectorScheduler] = None
        self.connector_worker: Optional[OffloadingConnectorWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          OffloadingConnectorMetadata)
        start_time = time.time()
        req_id=self._connector_metadata.reqs_to_load.keys()
        self.connector_worker.start_load_kv(self._connector_metadata)
        end_time = time.time() - start_time
        print(f"[OFFLOADING]Time taken to start load kv {end_time} seconds req.id {req_id}")
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata,
                          OffloadingConnectorMetadata)
        self.connector_worker.start_store_kv(self._connector_metadata)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = (self.offloaded_block_size //
                                  self.gpu_block_size)
        self.manager: OffloadingManager = spec.get_manager()

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored: defaultdict[ReqId, set[int]] = (defaultdict(
            set[int]))
        self._reqs_being_loaded: defaultdict[ReqId, set[int]] = (defaultdict(
            set[int]))

    def get_num_new_matched_tokens(
            self, request: Request,
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        num_blocks = request.num_tokens // self.offloaded_block_size

        block_hashes = [
            blk_hash.hash_value
            for blk_hash in request.block_hashes[self.block_size_factor -
                                                 1::self.block_size_factor]
        ]
        assert len(block_hashes) == num_blocks

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(block_hashes[start_block_idx:])
        if hits == 0:
            return 0, False

        num_hit_tokens = (self.offloaded_block_size *
                          (start_block_idx + hits) - num_computed_tokens)
        logger.info(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        return num_hit_tokens, True

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks,
                                 num_external_tokens: int):
        self._requests[request.request_id] = request
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        assert len(block_groups) == 1, "Only one group is supported"
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(block.block_hash is not None
                                      for block in blocks.blocks[0])
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert (num_external_tokens == num_pending_gpu_blocks *
                self.gpu_block_size)

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        block_hashes = [
            blk_hash.hash_value
            for blk_hash in request.block_hashes[self.block_size_factor -
                                                 1::self.block_size_factor]
        ]
        assert len(block_hashes) >= num_blocks

        block_hashes = block_hashes[start_block_idx:num_blocks]

        src_specs = self.manager.prepare_load(block_hashes)
        dst_specs = [
            GPULoadStoreSpec(gpu_block_id)
            for gpu_block_id in block_ids[num_computed_gpu_blocks:]
        ]

        self._reqs_to_load[request.request_id] = (src_specs, dst_specs)
        self._reqs_being_loaded[request.request_id] |= set(block_hashes)

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in itertools.chain(
            ((req_data.req_id, req_data.block_ids, False)
             for req_data in scheduler_output.scheduled_new_reqs),
                zip(
                    scheduler_output.scheduled_cached_reqs.req_ids,
                    scheduler_output.scheduled_cached_reqs.new_block_ids,
                    scheduler_output.scheduled_cached_reqs.
                    resumed_from_preemption)):

            if preempted:
                self._request_block_ids[req_id] = []

            if new_block_id_groups:
                assert len(new_block_id_groups) == 1
                new_block_ids = new_block_id_groups[0]
                self._request_block_ids[req_id] += new_block_ids

            block_ids = self._request_block_ids[req_id]

            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // self.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            num_gpu_blocks = num_blocks * self.block_size_factor
            block_hashes = [
                blk_hash.hash_value for blk_hash in
                req.block_hashes[self.block_size_factor -
                                 1:num_gpu_blocks:self.block_size_factor]
            ]
            assert len(block_hashes) == num_blocks

            new_block_hashes = block_hashes[start_block_idx:]
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning("Cannot store %s blocks", num_new_blocks)
                break

            self._next_stored_block_idx[req_id] = num_blocks

            block_hashes_to_store = set(store_output.block_hashes_to_store)
            if not block_hashes_to_store:
                continue

            self.manager.touch(block_hashes)

            dst_specs = store_output.store_specs
            src_specs: list[LoadStoreSpec] = []
            for idx, blk_hash in enumerate(new_block_hashes):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_specs.append(
                        GPULoadStoreSpec(block_ids[gpu_block_idx + i]))

            reqs_to_store[req_id] = (src_specs, dst_specs)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.info(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output))
        self._reqs_to_load = {}
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(list(block_hashes))

        for req_id in connector_output.finished_recving or []:
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                self.manager.complete_load(list(block_hashes))

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes,
                                   medium=event.medium)
            else:
                yield BlockStored(block_hashes=event.block_hashes,
                                  parent_block_hash=None,
                                  token_ids=[],
                                  lora_id=None,
                                  block_size=event.block_size,
                                  medium=event.medium)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.manager = OffloadingQueueManager()
        self._unregistered_gpu_kv_caches: dict[str, torch.Tensor] = {}

        self._job_counter = 0

        # req_id -> (job_id, store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> set(active job IDs)
        self._load_jobs: defaultdict[ReqId, set[int]] = defaultdict(set[int])
        self._store_jobs: defaultdict[ReqId, set[int]] = defaultdict(set[int])

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._unregistered_gpu_kv_caches = kv_caches

    def start_load_kv(self, metadata: OffloadingConnectorMetadata):
        # register offloading workers on first run
        if self._unregistered_gpu_kv_caches:
            for src_cls, dst_cls, xfer_fn, num_threads in (
                    self.spec.get_transfer_functions(
                        self._unregistered_gpu_kv_caches)):
                self.manager.register_worker(src_cls, dst_cls, xfer_fn,
                                             num_threads)
            self._unregistered_gpu_kv_caches = {}

        for req_id, transfer_spec in metadata.reqs_to_load.items():
            print(f"req.id {req_id}, transfer_spec {transfer_spec}")
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            self._load_jobs[req_id].add(job_id)
            self.manager.transfer_async(job_id, transfer_spec)

    def start_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            self.manager.transfer_async(job_id, transfer_spec)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for job_id, success in self.manager.get_finished():
            # we currently do not support job failures
            assert success
            req_id, store = self._jobs.pop(job_id)
            if store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_jobs = self._load_jobs[req_id]
                req_jobs.remove(job_id)
                if not req_jobs:
                    del self._load_jobs[req_id]
                    finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]
        if finished_recving != set() or finished_sending != set():
            print(f"[OFFLOADING]Finished sending: {finished_sending}, finished receiving: {finished_recving}")
        return finished_sending, finished_recving
