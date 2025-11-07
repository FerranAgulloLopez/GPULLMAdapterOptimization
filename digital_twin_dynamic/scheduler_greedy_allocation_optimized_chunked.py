import os
import math

# import matplotlib.pyplot as plt

from collections import deque
from typing import List, Deque, Set, Optional, Dict, OrderedDict as OrderedDictType
from collections import OrderedDict

from digital_twin_dynamic.structures import RequestGreedyAllocationChunked, RunningBatchGreedyAllocationChunked
from digital_twin_dynamic.scheduler_greedy_allocation import SchedulerGreedyAllocation


BLOCK_SIZE: int = 16


class SchedulerGreedyAllocationOptimizedChunked(SchedulerGreedyAllocation):

    def __init__(
            self,
            adapter_slots: int,
            available_gpu_memory: int,
            max_num_batched_tokens: int,
            debug_path: Optional[str] = None
    ):
        super().__init__(adapter_slots, available_gpu_memory, max_num_batched_tokens, debug_path)

        # define dynamic structures
        self.waiting_requests: OrderedDictType[int, RequestGreedyAllocationChunked] = OrderedDict()
        self.running_batch: RunningBatchGreedyAllocationChunked = RunningBatchGreedyAllocationChunked(
            BLOCK_SIZE,
            self.maximum_blocks,
            self.maximum_batched_tokens
        )

        # define dynamic structures for optimization
        self.waiting_adapters: Dict[str, int] = {}

        # define vars for optimization
        self.new_arrival_since_last_step: bool = False
        self.new_arrival_since_last_step_adapter_id: Set[str] = set()
        self.max_kv_reached: bool = False
        self.max_batched_tokens_reached: bool = False
        self.max_adapter_reached: bool = False

    # Update scheduler with new arriving requests
    def new_arrival(self, request: RequestGreedyAllocationChunked, arrival_time: float) -> None:
        self.new_arrival_since_last_step = True
        self.new_arrival_since_last_step_adapter_id.add(request.adapter_id)
        if request.adapter_id not in self.waiting_adapters:
            self.waiting_adapters[request.adapter_id] = 0
        self.waiting_adapters[request.adapter_id] += 1
        self.waiting_requests[request.id] = request

    # Return requests to run in this step
    def run_step(self, current_time: float) -> RunningBatchGreedyAllocationChunked:
        if self.debug_path:
            self.progress_time.append(current_time)
            self.progress_num_waiting.append(len(self.waiting_requests))
            self.progress_num_running.append(len(self.running_batch.running_requests))
            self.progress_num_finished.append(self.running_batch.finished_requests_count)
            self.progress_num_preempted.append(self.running_batch.preempted_requests_count)
            self.progress_blocks_in_use.append(self.running_batch.total_blocks)

        # add preempted requests to the beginning of the waiting queue
        for request in reversed(self.running_batch.preempted_requests_in_last_step):
            self.waiting_requests[request.id] = request
            self.waiting_requests.move_to_end(request.id, last=False)

        # check if recompute is required
        recompute: bool = False

        if self.new_arrival_since_last_step and not self.max_kv_reached and not self.max_batched_tokens_reached and not self.max_adapter_reached:
            recompute = True
        elif self.max_kv_reached and self.running_batch.removed_request_in_last_step:
            recompute = True
        elif self.max_batched_tokens_reached and self.running_batch.batched_tokens > 0:
            recompute = True
        elif self.max_adapter_reached and not self.max_kv_reached and not self.max_batched_tokens_reached:
            if self.running_batch.removed_adapter_in_last_step:
                recompute = True
            elif self.new_arrival_since_last_step and any(
                    adapter_id in self.running_batch.running_adapters for adapter_id in
                    self.new_arrival_since_last_step_adapter_id):
                recompute = True

        self.new_arrival_since_last_step = False
        self.new_arrival_since_last_step_adapter_id = set()

        # update running requests
        if recompute:
            self.max_kv_reached = False
            self.max_batched_tokens_reached = False
            self.max_adapter_reached = False

            waiting_adapters_running: Set[str] = {adapter for adapter in self.running_batch.running_adapters.keys() if adapter in self.waiting_adapters}

            new_running_requests: List[RequestGreedyAllocationChunked] = []
            for request in self.waiting_requests.values():

                # assure batched tokens limit
                if self.running_batch.batched_tokens >= self.maximum_batched_tokens:
                    self.max_batched_tokens_reached = True
                    break

                new_tokens: int = min(request.input_tokens, self.maximum_batched_tokens - self.running_batch.batched_tokens)
                if new_tokens == request.input_tokens:
                    new_tokens += 1
                new_blocks: int = math.ceil(new_tokens / BLOCK_SIZE)

                # assure memory limit for KV values
                if (self.running_batch.total_blocks + new_blocks) > self.maximum_blocks:
                    self.max_kv_reached = True
                    break

                # assure adapter limit
                if request.adapter_id not in self.running_batch.running_adapters and len(self.running_batch.running_adapters) == self.adapter_slots:
                    self.max_adapter_reached = True
                    if len(waiting_adapters_running) == 0:
                        break
                    continue

                # move to running requests
                request.batched_tokens = new_tokens
                request.blocks_in_use = new_blocks
                self.running_batch.add_request(request)
                self.waiting_adapters[request.adapter_id] += 1
                if self.waiting_adapters[request.adapter_id] == 0:
                    del self.waiting_adapters[request.adapter_id]
                if request.adapter_id not in waiting_adapters_running:
                    waiting_adapters_running.add(request.adapter_id)
                new_running_requests.append(request)

            for request in new_running_requests:
                del self.waiting_requests[request.id]

        return self.running_batch
