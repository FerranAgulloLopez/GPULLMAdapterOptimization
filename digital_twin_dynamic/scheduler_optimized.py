import os
import time

import matplotlib.pyplot as plt

from collections import deque
from typing import List, Deque, Set, Optional, Dict

from digital_twin_dynamic.structures import Request, RunningBatch
from digital_twin_dynamic.scheduler import Scheduler


class SchedulerOptimized(Scheduler):

    def __init__(
            self,
            adapter_slots: int,
            available_gpu_memory: int,
            max_num_batched_tokens: int,
            debug_path: Optional[str] = None
    ):
        super().__init__(adapter_slots, available_gpu_memory, max_num_batched_tokens, debug_path)

        # define vars for optimization
        self.new_arrival_since_last_step: bool = False
        self.new_arrival_since_last_step_adapter_id: Set[str] = set()
        self.max_kv_reached: bool = False
        self.max_batched_tokens_reached: bool = False
        self.max_adapter_reached: bool = False

        # TODO remove DEBUGGING
        self.reason_max_kv_reached: int = 0
        self.reason_max_batched_tokens_reached: int = 0
        self.reason_max_adapter_reached: int = 0

    # Update scheduler with new arriving requests
    def new_arrival(self, request: Request, arrival_time: float) -> None:
        self.new_arrival_since_last_step = True
        self.new_arrival_since_last_step_adapter_id.add(request.adapter_id)
        super().new_arrival(request, arrival_time)

    # Return requests to run in this step
    def run_step(self, current_time: float) -> RunningBatch:
        if self.debug_path:
            self.progress_time.append(current_time)
            self.progress_num_waiting.append(len(self.waiting_requests))
            self.progress_num_running.append(len(self.running_batch.running_requests))
            self.progress_num_finished.append(self.running_batch.finished_requests_count)
            self.progress_tokens_in_use.append(self.running_batch.total_tokens)

        # check if recompute is required
        recompute: bool = False

        if self.new_arrival_since_last_step and not self.max_kv_reached and not self.max_batched_tokens_reached and not self.max_adapter_reached:
            recompute = True
        elif self.max_kv_reached and self.running_batch.removed_request_in_last_step:
            recompute = True
        elif self.max_batched_tokens_reached and (self.running_batch.finished_prefill_in_last_step or self.running_batch.removed_request_in_last_step):
            recompute = True
        elif self.max_adapter_reached and not self.max_kv_reached and not self.max_batched_tokens_reached:
            if self.running_batch.removed_adapter_in_last_step:
                recompute = True
            elif self.new_arrival_since_last_step and any(adapter_id in self.running_batch.running_adapters for adapter_id in self.new_arrival_since_last_step_adapter_id):
                recompute = True

        self.new_arrival_since_last_step = False
        self.new_arrival_since_last_step_adapter_id = set()

        # update running requests
        if recompute:
            self.max_kv_reached = False
            self.max_batched_tokens_reached = False
            self.max_adapter_reached = False
            leftover_waiting_sequences: Deque[Request] = deque()
            while self.waiting_requests:
                request = self.waiting_requests.popleft()

                # assure memory limit for KV values
                if (self.running_batch.total_tokens + request.input_tokens + request.output_tokens) >= self.maximum_tokens:
                    leftover_waiting_sequences.appendleft(request)
                    self.max_kv_reached = True
                    self.reason_max_kv_reached += 1
                    break

                # assure batched tokens limit
                if (self.running_batch.compute_prefill_tokens + self.running_batch.compute_decode_tokens + request.input_tokens) >= self.maximum_batched_tokens:
                    leftover_waiting_sequences.appendleft(request)
                    self.max_batched_tokens_reached = True
                    self.reason_max_batched_tokens_reached += 1
                    break

                # assure adapter limit
                if request.adapter_id not in self.running_batch.running_adapters and len(self.running_batch.running_adapters) == self.adapter_slots:
                    leftover_waiting_sequences.appendleft(request)
                    self.max_adapter_reached = True
                    self.reason_max_adapter_reached += 1
                    continue

                # move to running requests
                self.running_batch.add_request(request)
            self.waiting_requests.extendleft(leftover_waiting_sequences)
        return self.running_batch

    def debug_chart(self):
        super().debug_chart()
        print(f'Reasons. Steps: {len(self.progress_time)}. KV: {self.reason_max_kv_reached}. Batched tokens: {self.reason_max_batched_tokens_reached}. Adapter: {self.reason_max_adapter_reached}')
