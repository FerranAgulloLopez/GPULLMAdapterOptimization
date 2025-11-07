import os
import math

import matplotlib.pyplot as plt

from collections import deque
from typing import List, Deque, Set, Optional

from digital_twin_dynamic.structures import Request


BLOCK_SIZE: int = 16


class SchedulerWithPreemptionOptimized:

    def __init__(
            self,
            adapter_slots: int,
            available_gpu_memory: int,
            debug_path: Optional[str] = None
    ):
        raise NotImplementedError('It is not yet updated to used max num batched tokens nor new running batch structure')

        # define constants
        self.adapter_slots: int = adapter_slots
        self.debug_path: str = debug_path
        maximum_tokens: int = available_gpu_memory
        self.maximum_blocks = maximum_tokens / BLOCK_SIZE

        # define dynamic structures
        self.waiting_requests: Deque[Request] = deque()
        self.running_requests: Deque[Request] = deque()
        self.running_adapters: Dict[str, int] = {}
        self.blocks_in_use = 0

        # define vars for monitoring metrics
        self.finished_requests_count = 0
        self.preempted_requests_count = 0
        self.itl_sum = 0
        self.ttft_sum = 0

        # define vars for optimization
        self.max_kv_reached: bool = False
        self.max_adapter_reached: bool = False
        self.recompute: bool = False

        # in case of debug, monitor progress
        if self.debug_path is not None:
            self.progress_time: List[float] = []
            self.progress_num_waiting: List[int] = []
            self.progress_num_running: List[int] = []
            self.progress_num_finished: List[int] = []
            self.progress_num_preempted: List[int] = []
            self.progress_blocks_in_use: List[float] = []

    # Update scheduler with new arriving requests
    def new_arrival(self, request: Request, arrival_time: float) -> None:
        if not self.max_batch_reached:
            if not self.max_adapter_reached:
                self.recompute = True
            elif request.adapter_id in self.running_adapters:
                self.recompute = True
        self.waiting_requests.append(request)

    # Return requests to run in this step
    def run_step(self, current_time: float) -> Deque[Request]:
        if self.debug_path:
            self.progress_time.append(current_time)
            self.progress_num_waiting.append(len(self.waiting_requests))
            self.progress_num_running.append(len(self.running_requests))
            self.progress_num_finished.append(self.finished_requests_count)
            self.progress_num_preempted.append(self.preempted_requests_count)
            self.progress_blocks_in_use.append(self.blocks_in_use)

        # remove completed requests, compute running adapters, schedule new running requests (preempt if necessary)
        aux_running_requests: Deque[Request] = deque()
        while self.running_requests:
            request: Request = self.running_requests.popleft()

            if request.remaining_output_tokens > 0:

                if request.blocks_in_use * BLOCK_SIZE < 1 + request.input_tokens + (request.output_tokens - request.remaining_output_tokens):
                    preempted_itself: bool = False
                    while not preempted_itself and self.blocks_in_use + 1 > self.maximum_blocks:
                        self.preempted_requests_count += 1
                        if self.running_requests:
                            preempted_request: Request = self.running_requests.pop()
                        else:
                            preempted_itself = True
                            preempted_request: Request = request
                        self.waiting_requests.appendleft(preempted_request)
                        self.blocks_in_use -= preempted_request.blocks_in_use
                        preempted_request.remaining_tokens = preempted_request.output_tokens
                        preempted_request.blocks_in_use = 0
                        preempted_request.ttft = 0
                        preempted_request.itl_sum = 0
                        self.running_adapters[preempted_request.adapter_id] -= 1
                        if self.running_adapters[preempted_request.adapter_id] == 0:
                            del self.running_adapters[preempted_request.adapter_id]
                            if self.max_adapter_reached:
                                self.recompute = True
                        if self.max_batch_reached:
                            self.recompute = True

                    if not preempted_itself:
                        self.blocks_in_use += 1
                        request.blocks_in_use += 1
                        aux_running_requests.append(request)

                else:
                    aux_running_requests.append(request)

            else:
                self.blocks_in_use -= request.blocks_in_use
                self.finished_requests_count += 1
                self.itl_sum += request.itl_sum / request.output_tokens
                self.ttft_sum += request.ttft
                self.running_adapters[request.adapter_id] -= 1
                if self.running_adapters[request.adapter_id] == 0:
                    del self.running_adapters[request.adapter_id]
                    if self.max_adapter_reached:
                        self.recompute = True
                if self.max_batch_reached:
                    self.recompute = True
        self.running_requests = aux_running_requests

        # update running requests
        if self.recompute:
            self.max_batch_reached = False
            self.max_adapter_reached = False
            self.recompute = False
            leftover_waiting_sequences: Deque[Request] = deque()
            while self.waiting_requests:
                request: Request = self.waiting_requests.popleft()

                new_blocks: int = math.ceil((request.input_tokens + 1) / BLOCK_SIZE)

                # assure memory limit for KV values
                if (self.blocks_in_use + new_blocks) > self.maximum_blocks:
                    leftover_waiting_sequences.appendleft(request)
                    self.max_batch_reached = True
                    break

                # assure adapter limit
                if request.adapter_id not in self.running_adapters and len(self.running_adapters) == self.adapter_slots:
                    leftover_waiting_sequences.appendleft(request)
                    self.max_adapter_reached = True
                    continue

                # move to running requests
                if request.adapter_id not in self.running_adapters:
                    self.running_adapters[request.adapter_id] = 0
                self.running_adapters[request.adapter_id] += 1
                self.running_requests.append(request)
                self.blocks_in_use += new_blocks
                request.blocks_in_use = new_blocks
            self.waiting_requests.extendleft(leftover_waiting_sequences)

        return self.running_requests

    def get_num_waiting_requests(self):
        return len(self.waiting_requests)

    def get_num_running_requests(self):
        return len(self.running_requests)

    def get_num_finished_requests(self):
        return self.finished_requests_count

    def get_mean_itl(self):
        return self.itl_sum / self.finished_requests_count

    def get_mean_ttft(self):
        return self.ttft_sum / self.finished_requests_count

    def debug_chart(self):
        assert self.debug_path is not None

        # waiting plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_waiting, label='num waiting')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_waiting'))

        # running plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_running, label='num running')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_running'))

        # finished plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_finished, label='num finished (accumulated)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_finished'))

        # preempted plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_preempted, label='num preempted (accumulated)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_preempted'))

        # KV cache usage plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_blocks_in_use, label='tokens in use')
        ax.axhline(y=self.maximum_blocks, label='maximum tokens')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('blocks (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_kv_cache'))
