import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from collections import deque
from typing import List, Deque, Set, Optional, Dict

from digital_twin_dynamic.structures import Request, RunningBatch


class Scheduler:

    def __init__(
            self,
            adapter_slots: int,
            available_gpu_memory: int,
            max_num_batched_tokens: int,
            debug_path: Optional[str] = None
    ):
        # define constants
        self.adapter_slots: int = adapter_slots
        self.debug_path: str = debug_path
        self.maximum_tokens: int = available_gpu_memory
        self.maximum_batched_tokens: int = round(max_num_batched_tokens * 1.3)  # to account for not simulated chunked prefill

        # define dynamic structures
        self.waiting_requests: Deque[Request] = deque()
        self.running_batch: RunningBatch = RunningBatch()

        # in case of debug, monitor progress
        if self.debug_path is not None:
            self.progress_time: List[float] = []
            self.progress_num_waiting: List[int] = []
            self.progress_num_running: List[int] = []
            self.progress_num_finished: List[int] = []
            self.progress_tokens_in_use: List[float] = []

    # Update scheduler with new arriving requests
    def new_arrival(self, request: Request, arrival_time: float) -> None:
        self.waiting_requests.append(request)

    # Return requests to run in this step
    def run_step(self, current_time: float) -> RunningBatch:
        if self.debug_path:
            self.progress_time.append(current_time)
            self.progress_num_waiting.append(len(self.waiting_requests))
            self.progress_num_running.append(len(self.running_batch.running_requests))
            self.progress_num_finished.append(self.running_batch.finished_requests_count)
            self.progress_tokens_in_use.append(self.running_batch.total_tokens)

        # update running requests
        leftover_waiting_sequences: Deque[Request] = deque()
        while self.waiting_requests:
            request = self.waiting_requests.popleft()

            # assure memory limit for KV values
            if (self.running_batch.total_tokens + request.input_tokens + request.output_tokens) >= self.maximum_tokens:
                leftover_waiting_sequences.appendleft(request)
                break

            # assure batched tokens limit
            if (self.running_batch.compute_prefill_tokens + self.running_batch.compute_decode_tokens + request.input_tokens) >= self.maximum_batched_tokens:
                leftover_waiting_sequences.appendleft(request)
                break

            # assure adapter limit
            if request.adapter_id not in self.running_batch.running_adapters and len(self.running_batch.running_adapters) == self.adapter_slots:
                leftover_waiting_sequences.appendleft(request)
                continue

            # move to running requests
            self.running_batch.add_request(request)
        self.waiting_requests.extendleft(leftover_waiting_sequences)

        return self.running_batch

    def get_num_waiting_requests(self):
        return len(self.waiting_requests)

    def get_num_running_requests(self):
        return len(self.running_batch.running_requests)

    def get_num_finished_requests(self):
        return self.running_batch.finished_requests_count

    def get_mean_itl(self):
        return self.running_batch.itl_sum / self.running_batch.finished_requests_count if self.running_batch.finished_requests_count > 0 else 0

    def get_mean_ttft(self, final_duration: float):
        if self.running_batch.finished_requests_count <= 0:
            return 0

        count: int = self.running_batch.finished_requests_count
        sum: float = self.running_batch.ttft_sum

        count += len(self.waiting_requests) + len(self.running_batch.running_requests)
        sum += (len(self.waiting_requests) + len(self.running_batch.running_requests)) * final_duration

        return sum / count

    def get_processed_input_tokens(self) -> int:
        return self.running_batch.processed_input_tokens

    def get_processed_output_tokens(self) -> int:
        return self.running_batch.processed_output_tokens

    def debug_chart(self):
        assert self.debug_path is not None

        # waiting plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_waiting, label='num waiting')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_waiting'))
        print('Mean waiting:', np.mean(self.progress_num_waiting))

        # running plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_running, label='num running')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_running'))
        print('Mean running:', np.mean(self.progress_num_running))

        # finished plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_num_finished, label='num finished (accumulated)')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_finished'))

        # KV cache usage plot
        fig, ax = plt.subplots()
        ax.plot(self.progress_time, self.progress_tokens_in_use, label='tokens in use')
        ax.axhline(y=self.maximum_tokens, label='maximum tokens')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('tokens (#)')
        ax.legend()
        fig.savefig(os.path.join(self.debug_path, 'scheduler_kv_cache'))
