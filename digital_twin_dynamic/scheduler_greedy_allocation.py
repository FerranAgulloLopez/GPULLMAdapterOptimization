import os
import math
import numpy as np

# import matplotlib.pyplot as plt

from collections import deque
from typing import List, Deque, Set, Optional

from digital_twin_dynamic.structures import RequestGreedyAllocation, RunningBatchGreedyAllocation


BLOCK_SIZE: int = 16


class SchedulerGreedyAllocation:

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
        maximum_tokens: int = available_gpu_memory
        self.maximum_blocks: int = round(maximum_tokens / BLOCK_SIZE)
        self.maximum_batched_tokens: int = max_num_batched_tokens

        # define dynamic structures
        self.waiting_requests: Deque[RequestGreedyAllocation] = deque()
        self.running_batch: RunningBatchGreedyAllocation = RunningBatchGreedyAllocation(BLOCK_SIZE, self.maximum_blocks)

        # in case of debug, monitor progress
        if self.debug_path is not None:
            self.progress_time: List[float] = []
            self.progress_num_waiting: List[int] = []
            self.progress_num_running: List[int] = []
            self.progress_num_finished: List[int] = []
            self.progress_num_preempted: List[int] = []
            self.progress_blocks_in_use: List[float] = []

    # Update scheduler with new arriving requests
    def new_arrival(self, request: RequestGreedyAllocation, arrival_time: float) -> None:
        self.waiting_requests.append(request)

    # Return requests to run in this step
    def run_step(self, current_time: float) -> RunningBatchGreedyAllocation:
        if self.debug_path:
            self.progress_time.append(current_time)
            self.progress_num_waiting.append(len(self.waiting_requests))
            self.progress_num_running.append(len(self.running_batch.running_requests))
            self.progress_num_finished.append(self.running_batch.finished_requests_count)
            self.progress_num_preempted.append(self.running_batch.preempted_requests_count)
            self.progress_blocks_in_use.append(self.running_batch.total_blocks)

        # add preempted requests to the beginning of the waiting queue
        self.waiting_requests.extendleft(self.running_batch.preempted_requests_in_last_step)

        # update running requests
        leftover_waiting_sequences: Deque[RequestGreedyAllocation] = deque()
        while self.waiting_requests:
            request: RequestGreedyAllocation = self.waiting_requests.popleft()

            new_blocks: int = math.ceil((request.input_tokens + 1) / BLOCK_SIZE)

            # assure memory limit for KV values
            if (self.running_batch.total_blocks + new_blocks) > self.maximum_blocks:
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
            request.blocks_in_use = new_blocks
            self.running_batch.add_request(request)
        self.waiting_requests.extendleft(leftover_waiting_sequences)

        return self.running_batch

    def get_num_waiting_requests(self) -> int:
        return len(self.waiting_requests)

    def get_num_running_requests(self) -> int:
        return len(self.running_batch.running_requests)

    def get_num_finished_requests(self) -> int:
        return self.running_batch.finished_requests_count

    def get_mean_itl(self) -> float:
        return self.running_batch.itl_sum / self.running_batch.finished_requests_count

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

    def debug_chart(self) -> None:
        assert self.debug_path is not None

        '''# waiting plot
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
        fig.savefig(os.path.join(self.debug_path, 'scheduler_kv_cache'))'''
