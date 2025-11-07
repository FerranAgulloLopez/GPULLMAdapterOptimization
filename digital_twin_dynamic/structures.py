import math
from typing import List, Dict, Set, Tuple, Deque
from collections import deque


class Request:
    def __init__(
            self,
            _id: int,
            arrival_time: float,
            input_tokens: int,
            output_tokens: int,
            adapter_id: str,
            adapter_size: int
    ):
        assert input_tokens > 0
        assert output_tokens > 1

        # constants
        self.id: int = _id
        self.arrival_time: float = arrival_time  # in seconds
        self.input_tokens: int = input_tokens
        self.output_tokens: int = output_tokens
        self.adapter_id: str = adapter_id
        self.adapter_size: int = adapter_size

        # dynamic
        self.remaining_input_tokens: int = input_tokens
        self.remaining_output_tokens: int = output_tokens

        # monitoring
        self.ttft: float = None  # in seconds
        self.itl_sum: float = 0  # in seconds

    def reset(self):
        self.remaining_input_tokens: int = self.input_tokens
        self.remaining_output_tokens: int = self.output_tokens
        self.ttft = None
        self.itl_sum = 0


class RequestGreedyAllocation(Request):
    def __init__(
            self,
            _id: int,
            arrival_time: float,
            input_tokens: int,
            output_tokens: int,
            adapter_id: str,
            adapter_size: int
    ):
        super().__init__(_id, arrival_time, input_tokens, output_tokens, adapter_id, adapter_size)
        self.blocks_in_use: int = 0

    def reset(self):
        super().reset()
        self.blocks_in_use = 0


class RequestGreedyAllocationChunked(RequestGreedyAllocation):
    def __init__(
            self,
            _id: int,
            arrival_time: float,
            input_tokens: int,
            output_tokens: int,
            adapter_id: str,
            adapter_size: int
    ):
        super().__init__(_id, arrival_time, input_tokens, output_tokens, adapter_id, adapter_size)
        self.batched_tokens: int = 0

    def reset(self):
        super().reset()
        self.batched_tokens = 0


class RunningBatch:
    def __init__(self):
        # define dynamic structures
        self.running_requests: Set[Request] = set()
        self.running_adapters: Dict[str, int] = {}
        self.maximum_adapter_size_id: int = None
        self.maximum_adapter_size_value: int = None

        # define dynamic vars
        self.total_tokens: int = 0
        self.compute_prefill_tokens: int = 0
        self.compute_decode_tokens: int = 0
        self.finished_prefill_in_last_step: bool = False
        self.added_adapter_in_last_step: bool = False
        self.removed_request_in_last_step: bool = False
        self.removed_adapter_in_last_step: bool = False

        # define vars for monitoring metrics
        self.finished_requests_count = 0
        self.itl_sum = 0
        self.ttft_sum = 0
        self.processed_input_tokens: int = 0
        self.processed_output_tokens: int = 0

    def add_request(self, request: Request):
        # updates requests
        self.running_requests.add(request)

        # updates adapters
        if request.adapter_id not in self.running_adapters:
            self.added_adapter_in_last_step = True
            self.running_adapters[request.adapter_id] = 0
            if self.maximum_adapter_size_id is None or self.maximum_adapter_size_value < request.adapter_size:
                self.maximum_adapter_size_id = request.adapter_id
                self.maximum_adapter_size_value = request.adapter_size
        self.running_adapters[request.adapter_id] += 1

        # updates tokens
        self.total_tokens += request.input_tokens + request.output_tokens
        self.compute_prefill_tokens += request.input_tokens

    def update_requests_after_step(self, current_time: float, step_time: float):
        self.finished_prefill_in_last_step = False
        self.added_adapter_in_last_step = False
        self.removed_request_in_last_step = False
        self.removed_adapter_in_last_step = False
        recompute_max_adapter_size: bool = False

        requests_to_remove: List[Request] = []
        for request in self.running_requests:
            if request.remaining_input_tokens > 0:  # after prefill
                self.finished_prefill_in_last_step = True

                # update request
                request.remaining_input_tokens = 0
                request.remaining_output_tokens -= 1
                request.ttft = current_time - request.arrival_time

                # update dynamic vars
                self.compute_prefill_tokens -= request.input_tokens
                self.compute_decode_tokens += 1

                # update monitoring metrics
                self.processed_output_tokens += 1
                self.processed_input_tokens += request.input_tokens
            else:  # after decode
                # update request
                request.itl_sum += step_time
                request.remaining_output_tokens -= 1

                # update monitoring metrics
                self.processed_output_tokens += 1

                # check if it is the last decode step
                if request.remaining_output_tokens == 0:  # last decode step
                    self.removed_request_in_last_step = True

                    # updates requests
                    requests_to_remove.append(request)

                    # updates adapters
                    self.running_adapters[request.adapter_id] -= 1
                    if self.running_adapters[request.adapter_id] == 0:
                        del self.running_adapters[request.adapter_id]
                        self.removed_adapter_in_last_step = True
                        if self.maximum_adapter_size_id == request.adapter_id:
                            recompute_max_adapter_size = True

                    # update dynamic vars
                    self.total_tokens -= request.input_tokens + request.output_tokens
                    self.compute_decode_tokens -= 1

                    # update monitoring metrics
                    self.finished_requests_count += 1
                    self.itl_sum += request.itl_sum / request.output_tokens
                    self.ttft_sum += request.ttft

        for request in requests_to_remove:
            self.running_requests.remove(request)

        if recompute_max_adapter_size:
            if len(self.running_requests) > 0:
                self.maximum_adapter_size_id, self.maximum_adapter_size_value = max(enumerate([request.adapter_size for request in self.running_requests]), key=lambda x: x[1])
            else:
                self.maximum_adapter_size_id = None
                self.maximum_adapter_size_value = None

    def get_compute_prefill_tokens(self) -> int:
        return self.compute_prefill_tokens

    def get_compute_decode_tokens(self) -> int:
        return self.compute_decode_tokens

    def get_running_adapters(self) -> Dict[str, int]:
        return self.running_adapters

    def get_len_running_adapters(self) -> int:
        return len(self.running_adapters)

    def get_maximum_adapter_size(self) -> int:
        return self.maximum_adapter_size_value


class RunningBatchGreedyAllocation:
    def __init__(self, block_size: int, maximum_blocks: int):
        # define constants
        self.block_size = block_size
        self.maximum_blocks = maximum_blocks

        # define dynamic structures
        self.running_requests: Deque[RequestGreedyAllocation] = deque()
        self.running_adapters: Dict[str, int] = {}
        self.maximum_adapter_size_id: int = None
        self.maximum_adapter_size_value: int = None
        self.preempted_requests_in_last_step: Deque[RequestGreedyAllocation] = deque()

        # define dynamic vars
        self.total_blocks: int = 0
        self.compute_prefill_tokens: int = 0
        self.compute_decode_tokens: int = 0
        self.finished_prefill_in_last_step: bool = False
        self.added_adapter_in_last_step: bool = False
        self.removed_request_in_last_step: bool = False
        self.removed_adapter_in_last_step: bool = False

        # define vars for monitoring metrics
        self.finished_requests_count = 0
        self.preempted_requests_count: int = 0
        self.itl_sum = 0
        self.ttft_sum = 0
        self.processed_input_tokens: int = 0
        self.processed_output_tokens: int = 0

    def add_request(self, request: RequestGreedyAllocation):
        # updates requests
        self.running_requests.append(request)

        # updates adapters
        if request.adapter_id not in self.running_adapters:
            self.added_adapter_in_last_step = True
            self.running_adapters[request.adapter_id] = 0
            if self.maximum_adapter_size_id is None or self.maximum_adapter_size_value < request.adapter_size:
                self.maximum_adapter_size_id = request.adapter_id
                self.maximum_adapter_size_value = request.adapter_size
        self.running_adapters[request.adapter_id] += 1

        # updates tokens
        self.total_blocks += request.blocks_in_use
        self.compute_prefill_tokens += request.input_tokens
        self.compute_decode_tokens += 1  # different from vLLM

    def update_requests_after_step(self, current_time: float, step_time: float):
        self.preempted_requests_in_last_step = deque()
        self.finished_prefill_in_last_step = False
        self.added_adapter_in_last_step = False
        self.removed_request_in_last_step = False
        self.removed_adapter_in_last_step = False
        recompute_max_adapter_size: bool = False

        aux_running_requests: Deque[RequestGreedyAllocation] = deque()
        while self.running_requests:
            request: RequestGreedyAllocation = self.running_requests.popleft()

            if request.remaining_input_tokens > 0:  # after prefill
                self.finished_prefill_in_last_step = True

                # update request
                request.remaining_input_tokens = 0
                request.remaining_output_tokens -= 1
                request.ttft = current_time - request.arrival_time

                # update dynamic vars
                self.compute_prefill_tokens -= request.input_tokens
                # self.compute_decode_tokens += 1  # different from vLLM

                # update monitoring metrics
                self.processed_output_tokens += 1
                self.processed_input_tokens += request.input_tokens
            else:  # after decode
                # update request
                request.itl_sum += step_time
                request.remaining_output_tokens -= 1

                # update monitoring metrics
                self.processed_output_tokens += 1

                # check if it is the last decode step
                if request.remaining_output_tokens == 0:  # last decode step
                    recompute_max_adapter_size = recompute_max_adapter_size or self.__remove_request(request)

                    # update monitoring metrics
                    self.finished_requests_count += 1
                    self.itl_sum += request.itl_sum / request.output_tokens
                    self.ttft_sum += request.ttft

            # check if there is enough request blocks for next step if request did not finish
            if request.remaining_output_tokens > 0:

                if request.blocks_in_use * self.block_size < 1 + request.input_tokens + (request.output_tokens - request.remaining_output_tokens):  # request blocks are not enough for next step

                    preempted_itself: bool = False
                    # assign new block to request, preempt other less priority requests if necessary
                    raise Exception('Not properly implemented, block size here below makes no sense')
                    while not preempted_itself and self.block_size + 1 > self.maximum_blocks:
                        self.removed_request_in_last_step = True
                        self.preempted_requests_count += 1
                        if self.running_requests:
                            preempted_request: RequestGreedyAllocation = self.running_requests.pop()
                        else:
                            preempted_itself = True
                            preempted_request: RequestGreedyAllocation = request
                        self.preempted_requests_in_last_step.append(preempted_request)
                        recompute_max_adapter_size = recompute_max_adapter_size or self.__remove_request(request)
                        self.processed_input_tokens -= request.input_tokens
                        self.processed_output_tokens -= request.output_tokens - request.remaining_output_tokens
                        preempted_request.reset()

                    if not preempted_itself:
                        self.total_blocks += 1
                        request.blocks_in_use += 1
                        aux_running_requests.append(request)

                else:  # request blocks are enough for next step
                    aux_running_requests.append(request)

        self.running_requests = aux_running_requests

        if recompute_max_adapter_size:
            if len(self.running_requests) > 0:
                self.maximum_adapter_size_id, self.maximum_adapter_size_value = max(enumerate([request.adapter_size for request in self.running_requests]), key=lambda x: x[1])
            else:
                self.maximum_adapter_size_id = None
                self.maximum_adapter_size_value = None

    def __remove_request(self, request: RequestGreedyAllocation) -> bool:
        self.removed_request_in_last_step = True
        recompute_max_adapter_size: bool = False

        # updates adapters
        self.running_adapters[request.adapter_id] -= 1
        if self.running_adapters[request.adapter_id] == 0:
            del self.running_adapters[request.adapter_id]
            self.removed_adapter_in_last_step = True
            if self.maximum_adapter_size_id == request.adapter_id:
                recompute_max_adapter_size = True

        # update dynamic vars
        self.total_blocks -= request.blocks_in_use
        self.compute_decode_tokens -= 1

        return recompute_max_adapter_size

    def get_compute_prefill_tokens(self) -> int:
        return self.compute_prefill_tokens

    def get_compute_decode_tokens(self) -> int:
        return self.compute_decode_tokens

    def get_running_adapters(self) -> Dict[str, int]:
        return self.running_adapters

    def get_len_running_adapters(self) -> int:
        return len(self.running_adapters)

    def get_maximum_adapter_size(self) -> int:
        return self.maximum_adapter_size_value

    def added_adapter_in_last_step(self) -> bool:
        return self.added_adapter_in_last_step


class RunningBatchGreedyAllocationChunked:
    def __init__(self, block_size: int, maximum_blocks: int, maximum_batched_tokens: int):
        # define constants
        self.block_size = block_size
        self.maximum_blocks = maximum_blocks
        self.maximum_batched_tokens = maximum_batched_tokens

        # define dynamic structures
        self.running_requests: Deque[RequestGreedyAllocationChunked] = deque()
        self.running_adapters: Dict[str, int] = {}
        self.maximum_adapter_size_id: int = None
        self.maximum_adapter_size_value: int = None
        self.preempted_requests_in_last_step: Deque[RequestGreedyAllocationChunked] = deque()

        # define dynamic vars
        self.total_blocks: int = 0
        self.batched_tokens: int = 0
        self.added_adapter_in_last_step: bool = False
        self.removed_request_in_last_step: bool = False
        self.removed_adapter_in_last_step: bool = False

        # define vars for monitoring metrics
        self.finished_requests_count = 0
        self.preempted_requests_count: int = 0
        self.itl_sum = 0
        self.ttft_sum = 0
        self.processed_input_tokens: int = 0
        self.processed_output_tokens: int = 0

    def add_request(self, request: RequestGreedyAllocationChunked):
        # updates requests
        self.running_requests.append(request)

        # updates adapters
        if request.adapter_id not in self.running_adapters:
            self.added_adapter_in_last_step = True
            self.running_adapters[request.adapter_id] = 0
            if self.maximum_adapter_size_id is None or self.maximum_adapter_size_value < request.adapter_size:
                self.maximum_adapter_size_id = request.adapter_id
                self.maximum_adapter_size_value = request.adapter_size
        self.running_adapters[request.adapter_id] += 1

        # updates tokens
        self.total_blocks += request.blocks_in_use
        self.batched_tokens += request.batched_tokens

        '''# TODO remove
        if request.adapter_id == 'dummy-328':
            print('Arrival. Request', request.id, 'r. input', request.remaining_input_tokens, 'r. output', request.remaining_output_tokens)'''

    def update_requests_after_step(self, current_time: float, step_time: float):
        # print(len(self.running_requests), current_time)

        self.preempted_requests_in_last_step = deque()
        self.added_adapter_in_last_step = False
        self.removed_request_in_last_step = False
        self.removed_adapter_in_last_step = False
        recompute_max_adapter_size: bool = False

        self.batched_tokens = 0
        aux_running_requests: Deque[RequestGreedyAllocation] = deque()
        # print('start loop 1')
        while self.running_requests:
            #print('loop 1. r.', len(self.running_requests))
            request: RequestGreedyAllocationChunked = self.running_requests.popleft()

            '''if request.adapter_id == 'dummy-328':
                print('Running. Request', request.id, ', r. input', request.remaining_input_tokens, ', r. output', request.remaining_output_tokens, ', batched', request.batched_tokens)'''

            if request.remaining_input_tokens > 0:  # prefill
                '''if request.adapter_id == 'dummy-328':
                    print('prefill')'''
                # update request with generation
                request.remaining_input_tokens -= request.batched_tokens
                if request.remaining_input_tokens == -1:  # finished prefill, # batched tokens included first output token
                    '''if request.adapter_id == 'dummy-328':
                        print('finished prefill')'''
                    request.remaining_input_tokens += 1
                    request.remaining_output_tokens -= 1
                    request.ttft = current_time - request.arrival_time
                    request.batched_tokens = 1
                else:
                    '''if request.adapter_id == 'dummy-328':
                        print('prefill on the run')'''
                    # assert (self.maximum_batched_tokens - self.batched_tokens) > 0  # TODO remove
                    new_tokens: int = min(request.remaining_input_tokens, self.maximum_batched_tokens - self.batched_tokens)
                    if new_tokens == request.remaining_input_tokens:
                        new_tokens += 1
                    request.batched_tokens = new_tokens

            else:  # after decode
                '''if request.adapter_id == 'dummy-328':
                    print('decode')'''
                # update request
                request.itl_sum += step_time
                request.remaining_output_tokens -= 1

                # check if it is the last decode step
                if request.remaining_output_tokens == 0:  # last decode step
                    '''if request.adapter_id == 'dummy-328':
                        print('finish decode')'''
                    recompute_max_adapter_size = recompute_max_adapter_size or self.__remove_request(request)
                    request.batched_tokens = 0

                    # update monitoring metrics
                    self.finished_requests_count += 1
                    self.itl_sum += request.itl_sum / request.output_tokens
                    self.ttft_sum += request.ttft
                    self.processed_input_tokens += request.input_tokens
                    self.processed_output_tokens += request.output_tokens

            # check if there is enough request blocks for next step if request did not finish
            if request.batched_tokens > 0:
                processed_tokens: int = (request.input_tokens - request.remaining_input_tokens) + (request.output_tokens - request.remaining_output_tokens)
                new_tokens: int = request.batched_tokens
                new_blocks: int = math.ceil((processed_tokens + new_tokens) / self.block_size) - request.blocks_in_use

                if new_blocks > 0:  # request blocks are not enough for next step

                    preempted_itself: bool = False
                    # assign new blocks to request, preempt other less priority requests if necessary
                    while not preempted_itself and (self.total_blocks + new_blocks) > self.maximum_blocks:
                        # print('BBBBBBBBBBBBBBB')
                        self.removed_request_in_last_step = True
                        self.preempted_requests_count += 1
                        if self.running_requests:
                            preempted_request: RequestGreedyAllocationChunked = self.running_requests.pop()
                        else:
                            preempted_itself = True
                            preempted_request: RequestGreedyAllocationChunked = request
                        self.preempted_requests_in_last_step.append(preempted_request)
                        recompute_max_adapter_size = recompute_max_adapter_size or self.__remove_request(preempted_request)
                        preempted_request.reset()

                    if not preempted_itself:
                        self.total_blocks += new_blocks
                        request.blocks_in_use += new_blocks
                        aux_running_requests.append(request)

                else:  # request blocks are enough for next step
                    aux_running_requests.append(request)

                self.batched_tokens += request.batched_tokens

            assert self.batched_tokens >= 0

        # print('finish loop 1')
        self.running_requests = aux_running_requests

        '''aux_running_adapters: Dict[str, int] = {}
        print('DDDDDDDDDDDDDDDDDDDD')
        for request in self.running_requests:
            adapter_id: str = request.adapter_id
            if adapter_id not in aux_running_adapters:
                aux_running_adapters[adapter_id] = 0
            aux_running_adapters[adapter_id] += 1
        for adapter_id, count in aux_running_adapters.items():
            assert count == self.running_adapters[adapter_id]'''

        if recompute_max_adapter_size:
            if len(self.running_requests) > 0:
                self.maximum_adapter_size_id, self.maximum_adapter_size_value = max(enumerate([request.adapter_size for request in self.running_requests]), key=lambda x: x[1])
            else:
                self.maximum_adapter_size_id = None
                self.maximum_adapter_size_value = None

    def __remove_request(self, request: RequestGreedyAllocationChunked) -> bool:
        self.removed_request_in_last_step = True
        recompute_max_adapter_size: bool = False

        # updates adapters
        self.running_adapters[request.adapter_id] -= 1
        if self.running_adapters[request.adapter_id] == 0:
            del self.running_adapters[request.adapter_id]
            self.removed_adapter_in_last_step = True
            if self.maximum_adapter_size_id == request.adapter_id:
                recompute_max_adapter_size = True

        # update dynamic vars
        self.total_blocks -= request.blocks_in_use

        return recompute_max_adapter_size

    def get_compute_prefill_tokens(self) -> int:
        raise NotImplementedError

    def get_compute_decode_tokens(self) -> int:
        return len(self.running_requests)  # TODO change interface

    def get_batched_tokens(self) -> int:
        return self.batched_tokens

    def get_running_adapters(self) -> Dict[str, int]:
        return self.running_adapters

    def get_len_running_adapters(self) -> int:
        return len(self.running_adapters)

    def get_maximum_adapter_size(self) -> int:
        return self.maximum_adapter_size_value

    def added_adapter_in_last_step(self) -> bool:
        return self.added_adapter_in_last_step
