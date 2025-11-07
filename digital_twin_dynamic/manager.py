import heapq
import time
import json
from copy import deepcopy
from typing import List, Optional, Tuple, Set, Dict

from digital_twin_dynamic.adapter_cache import AdapterCache
from digital_twin_dynamic.engine import Engine
#from digital_twin_dynamic.scheduler import Scheduler
#from digital_twin_dynamic.scheduler_optimized import SchedulerOptimized
#from digital_twin_dynamic.scheduler_greedy_allocation import SchedulerGreedyAllocation
#from digital_twin_dynamic.scheduler_greedy_allocation_optimized import SchedulerGreedyAllocationOptimized
from digital_twin_dynamic.scheduler_greedy_allocation_optimized_chunked import SchedulerGreedyAllocationOptimizedChunked
from digital_twin_dynamic.structures import Request, RequestGreedyAllocation, RequestGreedyAllocationChunked, RunningBatch, RunningBatchGreedyAllocation, RunningBatchGreedyAllocationChunked
from digital_twin_dynamic.estimators.predictor_scheduler_overhead.predictor_scheduler_overhead import scheduler_overhead_predictor


class DynamicSimulatorManager:
    def __init__(
            self,
            start_time: float,  # start time of simulation in seconds
            finish_time: float,  # finish time of simulation in seconds
            model: str,  # name of the model being used
            adapter_slots: int,  # GPU slots for adapters
            served_adapters: List[str],  # adapters' ids that can be served
            served_adapters_sizes: List[int],  # rank size of prior adapters
            available_gpu_memory: int,  # max token capacity after loading model and adapter weights
            max_num_batched_tokens: int,  # max token number to batch together
            request_arrivals: List[Tuple[float, Tuple[int, int, str]]],  # heap with all request arrivals with form (arrival_time, (input_tokens, output_tokens, adapter_id)). Arrival time should be in seconds
            print_outcome: Optional[bool] = True,  # print outcome in standard output
            include_computation_overhead: Optional[bool] = False,  # include computation overhead in engine estimation
            include_loading_overhead: Optional[bool] = True,  # include adapter loading overhead in estimation
            include_preemption: Optional[bool] = True,  # include preemption simulation in scheduler
            include_network_collapse: Optional[int] = None,  # include network collapse simulation, do not include more request in the waiting queue than the provided number
            debug_path: Optional[str] = None  # if not None, simulator works in debug mode and stores data in given directory
    ):
        # definition of execution constants
        self.start_time: float = start_time
        self.finish_time: float = finish_time
        self.adapter_slots: int = adapter_slots
        self.served_adapters: Dict[str, int] = {}
        for index in range(len(served_adapters)):
            adapter_id: str = served_adapters[index]
            adapter_size: int = served_adapters_sizes[index]
            if adapter_id not in self.served_adapters:
                self.served_adapters[adapter_id] = adapter_size
        self.served_adapters_num: int = len(self.served_adapters)
        self.request_arrivals: List[Tuple[float, Tuple[int, int, str]]] = request_arrivals
        self.total_arrivals: int = len(self.request_arrivals)
        self.include_loading_overhead: bool = include_loading_overhead
        self.include_network_collapse: int = include_network_collapse
        self.debug_path: str = debug_path

        # definition of other constants
        self.print_outcome: bool = print_outcome

        # definition of modules
        if include_preemption:
            self.scheduler = SchedulerGreedyAllocationOptimizedChunked(self.adapter_slots, available_gpu_memory, max_num_batched_tokens, self.debug_path)
            self.request_cls = RequestGreedyAllocationChunked
            self.batch_cls = RunningBatchGreedyAllocationChunked
        else:
            self.scheduler = SchedulerOptimized(self.adapter_slots, available_gpu_memory, max_num_batched_tokens, self.debug_path)
            self.request_cls = Request
            self.batch_cls = RunningBatch
        self.engine: Engine = Engine(self.adapter_slots, model, include_computation_overhead)
        self.adapter_cache: AdapterCache = AdapterCache(self.adapter_slots, model, served_adapters, served_adapters_sizes)

    def simulate(self) -> Dict[str, float]:
        # debug timers
        arrivals_time: float = 0
        scheduler_time: float = 0
        scheduler_overhead_time: float = 0
        adapter_cache_time: float = 0
        engine_time: float = 0

        init_simulation_time: float = time.perf_counter()

        # run simulation
        request_unique_id: int = 0
        start_step_time: float = self.start_time
        estimated_scheduler_time: float = 0
        while start_step_time < self.finish_time and self.request_arrivals:
            end_step_time: float = start_step_time

            # compute new arrivals
            init_time: float = time.perf_counter()
            while self.request_arrivals and self.request_arrivals[0][0] <= start_step_time:

                if self.include_network_collapse is not None and self.scheduler.get_num_waiting_requests() > self.include_network_collapse:
                    break

                arrival_time, (input_tokens, output_tokens, adapter_id) = heapq.heappop(self.request_arrivals)
                if adapter_id not in self.served_adapters:
                    raise ValueError('Received request for an unknown adapter')

                # add to scheduler
                self.scheduler.new_arrival(self.request_cls(request_unique_id, arrival_time, input_tokens, output_tokens, adapter_id, self.served_adapters[adapter_id]), arrival_time)
                request_unique_id += 1
            arrivals_time += time.perf_counter() - init_time

            # schedule and retrieve batch
            init_time: float = time.perf_counter()
            running_batch: RunningBatch = self.scheduler.run_step(end_step_time)
            scheduler_time += time.perf_counter() - init_time
            if self.include_loading_overhead:
                init_time: float = time.perf_counter()
                elapsed_time_scheduler: float = scheduler_overhead_predictor(
                    self.scheduler.get_num_running_requests(),
                    self.scheduler.get_num_waiting_requests(),
                    self.served_adapters_num,
                    self.adapter_slots
                ) / 1000  # it is in milliseconds originally (we want it in seconds)
                end_step_time += elapsed_time_scheduler
                estimated_scheduler_time += elapsed_time_scheduler
                scheduler_overhead_time += time.perf_counter() - init_time

            # skip to next arrival if not pending requests
            if self.scheduler.get_num_running_requests() == 0:
                if len(self.request_arrivals) > 0:
                    start_step_time = self.request_arrivals[0][0]
                else:
                    start_step_time = self.finish_time
                continue

            # upload missing adapters (updates end_step_time)
            if self.include_loading_overhead:
                init_time: float = time.perf_counter()
                loading_time = self.adapter_cache.run_step(running_batch)
                end_step_time += loading_time
                adapter_cache_time += time.perf_counter() - init_time

            # perform batch computation (updates end_step_time)
            init_time: float = time.perf_counter()
            computation_time = self.engine.run_step(running_batch)
            end_step_time += computation_time
            engine_time += time.perf_counter() - init_time

            # update running batch
            init_time: float = time.perf_counter()
            running_batch.update_requests_after_step(end_step_time, end_step_time - start_step_time)
            scheduler_time += time.perf_counter() - init_time

            # update time for next step
            start_step_time = end_step_time

        # prepare output
        output = {
            'duration': time.perf_counter() - init_simulation_time,
            'estimated_duration': start_step_time - self.start_time,
            'input_throughput': self.scheduler.get_processed_input_tokens() / (start_step_time - self.start_time),
            'output_throughput': self.scheduler.get_processed_output_tokens() / (start_step_time - self.start_time),
            'total_throughput': (self.scheduler.get_processed_input_tokens() + self.scheduler.get_processed_output_tokens()) / (start_step_time - self.start_time),
            'itl': self.scheduler.get_mean_itl() * 1000,
            'ttft': self.scheduler.get_mean_ttft(self.finish_time - self.start_time) * 1000,
            'total_loads_from_disk': self.adapter_cache.get_total_loads_from_disk(),
            'total_loads_from_memory': self.adapter_cache.get_total_loads_from_memory(),
            'loading_time_from_disk': self.adapter_cache.get_total_loading_time_from_disk(),
            'loading_time_from_memory': self.adapter_cache.get_total_loading_time_from_memory(),
            'arrivals': self.total_arrivals,
            'finished_requests': self.scheduler.get_num_finished_requests(),
            'scheduler_time': estimated_scheduler_time
        }

        if self.print_outcome:
            print(f'---Simulation End---')
            print(f'#Simulation results')
            print(json.dumps(output, indent=4))
            print(f'#Debug simulation ')
            print(
                f'Total elapsed time: {time.perf_counter() - init_simulation_time}. '
                f'Arrivals time: {arrivals_time} '
                f'Scheduler time: {scheduler_time} '
                f'Scheduler overhead time: {scheduler_overhead_time} '
                f'Adapter cache time: {adapter_cache_time} '
                f'Engine time: {engine_time} '
            )

        if self.debug_path:
            self.scheduler.debug_chart()

        return output
