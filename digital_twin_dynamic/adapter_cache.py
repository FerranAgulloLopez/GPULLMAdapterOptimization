from collections import OrderedDict
from typing import List, Set, Tuple, Dict

from digital_twin_dynamic.estimators.predictor_loading.predictor_loading import loading_time_predictor, CONSTANTS_PER_MODEL
from digital_twin_dynamic.structures import Request, RunningBatch


class AdapterCache:
    def __init__(
            self,
            adapter_slots: int,
            model: str,
            served_adapters: List[str],
            served_adapters_sizes: List[int]
    ):
        self.adapter_slots = adapter_slots
        self.adapters_sizes: Dict[str, int] = {}
        self.max_adapter_size: int = None
        for index in range(len(served_adapters)):
            self.adapters_sizes[served_adapters[index]] = served_adapters_sizes[index]
            if self.max_adapter_size is None or self.max_adapter_size < served_adapters_sizes[index]:
                self.max_adapter_size = served_adapters_sizes[index]

        self.loaded_adapters_in_CPU: Set = set()
        self.loaded_adapters_in_GPU: OrderedDict = OrderedDict()
        self.total_loads_from_disk: int = 0
        self.total_loads_from_memory: int = 0
        self.total_loading_time_from_disk: float = 0
        self.total_loading_time_from_memory: float = 0

        self.loading_constants = CONSTANTS_PER_MODEL[model]

    # Update adapter cache with requests to run in this step, return loading elapsed time
    def run_step(self, running_batch: RunningBatch) -> float:
        if not running_batch.added_adapter_in_last_step:
            return 0

        running_adapters: Dict[str, int] = running_batch.get_running_adapters()
        # we assume all adapters fit in the available space (should be assured by scheduler implementation)
        # touch previously loaded adapters that have still running requests to assure to not unload them
        for adapter_id in list(self.loaded_adapters_in_GPU.keys()):
            if adapter_id in running_adapters:
                self.loaded_adapters_in_GPU.move_to_end(adapter_id)  # touch adapter
        # assert len(running_batch.running_adapters) <= self.adapter_slots

        # load running and unloaded adapters
        step_loading_time: float = 0
        for adapter_id in running_adapters.keys():

            if adapter_id not in self.loaded_adapters_in_CPU:
                # load adapter from disk to CPU
                # we do not count any time because we assume that in an online system it was preloaded into CPU memory
                self.loaded_adapters_in_CPU.add(adapter_id)

            if adapter_id not in self.loaded_adapters_in_GPU:
                if len(self.loaded_adapters_in_GPU) == self.adapter_slots:
                    # unload the oldest adapter (unloading time is not considered)
                    self.loaded_adapters_in_GPU.popitem(last=False)

                # load adapter from CPU to GPU
                loading_estimation: float = loading_time_predictor(
                    [
                        self.max_adapter_size,
                        self.adapters_sizes[adapter_id]
                    ],
                    **self.loading_constants
                ) / 1000  # it is in milliseconds originally (we want it in seconds)
                step_loading_time += loading_estimation
                self.total_loading_time_from_memory += loading_estimation
                self.total_loads_from_memory += 1
                self.loaded_adapters_in_GPU[adapter_id] = None  # value is not considered here

        return step_loading_time

    def get_total_loads_from_disk(self) -> int:
        return self.total_loads_from_disk

    def get_total_loads_from_memory(self) -> int:
        return self.total_loads_from_memory

    def get_total_loading_time_from_disk(self) -> float:
        return self.total_loading_time_from_disk

    def get_total_loading_time_from_memory(self) -> float:
        return self.total_loading_time_from_memory
