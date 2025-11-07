import copy
import random

from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from typing import List, Tuple, Deque, Dict, Any, Optional
from collections import deque
import numpy as np
from benchmarks.lora.predict_digital_twin import predict_digital_twin


ML_MODEL_PATH_THROUGHPUT: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/llama-3.1-8b-instruct/reg/rf',
    'qwen-2.5-7b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/qwen-2.5-7b-instruct/reg/knn',
}
ML_MODEL_PATH_STARVATION: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/llama-3.1-8b-instruct/class/rf',
    'qwen-2.5-7b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/qwen-2.5-7b-instruct/class/rf',
}
GPU_MEMORY_AVAILABILITY: Dict[str, Dict[int, int]] = {
    'qwen-2.5-7b-instruct': {8: 384, 16: 384, 32: 224},
    'llama-3.1-8b-instruct': {8: 384, 32: 96, 16: 224}
}

TESTED_ADAPTERS: List[int] = [8, 16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088]
TESTED_SLOTS: List[int] = [8, 16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088]


def zigzag_sort(nums):
    nums_sorted = sorted(nums)  # ascending
    result = []
    while nums_sorted:
        if nums_sorted:  # take largest
            result.append(nums_sorted.pop(-1))
        if nums_sorted:  # take smallest
            result.append(nums_sorted.pop(0))
    return result


class PlacementAlgorithmProposalStarvation3(PlacementAlgorithmInterface):

    def __init__(self):
        return

    def define_placement(
            self,
            model: str,
            servers: List[str],
            adapters: List[str],
            adapters_ranks: List[int],
            adapters_rates: List[float],
            mean_input_length: float,
            mean_output_length: float,
            ml_model_path_throughput: Optional[str] = None,
            ml_model_path_starvation: Optional[str] = None,
    ) -> Tuple[List[int], List[str]]:
        """
            Determines the placement of adapters across available servers, it packs adapters in servers until reaching starvation, with the support of the modelling results.

            Parameters:
            - servers (List[str]): A list of server string IDs.
            - adapters (List[str]): A list of adapter string IDs.
            - adapters_ranks (List[int]): Rank associated with each adapter, linked by position with adapters input parameter.
            - adapters_rates (List[float]): Rate associated with each adapter, linked by position with adapters input parameter.
            - mean_input_length (float): Average length of input tokens per request.
            - mean_output_length (float): Average length of output tokens per request.

            Returns:
            - Tuple[List[int], List[str]]:
                - First list: Adapter slot associated with each server, linked by position with servers input parameter.
                - Second list: Server associated with each adapter, linked by position with adapters input parameter.
        """
        global ML_MODEL_PATH_THROUGHPUT, ML_MODEL_PATH_STARVATION

        if ml_model_path_throughput is None:
            ml_model_path_throughput = ML_MODEL_PATH_THROUGHPUT
        if ml_model_path_starvation is None:
            ml_model_path_starvation = ML_MODEL_PATH_STARVATION

        # group adapters per rank
        grouped_adapters: Dict[int, List[Tuple[float, str]]] = {}
        for index, adapter in enumerate(adapters):
            adapter_rank = adapters_ranks[index]
            if adapter_rank not in grouped_adapters:
                grouped_adapters[adapter_rank] = []
            grouped_adapters[adapter_rank].append((adapters_rates[index], adapter))

        # sort grouped adapters per rate and transform to queue
        grouped_sorted_adapters: Dict[int, Deque[Tuple[float, str]]] = {}
        for key, value in grouped_adapters.items():
            # grouped_sorted_adapters[key] = deque(sorted(value, key=lambda x: x[0], reverse=False))
            # random.shuffle(value)
            value = zigzag_sort(value)
            grouped_sorted_adapters[key] = deque(value)
        del grouped_adapters

        # sort grouped adapters per rank and transform to queue
        grouped_sorted_adapters: List[Tuple[int, Deque[Tuple[float, str]]]] = [
            (key, value) for key, value in grouped_sorted_adapters.items()
        ]
        grouped_sorted_adapters: Deque[Tuple[int, Deque[Tuple[float, str]]]] = deque(
            sorted(grouped_sorted_adapters, key=lambda x: x[0], reverse=True)
        )

        # transform servers list to deque
        remaining_servers: Deque[str] = deque(servers)

        # create auxiliary data structures for maintaining server already allocated workload
        self.__init_server_workload(servers)
        allocated_adapters: Dict[str, str] = {}

        # main loop, until no more adapters to allocate
        while grouped_sorted_adapters:
            # retrieve grouped adapters of specific rank
            adapter_rank, group_adapters = grouped_sorted_adapters.popleft()

            # retrieve server to use
            if not remaining_servers:
                raise Exception('Not enough servers for input workload')
            server = remaining_servers.popleft()
            server_starved: bool = False

            # try to allocate adapters
            while group_adapters and not server_starved:
                # retrieve an adapter
                adapter_rate, adapter = group_adapters.popleft()

                # greedy include adapter in server
                self.__greedy_include_adapter(
                    adapter,
                    adapter_rate,
                    adapter_rank,
                    server,
                )

                # check if we got to a testing point
                testing_point: bool = self.__check_if_testing_point(server)

                if testing_point:
                    # try to allocate all greedily included adapters
                    allocated, returning_adapters = self.__try_server_allocation(
                        model=model,
                        server=server,
                        mean_input_length=mean_input_length,
                        mean_output_length=mean_output_length,
                        ml_model_path_throughput=ml_model_path_throughput[model],
                        ml_model_path_starvation=ml_model_path_starvation[model]
                    )

                    if allocated:
                        # adapters were allocated without inducing starvation
                        # we update output placement info with allocated adapters
                        for adapter in returning_adapters:
                            allocated_adapters[adapter] = server
                        # if server has more capacity we include it again
                        if not self.__check_if_more_capacity(server):
                            server_starved = True
                            grouped_sorted_adapters.appendleft((adapter_rank, group_adapters))
                    else:
                        # adapter cannot be allocated without causing starvation
                        pending_adapters, pending_rates, pending_ranks = returning_adapters
                        # we include them back to the lists
                        # we do not include the server back to the list
                        leftover_grouped_adapters: Dict[int, Deque[Tuple[float, str]]] = {}
                        for index, adapter in enumerate(pending_adapters):
                            if pending_ranks[index] not in leftover_grouped_adapters:
                                leftover_grouped_adapters[pending_ranks[index]] = deque()
                            leftover_grouped_adapters[pending_ranks[index]].appendleft((pending_rates[index], adapter))
                        ranks: List[int] = list(leftover_grouped_adapters.keys())
                        ranks = sorted(ranks)  # order in reverse to properly include ranks in original order
                        for rank in ranks:
                            adapters_queue: Deque[Tuple[float, str]] = leftover_grouped_adapters[rank]
                            if rank == adapter_rank:
                                group_adapters.extendleft(adapters_queue)
                                grouped_sorted_adapters.appendleft((adapter_rank, group_adapters))
                            else:
                                grouped_sorted_adapters.appendleft((rank, adapters_queue))

                        # we break the loop, we need another server
                        server_starved = True

            if not grouped_sorted_adapters:
                # check if any server remains with pending adapters to test allocation
                for server in self.server_workload.keys():
                    if self.__check_if_pending_adapters(server):
                        allocated, returning_adapters = self.__try_server_allocation(
                            model=model,
                            server=server,
                            mean_input_length=mean_input_length,
                            mean_output_length=mean_output_length,
                            ml_model_path_throughput=ml_model_path_throughput[model],
                            ml_model_path_starvation=ml_model_path_starvation[model],
                        )
                        if allocated:
                            # adapters were allocated without inducing starvation
                            # we update output placement info with allocated adapters
                            for adapter in returning_adapters:
                                allocated_adapters[adapter] = server
                        else:
                            # adapter cannot be allocated without causing starvation
                            pending_adapters, pending_rates, pending_ranks = returning_adapters
                            # we include them back to the lists
                            # we do not include the server back to the list
                            leftover_grouped_adapters: Dict[int, Deque[Tuple[float, str]]] = {}
                            for index, adapter in enumerate(pending_adapters):
                                if pending_ranks[index] not in leftover_grouped_adapters:
                                    leftover_grouped_adapters[pending_ranks[index]] = deque()
                                leftover_grouped_adapters[pending_ranks[index]].appendleft((pending_rates[index], adapter))
                            ranks: List[int] = list(leftover_grouped_adapters.keys())
                            ranks = sorted(ranks)  # order in reverse to properly include ranks in original order
                            for rank in ranks:
                                adapters_queue: Deque[Tuple[float, str]] = leftover_grouped_adapters[rank]
                                if rank == adapter_rank:
                                    group_adapters.extendleft(adapters_queue)
                                    grouped_sorted_adapters.appendleft((adapter_rank, group_adapters))
                                else:
                                    grouped_sorted_adapters.appendleft((rank, adapters_queue))
                            server_starved = True

            if not server_starved:
                remaining_servers.appendleft(server)

        # transform output to right format
        servers_adapter_slots: List[int] = []
        adapters_servers: List[str] = []
        for server in servers:
            servers_adapter_slots.append(self.server_workload[server]['adapter_slots'])
        for adapter in adapters:
            adapters_servers.append(allocated_adapters[adapter])

        return servers_adapter_slots, adapters_servers

    def __init_server_workload(self, servers: List[str]) -> None:
        global TESTED_ADAPTERS, TESTED_SLOTS
        remaining_served_adapters_to_test: List[int] = copy.deepcopy(TESTED_ADAPTERS)
        remaining_adapter_slots_to_test: List[int] = copy.deepcopy(TESTED_SLOTS)
        self.server_workload: Dict[str, Dict[str, Any]] = {
            server:
                {
                    'served_adapters': 0,
                    'adapter_slots': 0,
                    'rates': [],
                    'sizes': [],
                    'remaining_served_adapters_to_test': remaining_served_adapters_to_test,
                    'remaining_adapter_slots_to_test': remaining_adapter_slots_to_test,
                    'pending_adapters': [],
                    'pending_rates': [],
                    'pending_sizes': [],
                }
            for server in servers
        }

    def __try_server_allocation(
            self,
            model: str,
            server: str,
            mean_input_length: float,
            mean_output_length: float,
            ml_model_path_throughput: str,
            ml_model_path_starvation: str,
    ) -> Tuple[bool, List[Any]]:
        global GPU_MEMORY_AVAILABILITY

        server_init_workload: Dict[str, Any] = self.server_workload[server]

        assert server_init_workload['remaining_adapter_slots_to_test'][0] <= \
               server_init_workload['remaining_served_adapters_to_test'][0]

        served_adapters_to_test: int = server_init_workload['remaining_served_adapters_to_test'][0]
        rates_to_test: List[float] = server_init_workload['rates'] + server_init_workload['pending_rates']
        sizes_to_test: List[int] = server_init_workload['sizes'] + server_init_workload['pending_sizes']
        if server_init_workload['served_adapters'] + len(server_init_workload['pending_adapters']) < \
                server_init_workload['remaining_served_adapters_to_test'][0]:
            missing_number: int = served_adapters_to_test - len(server_init_workload['pending_adapters'])
            rates_to_test += [np.mean(np.asarray(rates_to_test))] * missing_number
            sizes_to_test += [np.min(np.asarray(sizes_to_test))] * missing_number

        if server_init_workload['adapter_slots'] > 0:
            updated_slots: int = server_init_workload['adapter_slots']
            remaining_adapter_slots_to_test: List[int] = server_init_workload['remaining_adapter_slots_to_test']
            input_ml_data: Dict[str, Any] = self.__create_ml_input(
                rates=rates_to_test,
                sizes=sizes_to_test,
                served_adapters=served_adapters_to_test,
                adapter_slots=server_init_workload['adapter_slots'],
            )
        elif server_init_workload['remaining_adapter_slots_to_test']:
            updated_slots: int = server_init_workload['remaining_adapter_slots_to_test'][0]
            remaining_adapter_slots_to_test: List[int] = server_init_workload['remaining_adapter_slots_to_test'][1:]
            input_ml_data: Dict[str, Any] = self.__create_ml_input(
                rates=rates_to_test,
                sizes=sizes_to_test,
                served_adapters=served_adapters_to_test,
                adapter_slots=server_init_workload['remaining_adapter_slots_to_test'][0],
            )
        else:
            input_ml_data = None

        assert input_ml_data is not None

        starvation: bool = False
        adapter_slots = input_ml_data['adapter_slots']
        max_size = input_ml_data['max_size']
        assert max_size in GPU_MEMORY_AVAILABILITY[model]
        memory_error = False if adapter_slots <= GPU_MEMORY_AVAILABILITY[model][max_size] else True
        if memory_error:
            starvation = True

        if not memory_error:
            output_ml_data: Dict[str, Any] = predict_digital_twin(
                regression=False,
                model_path=ml_model_path_starvation,
                x_features=input_ml_data,
                y_features_to_predict=['starvation'],
            )
            starvation: bool = bool(output_ml_data['starvation'][0])

            if starvation:
                a = 0

            if starvation and updated_slots < served_adapters_to_test:
                while updated_slots < server_init_workload['served_adapters'] and starvation and not memory_error:
                    updated_slots: int = remaining_adapter_slots_to_test[0]
                    remaining_adapter_slots_to_test: List[int] = remaining_adapter_slots_to_test[1:]

                    input_ml_data: Dict[str, Any] = self.__create_ml_input(
                        rates=rates_to_test,
                        sizes=sizes_to_test,
                        served_adapters=served_adapters_to_test,
                        adapter_slots=updated_slots,
                    )

                    starvation: bool = False
                    adapter_slots = input_ml_data['adapter_slots']
                    max_size = input_ml_data['max_size']
                    assert max_size in GPU_MEMORY_AVAILABILITY[model]
                    memory_error = False if adapter_slots <= GPU_MEMORY_AVAILABILITY[model][max_size] else True
                    if memory_error:
                        starvation = True

                    if not memory_error:
                        output_ml_data: Dict[str, Any] = predict_digital_twin(
                            regression=False,
                            model_path=ml_model_path_starvation,
                            x_features=input_ml_data,
                            y_features_to_predict=['starvation'],
                        )
                        starvation: bool = bool(output_ml_data['starvation'][0])

        if starvation:
            # starvation, we cannot allocate in this server
            # we return pending adapters
            pending_adapters: List[str] = server_init_workload['pending_adapters']
            pending_rates: List[float] = server_init_workload['pending_rates']
            pending_sizes: List[int] = server_init_workload['pending_sizes']
            # we update the server allocation
            self.server_workload[server]['pending_adapters'] = []
            self.server_workload[server]['pending_rates'] = []
            self.server_workload[server]['pending_sizes'] = []
            return False, (pending_adapters, pending_rates, pending_sizes)
        else:
            # we can allocate without starvation
            # we return the allocated adapters
            allocated_adapters: List[str] = server_init_workload['pending_adapters']
            # we update the server allocation
            self.server_workload[server]['served_adapters'] = server_init_workload['remaining_served_adapters_to_test'][0]
            self.server_workload[server]['adapter_slots'] = updated_slots
            self.server_workload[server]['rates'] = server_init_workload['rates'] + server_init_workload['pending_rates']
            self.server_workload[server]['sizes'] = server_init_workload['sizes'] + server_init_workload['pending_sizes']
            self.server_workload[server]['remaining_served_adapters_to_test'] = server_init_workload['remaining_served_adapters_to_test'][1:]
            self.server_workload[server]['remaining_adapter_slots_to_test'] = remaining_adapter_slots_to_test
            self.server_workload[server]['pending_adapters'] = []
            self.server_workload[server]['pending_rates'] = []
            self.server_workload[server]['pending_sizes'] = []
            return True, allocated_adapters

    def __greedy_include_adapter(
            self,
            adapter: str,
            adapter_rate: float,
            adapter_rank: int,
            server: str
    ):
        self.server_workload[server]['pending_adapters'].append(adapter)
        self.server_workload[server]['pending_rates'].append(adapter_rate)
        self.server_workload[server]['pending_sizes'].append(adapter_rank)

    def __check_if_testing_point(
            self,
            server: str
    ) -> bool:
        return len(self.server_workload[server]['pending_adapters']) + self.server_workload[server]['served_adapters'] == self.server_workload[server]['remaining_served_adapters_to_test'][0]

    def __create_ml_input(
            self,
            rates: List[float],
            sizes: List[int],
            served_adapters: int,
            adapter_slots: int,
    ) -> Dict[str, Any]:
        return {
            'sum_rate': np.sum(rates),
            'std_rate': np.std(rates),
            'max_size': np.max(sizes),
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'served_adapters': served_adapters,
            'adapter_slots': adapter_slots,
        }

    def __check_if_more_capacity(
            self,
            server: str
    ) -> bool:
        return len(self.server_workload[server]['remaining_served_adapters_to_test']) > 0

    def __check_if_pending_adapters(
            self,
            server: str
    ) -> bool:
        return len(self.server_workload[server]['pending_adapters']) > 0
