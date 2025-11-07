import copy

from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from typing import List, Tuple, Deque, Dict, Any, Optional
from collections import deque
import numpy as np
from benchmarks.lora.predict_digital_twin import predict_digital_twin


ML_MODEL_PATH: Dict[str, str] = {
    'llama-3.1-8b-instruct': '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/models/trained_ml/llama-31-8b-instruct'
}


class PlacementAlgorithmProposal(PlacementAlgorithmInterface):

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
            ml_model_path: Optional[str] = None,
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
        global ML_MODEL_PATH

        if ml_model_path is None:
            ml_model_path = ML_MODEL_PATH

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
            grouped_sorted_adapters[key] = deque(sorted(value, key=lambda x: x[0], reverse=True))
        del grouped_adapters

        # sort grouped adapters per rank and transform to queue
        grouped_sorted_adapters: List[Tuple[int, Deque[Tuple[float, str]]]] = [
            (key, value) for key, value in grouped_sorted_adapters.items()
        ]
        grouped_sorted_adapters: Deque[Tuple[int, Deque[Tuple[float, str]]]] = deque(
            sorted(grouped_sorted_adapters, key=lambda x: x[0])
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
                raise Exception('Not enough servers for input rate')
            server = remaining_servers.popleft()

            # try to allocate adapters
            leftover_adapters: Deque[Tuple[float, str]] = deque()
            while group_adapters:
                # retrieve an adapter
                adapter_rate, adapter = group_adapters.popleft()

                # try to allocate
                can_allocate, slot_needed = self.__try_server_starvation_with_adapter(
                    model,
                    server,
                    adapter_rate,
                    adapter_rank,
                    mean_input_length,
                    mean_output_length,
                    ml_model_path
                )

                if can_allocate:
                    # allocate
                    self.server_workload[server] = self.__include_adapter(slot_needed, adapter_rate, adapter_rank, self.server_workload[server])
                    allocated_adapters[adapter] = server
                else:
                    # try with other adapters with lower rate
                    # (store pending adapters for allocating them in another server)
                    unallocated_rate: float = adapter_rate
                    leftover_adapters.append((adapter_rate, adapter))
                    found: bool = False
                    while not found and group_adapters:
                        adapter_rate, adapter = group_adapters.popleft()
                        if adapter_rate < unallocated_rate * 0.9:
                            found = True
                            group_adapters.appendleft((adapter_rate, adapter))
                        else:
                            leftover_adapters.append((adapter_rate, adapter))

            if leftover_adapters:
                # if not all adapters could be allocated
                # we do not put server back to pool of servers for other ranks  # TODO rethink this
                # add leftover adapters
                grouped_sorted_adapters.appendleft((adapter_rank, leftover_adapters))
            else:
                # free space in server, we put it back to the pool
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
        self.server_workload: Dict[str, Dict[str, Any]] = {
            server:
                {
                    'sum_rate': None,
                    'std_rate': None,
                    'max_size': None,
                    'mean_size': None,
                    'std_size': None,
                    'served_adapters': 0,
                    'adapter_slots': 0,
                    'rates': [],
                    'sizes': [],
                }
            for server in servers
        }

    def __try_server_starvation_with_adapter(
            self,
            model: str,
            server: str,
            adapter_rate: float,
            adapter_rank: int,
            mean_input_length: float,
            mean_output_length: float,
            ml_model_path,
    ) -> Tuple[bool, bool]:
        if self.server_workload[server]['adapter_slots'] > 0:
            input_data_plus_adapter = self.__include_adapter(
                slot_needed=False,
                adapter_rate=adapter_rate,
                adapter_rank=adapter_rank,
                server_workload=self.server_workload[server]
            )
        input_data_plus_adapter_plus_slot = self.__include_adapter(
            slot_needed=True,
            adapter_rate=adapter_rate,
            adapter_rank=adapter_rank,
            server_workload=self.server_workload[server]
        )

        if self.server_workload[server]['adapter_slots'] > 0:
            output_data_plus_adapter: Dict[str, Any] = predict_digital_twin(
                model_path=ml_model_path[model],
                x_features=input_data_plus_adapter,
                y_features_to_predict=['total_throughput'],
            )
        output_data_plus_adapter_plus_slot: Dict[str, Any] = predict_digital_twin(
            model_path=ml_model_path[model],
            x_features=input_data_plus_adapter_plus_slot,
            y_features_to_predict=['total_throughput'],
        )

        if self.server_workload[server]['adapter_slots'] > 0:
            total_throughput_plus_adapter: float = output_data_plus_adapter['total_throughput'][0]
        else:
            total_throughput_plus_adapter: float = -1.0
        total_throughput_plus_adapter_plus_slot: float = output_data_plus_adapter_plus_slot['total_throughput'][0]

        ideal_throughput: float = sum(input_data_plus_adapter_plus_slot['rates']) * (mean_input_length + mean_output_length)

        if max(total_throughput_plus_adapter, total_throughput_plus_adapter_plus_slot) < (ideal_throughput * 0.9):
            return False, None
        else:
            return True, total_throughput_plus_adapter_plus_slot > total_throughput_plus_adapter


    def __include_adapter(
            self,
            slot_needed: bool,
            adapter_rate: float,
            adapter_rank: int,
            server_workload: Dict[str, Any],
    ) -> Dict[str, Any]:
        rates: List[float] = copy.deepcopy(server_workload['rates'])
        rates.append(adapter_rate)
        sum_rate: float = np.sum(np.asarray(rates))
        std_rate: float = np.std(np.asarray(rates))

        sizes: List[int] = copy.deepcopy(server_workload['sizes'])
        sizes.append(adapter_rank)
        max_size: int = adapter_rank if server_workload['max_size'] is None else max(server_workload['max_size'], adapter_rank)
        mean_size: float = np.mean(np.asarray(sizes))
        std_size: float = np.std(np.asarray(sizes))

        served_adapters: int = server_workload['served_adapters'] + 1
        adapter_slots: int = server_workload['adapter_slots'] + 1 if slot_needed else server_workload['adapter_slots']

        return {
            'sum_rate': sum_rate,
            'std_rate': std_rate,
            'max_size': max_size,
            'mean_size': mean_size,
            'std_size': std_size,
            'served_adapters': served_adapters,
            'adapter_slots': adapter_slots,
            'rates': rates,
            'sizes': sizes,
        }
