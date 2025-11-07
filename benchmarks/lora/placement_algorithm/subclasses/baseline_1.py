from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from typing import List, Tuple, Deque, Dict, Optional
from collections import deque


BENCHMARK_MAX_ACHIEVABLE_TOTAL_THROUGHPUT: Dict[str, float] = {
    'llama-3.1-8b-instruct': 9360.030096854402 + 8454.502307321163,
    'qwen-2.5-7b-instruct': 9486.127903110408 + 8453.898321446994
}

class PlacementAlgorithmBASELINE1(PlacementAlgorithmInterface):

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
            benchmark_max_achievable_total_throughput: Dict[str, float] = None
    ) -> Tuple[List[int], List[str]]:
        """
            Determines the placement of adapters across available servers, it packs adapters in servers until benchmark maximum achievable throughput. Slots are set to adapters being served.

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
        global BENCHMARK_MAX_ACHIEVABLE_TOTAL_THROUGHPUT

        if benchmark_max_achievable_total_throughput is None:
            maximum_achievable: float = BENCHMARK_MAX_ACHIEVABLE_TOTAL_THROUGHPUT[model]
        else:
            maximum_achievable = benchmark_max_achievable_total_throughput[model]

        rate_per_adapter: Dict[str, float] = {adapter: adapters_rates[index] for index, adapter in enumerate(adapters)}
        aggregated_rate_per_server_tokens: Dict[str, float] = {server: 0 for server in servers}

        remaining_servers: Deque[str] = deque(servers)
        remaining_adapters: Deque[str] = deque(adapters)

        servers_adapter_slots_dict: Dict[str, int] = {server: 0 for server in servers}
        adapters_servers_dict: Dict[str, str] = {}

        while remaining_adapters:
            adapter: str = remaining_adapters.popleft()
            adapter_rate: float = rate_per_adapter[adapter]
            if not remaining_servers:
                raise Exception('Not enough servers for input rate')
            server: str = remaining_servers.popleft()
            server_aggregated_rate_tokens: float = aggregated_rate_per_server_tokens[server]

            adapter_rate_tokens: float = adapter_rate * (mean_input_length + mean_output_length)
            if (server_aggregated_rate_tokens + adapter_rate_tokens) <= maximum_achievable:
                aggregated_rate_per_server_tokens[server] += adapter_rate_tokens
                adapters_servers_dict[adapter] = server
                servers_adapter_slots_dict[server] += 1
                remaining_servers.appendleft(server)
            else:
                remaining_adapters.appendleft(adapter)

        servers_adapter_slots: List[int] = []
        adapters_servers: List[str] = []
        for server in servers:
            servers_adapter_slots.append(servers_adapter_slots_dict[server])
        for adapter in adapters:
            adapters_servers.append(adapters_servers_dict[adapter])

        return servers_adapter_slots, adapters_servers
