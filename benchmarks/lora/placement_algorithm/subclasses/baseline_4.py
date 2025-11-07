from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from typing import List, Tuple, Dict
import heapq


class PlacementAlgorithmBASELINE4(PlacementAlgorithmInterface):

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
            mean_output_length: float
    ) -> Tuple[List[int], List[str]]:
        """
            Determines the placement of adapters across available servers with a greedy minimization approach, where each adapter is assigned to the server with the lowest aggregated rate per second.

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
        adapters_servers: List[str] = []

        aggregated_rate_per_server: List[Tuple[float, str]] = [(0, server) for server in servers]
        heapq.heapify(aggregated_rate_per_server)

        for index_adapter in range(len(adapters)):
            aux_aggregated_rate, min_server = heapq.heappop(aggregated_rate_per_server)
            adapters_servers.append(min_server)
            heapq.heappush(aggregated_rate_per_server, (aux_aggregated_rate + adapters_rates[index_adapter], min_server))

        servers_adapter_slots: Dict[str, int] = {}
        for server in adapters_servers:
            if server not in servers_adapter_slots:
                servers_adapter_slots[server] = 0
            servers_adapter_slots[server] += 1

        servers_adapter_slots_final: List[int] = []
        for server in servers:
            servers_adapter_slots_final.append(servers_adapter_slots[server])

        return servers_adapter_slots_final, adapters_servers
