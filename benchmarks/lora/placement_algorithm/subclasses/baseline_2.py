from benchmarks.lora.placement_algorithm.subclasses.baseline_1 import PlacementAlgorithmBASELINE1
from typing import List, Tuple, Deque, Dict
import heapq
from collections import deque


class PlacementAlgorithmBASELINE2(PlacementAlgorithmBASELINE1):

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
            Determines the placement of adapters across available servers, it packs adapters in servers until benchmark maximum achievable throughput. Slots are set to 1/3 of adapters being served.

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
        servers_adapter_slots, adapters_servers = super().define_placement(
            model,
            servers,
            adapters,
            adapters_ranks,
            adapters_rates,
            mean_input_length,
            mean_output_length
        )

        for index, slots in enumerate(servers_adapter_slots):
            servers_adapter_slots[index] = max(1, round(slots / 2))

        return servers_adapter_slots, adapters_servers
