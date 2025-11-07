from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from typing import List, Tuple, Dict
import random


class PlacementAlgorithmBASELINE3_2(PlacementAlgorithmInterface):

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
            Determines the placement of adapters across available servers with a totally random approach.

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

        for index_adapter in range(len(adapters)):
            server = random.choice(servers)
            adapters_servers.append(server)

        servers_adapter_slots: Dict[str, int] = {}
        for server in adapters_servers:
            if server not in servers_adapter_slots:
                servers_adapter_slots[server] = 0
            servers_adapter_slots[server] += 1

        servers_adapter_slots_final: List[int] = []
        for server in servers:
            if server in servers_adapter_slots:
                adapter_slots: int = random.randint(1, servers_adapter_slots[server])
            else:
                adapter_slots = 0
            servers_adapter_slots_final.append(adapter_slots)

        return servers_adapter_slots_final, adapters_servers
