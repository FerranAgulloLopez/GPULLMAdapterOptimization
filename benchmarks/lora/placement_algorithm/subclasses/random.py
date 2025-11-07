from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from typing import List, Tuple
import random


class PlacementAlgorithmRandom(PlacementAlgorithmInterface):

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
            Determines the placement of adapters across available servers with a random approach, where each adapter is randomly assigned a server.

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
        raise NotImplementedError('Not updated')
        servers_adapter_slots: List[int] = [128] * len(servers)
        adapters_servers: List[str] = []

        for _ in adapters:
            random_server_id: str = random.choice(servers)
            adapters_servers.append(random_server_id)

        return servers_adapter_slots, adapters_servers
