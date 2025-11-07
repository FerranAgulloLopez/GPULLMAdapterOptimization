from abc import ABC, abstractmethod
from typing import List, Tuple, Type


class PlacementAlgorithmInterface(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError('Interface class cannot be instantiated')

    @abstractmethod
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
            Determines the placement of adapters across available servers.

            Parameters:
            - servers (List[int]): A list of server integer IDs.
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
        pass
