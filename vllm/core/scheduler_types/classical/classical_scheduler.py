import json
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from prometheus_client import (Gauge)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler, SchedulingBudget, SchedulerPrefillOutputs, SchedulerRunningOutputs
from vllm.engine.metrics import StatLogger
from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)
from vllm.core.policy import Policy, PolicyFactory

logger = init_logger(__name__)


class ClassicalScheduler(Scheduler):

    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig],
            queue_limit_policy: str,
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        self.policy = PolicyFactory.get_policy(policy_name="lost_work")  # important for ordering running preempted requests

    def _schedule_prefills(
            self,
            waiting_queue: Deque[SequenceGroup],
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        assert curr_loras is not None

        new_waiting_queue: Deque = deque()
        seq_group_dict: Dict[str, SequenceGroup] = {seq_group.request_id: seq_group for seq_group in waiting_queue}

        # order seq groups
        MAX_STEPS = 300
        while len(new_waiting_queue) < len(waiting_queue) or len(new_waiting_queue) > MAX_STEPS:
            min_cost: int = None
            min_seq_group: str = None
            for request_id_1, seq_group_1 in seq_group_dict.items():
                seq_group_cost = 0
                for request_id_2, seq_group_2 in seq_group_dict.items():
                    if request_id_1 != request_id_2:
                        seq_group_cost += self.__compute_cost(
                            seq_group_1,
                            seq_group_2,
                            curr_loras
                        )
                if min_cost is None or seq_group_cost < min_cost:
                    min_cost = seq_group_cost
                    min_seq_group = request_id_1
            new_waiting_queue.append(seq_group_dict[min_seq_group])
            del seq_group_dict[min_seq_group]

        # send to system
        remaining_waiting, prefills = super()._schedule_prefills(
            new_waiting_queue,
            budget,
            curr_loras,
            enable_chunking
        )

        # update parameter values
        for seq_group in seq_group_dict.values():
            remaining_waiting.append(seq_group)

        return remaining_waiting, prefills

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        raise NotImplementedError("Classical scheduler does not implement preemption by swapping")  # TODO implement

    def do_alternative_logging(self, stat_logger: StatLogger) -> None:
        super().do_alternative_logging(stat_logger)
        pass

    def __compute_cost(
            self,
            seq_group_1: SequenceGroup,
            seq_group_2: SequenceGroup,
            curr_loras: Set[int]
    ) -> int:
        def __request_allocation_time(size: int) -> int:
            return 0

        # pre -> seq_group_1 != seq_group_2
        assert seq_group_1.lora_int_id > 0
        assert seq_group_2.lora_int_id > 0

        cost: int = 0
        seq_group_1_size: int = sum(seq.get_prompt_len() for seq in seq_group_1.get_seqs())
        seq_group_2_size: int = sum(seq.get_prompt_len() for seq in seq_group_2.get_seqs())

        cost += __request_allocation_time()
        if seq_group_1.lora_int_id in curr_loras:
            pass

        return 0
