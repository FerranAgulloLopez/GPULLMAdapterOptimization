import json
from collections import deque
from typing import Dict, Optional, Set, Tuple, Union, Iterable

from prometheus_client import (Gauge)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.policy import Policy
from vllm.core.scheduler import Scheduler, SchedulingBudget, SchedulerPrefillOutputs, SchedulerRunningOutputs
from vllm.engine.metrics import StatLogger
from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)

logger = init_logger(__name__)


class VTCCounter(Scheduler):

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        prefill_weight: Optional[float] = 1,
        decode_weight: Optional[float] = 2,
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        self.vtc_seq_group_user_relation: Dict[str, str] = {}

        self.vtc_cost_by_user: Dict[str, float] = {}
        self.vtc_waiting_seq_groups_by_user: Dict[str, int] = {}
        self.vtc_running_seq_groups_by_user: Dict[str, int] = {}  # only for visualization
        self.vtc_last_waiting_user: str = None
        self.vtc_prefill_weight: float = prefill_weight
        self.vtc_decode_weight: float = decode_weight

        self.vtc_initialized_visualization: bool = False

    def add_seq_group(self, seq_group: SequenceGroup, user_id: Optional[str] = None) -> None:
        super().add_seq_group(seq_group, user_id)
        if user_id is not None:
            self.vtc_seq_group_user_relation[seq_group.request_id] = user_id
            if user_id not in self.vtc_cost_by_user:
                self.vtc_cost_by_user[user_id] = 0
            if user_id not in self.vtc_waiting_seq_groups_by_user:
                if len(self.vtc_waiting_seq_groups_by_user) == 0:
                    if self.vtc_last_waiting_user is not None:
                        self.vtc_cost_by_user[user_id] = max(
                            self.vtc_cost_by_user[user_id],
                            self.vtc_cost_by_user[self.vtc_last_waiting_user]
                        )
                else:
                    lowest_counter_with_waiting_seq_groups = min(
                        [
                            self.vtc_cost_by_user[user_id]
                            for user_id in self.vtc_waiting_seq_groups_by_user.keys()
                        ]
                    )
                    self.vtc_cost_by_user[user_id] = max(
                        self.vtc_cost_by_user[user_id],
                        lowest_counter_with_waiting_seq_groups
                    )
                self.vtc_waiting_seq_groups_by_user[user_id] = 0
            self.vtc_waiting_seq_groups_by_user[user_id] += 1
            if user_id not in self.vtc_running_seq_groups_by_user:
                self.vtc_running_seq_groups_by_user[user_id] = 0

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        super().abort_seq_group(request_id)
        if isinstance(request_id, Iterable):
            for request_id_single in request_id:
                self.__abort_seq_group_individual(request_id_single)
        else:
            self.__abort_seq_group_individual(request_id)

    def __abort_seq_group_individual(self, request_id: Union[str, Iterable[str]]) -> None:
        if request_id in self.vtc_seq_group_user_relation:
            user_id: str = self.vtc_seq_group_user_relation[request_id]

            self.vtc_running_seq_groups_by_user[user_id] -= 1
            # TODO DEBUG
            if self.vtc_running_seq_groups_by_user[user_id] < 0:
                self.vtc_running_seq_groups_by_user[user_id] = 0
            # if self.vtc_running_seq_groups_by_user[user_id] == 0:
            #     del self.vtc_running_seq_groups_by_user[user_id]

            del self.vtc_seq_group_user_relation[request_id]  # TODO caution with ignored seq groups

    def _schedule_prefills(
            self,
            waiting_queue: deque,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        remaining_waiting, prefills = super()._schedule_prefills(
            waiting_queue,
            budget,
            curr_loras,
            enable_chunking
        )

        for seq_group in prefills.seq_groups:
            seq_group = seq_group.seq_group  # Coming from vLLM bug, received object is ScheduledSequenceGroup instead
            if seq_group.request_id in self.vtc_seq_group_user_relation:
                user_id: str = self.vtc_seq_group_user_relation[seq_group.request_id]

                self.vtc_waiting_seq_groups_by_user[user_id] -= 1
                # TODO DEBUG
                if self.vtc_waiting_seq_groups_by_user[user_id] < 0:
                    self.vtc_waiting_seq_groups_by_user[user_id] = 0

                self.vtc_running_seq_groups_by_user[user_id] += 1

                self.vtc_cost_by_user[user_id] += sum(
                    [
                        seq.get_prompt_len()
                        for seq in seq_group.get_seqs()
                    ]
                ) * self.vtc_prefill_weight

                self.vtc_last_waiting_user = user_id

        for seq_group in prefills.ignored_seq_groups:
            if seq_group.request_id in self.vtc_seq_group_user_relation:
                user_id: str = self.vtc_seq_group_user_relation[seq_group.request_id]
                # del self.vtc_seq_group_user_relation[seq_group.request_id]  # TODO check if not duplicate of abort_seq_group method

                self.vtc_waiting_seq_groups_by_user[user_id] -= 1
                # TODO DEBUG
                if self.vtc_waiting_seq_groups_by_user[user_id] < 0:
                    self.vtc_waiting_seq_groups_by_user[user_id] = 0

        return remaining_waiting, prefills

    def _schedule_running(
            self,
            running_queue: deque,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            policy: Policy,
            enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        remaining_running, running_scheduled = super()._schedule_running(
            running_queue,
            budget,
            curr_loras,
            policy,
            enable_chunking
        )

        for seq_group in running_scheduled.prefill_seq_groups:
            seq_group = seq_group.seq_group  # Coming from vLLM bug, received object is ScheduledSequenceGroup instead
            if seq_group.request_id in self.vtc_seq_group_user_relation:
                user_id: str = self.vtc_seq_group_user_relation[seq_group.request_id]
                self.vtc_cost_by_user[user_id] += 1 * self.vtc_decode_weight  # TODO retrieve real new output len

        for seq_group in running_scheduled.decode_seq_groups:
            seq_group = seq_group.seq_group  # Coming from vLLM bug, received object is ScheduledSequenceGroup instead
            if seq_group.request_id in self.vtc_seq_group_user_relation:
                user_id: str = self.vtc_seq_group_user_relation[seq_group.request_id]
                self.vtc_cost_by_user[user_id] += 1 * self.vtc_decode_weight  # TODO retrieve real new output len

        return remaining_running, running_scheduled

    def do_alternative_logging(self, stat_logger: StatLogger) -> None:
        super().do_alternative_logging(stat_logger)
        '''
        if not self.vtc_initialized_visualization:
            stat_logger.metrics.vtc_cost_by_user = {}
            stat_logger.metrics.vtc_running_by_user = {}
            stat_logger.metrics.vtc_waiting_by_user = {}
            self.vtc_initialized_visualization = True
        for user, vtc_cost in self.vtc_cost_by_user.items():
            if user not in stat_logger.metrics.vtc_cost_by_user:
                stat_logger.metrics.vtc_cost_by_user[user] = Gauge(
                    name=f"vllm:vtc_cost_by_user_{user}",
                    documentation=f"VTC cost counter of user {user}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.vtc_cost_by_user[user].labels(**stat_logger.labels).set(vtc_cost)
        for user, waiting_seq_groups in self.vtc_waiting_seq_groups_by_user.items():
            if user not in stat_logger.metrics.vtc_waiting_by_user:
                stat_logger.metrics.vtc_waiting_by_user[user] = Gauge(
                    name=f"vllm:vtc_waiting_by_user_{user}",
                    documentation=f"VTC waiting seq groups of user {user}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.vtc_waiting_by_user[user].labels(**stat_logger.labels).set(waiting_seq_groups)
        for user in stat_logger.metrics.vtc_waiting_by_user.keys():
            if user not in self.vtc_waiting_seq_groups_by_user:
                stat_logger.metrics.vtc_waiting_by_user[user].labels(**stat_logger.labels).set(0)
        for user, running_seq_groups in self.vtc_running_seq_groups_by_user.items():
            if user not in stat_logger.metrics.vtc_running_by_user:
                stat_logger.metrics.vtc_running_by_user[user] = Gauge(
                    name=f"vllm:vtc_running_by_user_{user}",
                    documentation=f"VTC running seq groups of user {user}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.vtc_running_by_user[user].labels(**stat_logger.labels).set(running_seq_groups)
        for user in stat_logger.metrics.vtc_running_by_user.keys():
            if user not in self.vtc_running_seq_groups_by_user:
                stat_logger.metrics.vtc_running_by_user[user].labels(**stat_logger.labels).set(0)
        '''
