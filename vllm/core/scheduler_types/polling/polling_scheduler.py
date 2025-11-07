import json
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from prometheus_client import (Gauge)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.scheduler import PreemptionMode, SchedulingBudget, SchedulerPrefillOutputs, SchedulerRunningOutputs
from vllm.core.scheduler_types.polling.queue_choosing_policy import QueueChoosingPolicy
from vllm.core.scheduler_types.polling.queue_limit_policy import QueueLimitPolicy
from vllm.core.scheduler_types.polling.queue_limit_policy_global import QueueLimitPolicyGlobal
from vllm.core.scheduler_types.polling.request_order_policy import RequestOrderPolicy
from vllm.core.scheduler_types.vtc_counter.vtc_counter import VTCCounter
from vllm.engine.metrics import StatLogger
from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)
from vllm.core.policy import Policy, PolicyFactory

logger = init_logger(__name__)


class PollingScheduler(VTCCounter):

    def __init__(
            self,
            scheduler_config: SchedulerConfig,
            cache_config: CacheConfig,
            lora_config: Optional[LoRAConfig],
            request_order_policy: str,
            queue_choosing_policy: str,
            queue_limit_policy_is_global: str,
            queue_limit_policy: str,
            prefill_weight: Optional[float] = 1,  # TODO get defaults from superclass
            decode_weight: Optional[float] = 2,  # TODO idem
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config, prefill_weight, decode_weight)
        request_order_policy_name: str = request_order_policy
        self.polling_request_order_policy: RequestOrderPolicy = RequestOrderPolicy.select_type(
            request_order_policy_name,
            self.scheduler_config.max_model_len
        )

        queue_choosing_policy_name: str = queue_choosing_policy
        self.polling_queue_choosing_policy: QueueChoosingPolicy = QueueChoosingPolicy.select_type(
            queue_choosing_policy_name,
            self.polling_request_order_policy
        )

        queue_limit_policy_is_global: bool = True if queue_limit_policy_is_global.lower() in ['true', '1', 'y', 'yes'] else False
        queue_limit_policy_name: str = queue_limit_policy
        if queue_limit_policy_is_global:
            self.polling_queue_limit_policy_is_global = True
            self.polling_queue_limit_policy_global: QueueLimitPolicyGlobal = QueueLimitPolicyGlobal.select_type(
                queue_limit_policy_name,
                self.lora_config.max_loras,
                self.polling_request_order_policy,
                self.polling_queue_choosing_policy
            )
        else:
            self.polling_queue_limit_policy_is_global = False
            self.polling_queue_limit_policy: QueueLimitPolicy = QueueLimitPolicy.select_type(
                queue_limit_policy_name,
                self.polling_request_order_policy
            )

        self.polling_running_queues: Dict[int, Set[SequenceGroup]] = {}
        self.polling_waiting_queues: Dict[int, Set[SequenceGroup]] = {}

        self.polling_seq_groups_arrival_check: Dict[str, bool] = {}  # True -> arrived before switch; False -> arrived after switch
        self.polling_queue_number_sent_to_run: Dict[int, int] = {}

        self.polling_initialized_visualization: bool = False
        self.max_cpu_loras = lora_config.max_cpu_loras

        self.policy = PolicyFactory.get_policy(policy_name="lost_work")  # important for ordering running preempted requests
        
    def add_seq_group(self, seq_group: SequenceGroup, user_id: Optional[str] = None) -> None:
        super().add_seq_group(seq_group, user_id)
        self.waiting = deque()  # avoid waiting queue from superclass to expand infinitely

        lora_int_id = seq_group.lora_int_id
        if lora_int_id <= 0:
            raise NotImplementedError("Polling scheduler does not manage non-LORA requests")  # TODO implement
        if user_id is None:
            raise NotImplementedError("Polling scheduler does not manage requests without specified user")

        if lora_int_id in self.polling_running_queues:
            self.polling_running_queues[lora_int_id].add(seq_group)
            self.polling_seq_groups_arrival_check[seq_group.request_id] = False
        elif lora_int_id in self.polling_waiting_queues:
            self.polling_waiting_queues[lora_int_id].add(seq_group)
            self.polling_seq_groups_arrival_check[seq_group.request_id] = True
        else:
            self.polling_waiting_queues[lora_int_id] = {seq_group}
            self.polling_seq_groups_arrival_check[seq_group.request_id] = True
            self.polling_queue_number_sent_to_run[lora_int_id] = 0

    def has_unfinished_seqs(self) -> bool:
        return (
                super().has_unfinished_seqs()
                or len(self.polling_running_queues) != 0
                or len(self.polling_waiting_queues)
        )

    def get_num_unfinished_seq_groups(self) -> int:
        return (
                sum([len(queue_seq_groups) for queue_seq_groups in self.polling_running_queues.values()])
                + sum([len(queue_seq_groups) for queue_seq_groups in self.polling_waiting_queues.values()])
                + len(self.running)
                + len(self.swapped)
        )

    def _schedule_prefills(
            self,
            waiting_queue: deque,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        running_by_lora: Dict[int, int] = {}
        for seq_group in self.running:
            lora_int_id = seq_group.lora_int_id
            if lora_int_id in running_by_lora:
                running_by_lora[lora_int_id] += 1
            else:
                running_by_lora[lora_int_id] = 1

        # update queues state - possible changes from running to waiting or deletion
        for queue in list(self.polling_running_queues.keys()):  # transformed to list to avoid changing dict during iteration error
            seq_groups: Set[SequenceGroup] = self.polling_running_queues[queue]
            if queue not in curr_loras:
                if len(seq_groups) > 0:
                    self.polling_waiting_queues[queue] = seq_groups
                    del self.polling_running_queues[queue]
                    for seq_group in seq_groups:
                        self.polling_seq_groups_arrival_check[seq_group.request_id] = True
                    self.polling_queue_number_sent_to_run[queue] = 0
                else:
                    del self.polling_running_queues[queue]
                    del self.polling_queue_number_sent_to_run[queue]

        # update queues state - possible changes from waiting to running (free spot)
        if len(self.polling_running_queues) < self.lora_config.max_loras and len(self.polling_waiting_queues) > 0:
            waiting_queues_ordered_by_priority: Deque[int] = self.polling_queue_choosing_policy.order_by_policy(
                queues_to_order=self.polling_waiting_queues,
                waiting_queues=self.polling_waiting_queues,
                running_queues=self.polling_running_queues,
                vtc_seq_group_user_relation=self.vtc_seq_group_user_relation,
                vtc_cost_by_user=self.vtc_cost_by_user
            )
            while len(self.polling_running_queues) < self.lora_config.max_loras and len(waiting_queues_ordered_by_priority) > 0:
                highest_priority_queue: int = waiting_queues_ordered_by_priority.popleft()
                self.polling_running_queues[highest_priority_queue] = self.polling_waiting_queues[highest_priority_queue]
                del self.polling_waiting_queues[highest_priority_queue]

        # collect seq groups from running queues - depending on policy
        seq_groups_to_run: Set[SequenceGroup] = set()
        if self.polling_queue_limit_policy_is_global:
            seq_groups_to_run = self.polling_queue_limit_policy_global.select_seq_groups_to_run(
                running_queues=self.polling_running_queues,
                waiting_queues=self.polling_waiting_queues,
                seq_groups_arrival_check=self.polling_seq_groups_arrival_check,
                queue_number_sent_to_run=self.polling_queue_number_sent_to_run,
                vtc_seq_group_user_relation=self.vtc_seq_group_user_relation,
                vtc_cost_by_user=self.vtc_cost_by_user,
                running_by_lora=running_by_lora
            )
        else:
            for queue, seq_groups in self.polling_running_queues.items():
                if len(self.polling_waiting_queues) > 0:
                    seq_groups_to_add: Set[SequenceGroup] = self.polling_queue_limit_policy.select_seq_groups_from_queue(
                        seq_groups=seq_groups,
                        seq_groups_arrival_check=self.polling_seq_groups_arrival_check,
                        already_sent_requests=self.polling_queue_number_sent_to_run[queue],
                        vtc_seq_group_user_relation=self.vtc_seq_group_user_relation,
                        vtc_cost_by_user=self.vtc_cost_by_user
                    )
                    seq_groups_to_run.update(seq_groups_to_add)
                else:
                    seq_groups_to_run.update(seq_groups)

        # order seq groups
        seq_groups_to_run: List[SequenceGroup] = list(seq_groups_to_run)
        random.shuffle(seq_groups_to_run)  # avoid possible biases
        seq_groups_to_run: Deque[SequenceGroup] = self.polling_request_order_policy.order_seq_groups(
            seq_groups_to_order=set(seq_groups_to_run),
            vtc_seq_group_user_relation=self.vtc_seq_group_user_relation,
            vtc_cost_by_user=self.vtc_cost_by_user
        )

        # send to system
        remaining_waiting, prefills = super()._schedule_prefills(
            seq_groups_to_run,
            budget,
            curr_loras,
            enable_chunking
        )

        # update parameter values
        for seq_group in prefills.seq_groups:
            seq_group = seq_group.seq_group  # Coming from vLLM bug, received object is ScheduledSequenceGroup instead
            request_id = seq_group.request_id
            lora_int_id = seq_group.lora_int_id

            del self.polling_seq_groups_arrival_check[request_id]
            self.polling_queue_number_sent_to_run[lora_int_id] += 1
            self.polling_running_queues[lora_int_id].remove(seq_group)

        for seq_group in prefills.ignored_seq_groups:
            request_id = seq_group.request_id
            lora_int_id = seq_group.lora_int_id

            del self.polling_seq_groups_arrival_check[request_id]
            self.polling_queue_number_sent_to_run[lora_int_id] += 1
            self.polling_running_queues[lora_int_id].remove(seq_group)

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
        self.waiting = deque()  # avoid waiting queue from superclass to expand infinitely
        for seq_group in running_scheduled.preempted:
            lora_int_id = seq_group.lora_int_id
            self.polling_running_queues[lora_int_id].add(seq_group)
            self.polling_seq_groups_arrival_check[seq_group.request_id] = True
            self.polling_queue_number_sent_to_run[lora_int_id] -= 1

        return remaining_running, running_scheduled

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        raise NotImplementedError("Polling scheduler does not implement preemption by swapping")  # TODO implement

    def do_alternative_logging(self, stat_logger: StatLogger) -> None:
        super().do_alternative_logging(stat_logger)
        if not self.polling_initialized_visualization:
            stat_logger.metrics.polling_running_by_adapter = {}
            stat_logger.metrics.polling_waiting_by_adapter = {}
            self.polling_initialized_visualization = True

        polling_running_by_adapter: Dict[int, int] = {}
        for index in range(self.max_cpu_loras + 1):
            polling_running_by_adapter[index] = 0

        for seq_group in self.running:
            lora_int_id = seq_group.lora_int_id
            polling_running_by_adapter[lora_int_id] += 1

        polling_waiting_by_adapter: Dict[int, int] = {}
        for index in range(self.max_cpu_loras + 1):
            if index in self.polling_running_queues:
                polling_waiting_by_adapter[index] = len(self.polling_running_queues[index])
            elif index in self.polling_waiting_queues:
                polling_waiting_by_adapter[index] = len(self.polling_waiting_queues[index])
            else:
                polling_waiting_by_adapter[index] = 0

        for lora_int_id, running in polling_running_by_adapter.items():
            if lora_int_id not in stat_logger.metrics.polling_running_by_adapter:
                stat_logger.metrics.polling_running_by_adapter[lora_int_id] = Gauge(
                    name=f"vllm:running_by_adapter_{lora_int_id}",
                    documentation=f"Number of running requests of adapter {lora_int_id}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.polling_running_by_adapter[lora_int_id].labels(**stat_logger.labels).set(running)

        for lora_int_id, waiting in polling_waiting_by_adapter.items():
            if lora_int_id not in stat_logger.metrics.polling_waiting_by_adapter:
                stat_logger.metrics.polling_waiting_by_adapter[lora_int_id] = Gauge(
                    name=f"vllm:waiting_by_adapter_{lora_int_id}",
                    documentation=f"Number of waiting requests of adapter {lora_int_id}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.polling_waiting_by_adapter[lora_int_id].labels(**stat_logger.labels).set(waiting)

        '''
        if not self.polling_initialized_visualization:
            stat_logger.metrics.polling_waiting_by_queue = {}
            stat_logger.metrics.polling_running_by_queue = {}
            self.polling_initialized_visualization = True
        for queue, seq_groups in self.polling_waiting_queues.items():
            if queue not in stat_logger.metrics.polling_waiting_by_queue:
                stat_logger.metrics.polling_waiting_by_queue[queue] = Gauge(
                    name=f"vllm:polling_waiting_by_queue_{queue}",
                    documentation=f"Polling waiting seq groups of queue {queue}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.polling_waiting_by_queue[queue].labels(**stat_logger.labels).set(len(seq_groups))
        for queue in stat_logger.metrics.polling_waiting_by_queue.keys():
            if queue not in self.polling_waiting_queues:
                stat_logger.metrics.polling_waiting_by_queue[queue].labels(**stat_logger.labels).set(0)
        for queue, seq_groups in self.polling_running_queues.items():
            if queue not in stat_logger.metrics.polling_running_by_queue:
                stat_logger.metrics.polling_running_by_queue[queue] = Gauge(
                    name=f"vllm:polling_running_by_queue_{queue}",
                    documentation=f"Polling running seq groups of queue {queue}.",
                    labelnames=list(stat_logger.labels.keys())
                )
            stat_logger.metrics.polling_running_by_queue[queue].labels(**stat_logger.labels).set(len(seq_groups))
        for queue in stat_logger.metrics.polling_running_by_queue.keys():
            if queue not in self.polling_running_queues:
                stat_logger.metrics.polling_running_by_queue[queue].labels(**stat_logger.labels).set(0)
        '''
