from collections import deque
from typing import Dict, Set, Deque, Iterable

from vllm.core.scheduler_types.polling.request_order_policy import RequestOrderPolicy
from vllm.core.scheduler_types.polling.queue_choosing_policy import QueueChoosingPolicy
from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)

logger = init_logger(__name__)


class QueueLimitPolicyGlobal:  # TODO transform into interface

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        self.requests_order_policy = requests_order_policy

    def select_seq_groups_to_run(
            self,
            running_queues: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            seq_groups_arrival_check: Dict[str, bool],
            queue_number_sent_to_run: Dict[int, int],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float],
            running_by_lora: Dict[int, int]
    ) -> Set[SequenceGroup]:
        raise NotImplementedError()

    @classmethod
    def select_type(cls, _type: str, max_loras: int, requests_order_policy: RequestOrderPolicy, queue_choosing_policy: QueueChoosingPolicy):
        if 'limited-exhaustive-capacity-aware_' in _type:
            return LimitedExhaustiveCapacityAwareQueueLimitPolicyGlobal(_type, requests_order_policy, queue_choosing_policy)
        elif 'limited-exhaustive-bounded_' in _type:
            return LimitedExhaustiveBoundedQueueLimitPolicyGlobal(_type, max_loras, requests_order_policy, queue_choosing_policy)
        else:
            raise NotImplementedError()


class LimitedExhaustiveCapacityAwareQueueLimitPolicyGlobal(QueueLimitPolicyGlobal):  # TODO transform into interface

    def __init__(self, _type: str, requests_order_policy: RequestOrderPolicy, queue_choosing_policy: QueueChoosingPolicy):
        super().__init__(requests_order_policy)
        self.k: int = int(_type.split('_')[-2])
        self.n: int = int(_type.split('_')[-1])
        if self.k <= 0:
            raise ValueError(f'k parameter should be an integer value over 0')
        if self.n < 0:
            raise ValueError(f'n parameter should be an integer value over or equal to 0')
        self.preempted_queues: Set[int] = set()

    def __eviction_order_policy(
            self,
            queues: Iterable[int],
            queue_number_sent_to_run: Dict[int, int],
    ) -> Deque[int]:
        return deque([
            queue
            for _, queue in sorted(
                zip(
                    [queue_number_sent_to_run[queue] for queue in queues],
                    queues
                ), key=lambda pair: pair[0]
            )
        ])

    def select_seq_groups_to_run(
            self,
            running_queues: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            seq_groups_arrival_check: Dict[str, bool],
            queue_number_sent_to_run: Dict[int, int],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float],
            running_by_lora: Dict[int, int]
    ) -> Set[SequenceGroup]:
        for queue in list(self.preempted_queues):
            if queue not in running_queues or queue_number_sent_to_run[queue] == 0:
                self.preempted_queues.remove(queue)

        number_to_preempt: int = min(self.n, len(waiting_queues))
        if number_to_preempt < len(self.preempted_queues):
            ordered_preempted_queues: Deque[int] = self.__eviction_order_policy(
                queues=self.preempted_queues,
                queue_number_sent_to_run=queue_number_sent_to_run
            )
            while number_to_preempt < len(self.preempted_queues) and len(ordered_preempted_queues) > 0:
                queue: int = ordered_preempted_queues.popleft()
                self.preempted_queues.remove(queue)
        elif number_to_preempt > len(self.preempted_queues):
            ordered_running_queues: Deque[int] = self.__eviction_order_policy(
                queues=running_queues.keys(),
                queue_number_sent_to_run=queue_number_sent_to_run
            )
            while number_to_preempt > len(self.preempted_queues) and len(ordered_running_queues) > 0:
                running_queue: int = ordered_running_queues.pop()
                if queue_number_sent_to_run[running_queue] > self.k:
                    self.preempted_queues.add(running_queue)

        running_seq_groups_to_run: Set[SequenceGroup] = set()
        for queue, queue_seq_groups in running_queues.items():
            if queue not in self.preempted_queues:
                running_seq_groups_to_run.update(queue_seq_groups)
        return running_seq_groups_to_run


class LimitedExhaustiveBoundedQueueLimitPolicyGlobal(QueueLimitPolicyGlobal):  # TODO transform into interface

    def __init__(self, _type: str, max_loras: int, requests_order_policy: RequestOrderPolicy, queue_choosing_policy: QueueChoosingPolicy):
        super().__init__(requests_order_policy)
        self.k: int = int(_type.split('_')[-2])
        self.n: int = int(_type.split('_')[-1])
        if self.k <= 0:
            raise ValueError(f'k parameter should be an integer value over 0')
        if self.n == 0:
            raise ValueError(f'n parameter should be an integer value not equal to 0')
        if self.n < 0:
            self.n = round(max_loras / abs(self.n))
        self.preempted_queues: Set[int] = set()
        self.requests_order_policy: RequestOrderPolicy = requests_order_policy
        self.eviction_order_policy: QueueChoosingPolicy = queue_choosing_policy

    def select_seq_groups_to_run(
            self,
            running_queues: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            seq_groups_arrival_check: Dict[str, bool],
            queue_number_sent_to_run: Dict[int, int],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float],
            running_by_lora: Dict[int, int]
    ) -> Set[SequenceGroup]:
        # print('-------------------------------------')
        # print('running_queues', running_queues.keys())
        # print('waiting_queues', waiting_queues.keys())
        # print('1. self.preempted_queues', self.preempted_queues)
        # print('queue_number_sent_to_run', queue_number_sent_to_run)
        # print('running_by_lora', running_by_lora)
        # print('---------------------------------------')
        # print(f'Entering. {len(running_queues)} running queues with {[(len(seq_groups), running_by_lora[queue] if queue in running_by_lora else None) for queue, seq_groups in running_queues.items()]} requests')
        # print(f'Entering. {len(waiting_queues)} waiting queues with {sum([len(seq_groups) for seq_groups in waiting_queues.values()])} requests')
        for queue in list(self.preempted_queues):
            if queue not in running_queues or queue_number_sent_to_run[queue] == 0:
                self.preempted_queues.remove(queue)
        # print('2. self.preempted_queues', self.preempted_queues)

        number_to_preempt: int = min(self.n, len(waiting_queues))
        # print('number_to_preempt', number_to_preempt)
        if number_to_preempt < len(self.preempted_queues):
            ordered_preempted_queues: Deque[int] = self.eviction_order_policy.order_by_policy(
                {queue: running_queues[queue] for queue in self.preempted_queues},
                waiting_queues,
                running_queues,
                vtc_seq_group_user_relation,
                vtc_cost_by_user
            )
            while number_to_preempt < len(self.preempted_queues) and len(ordered_preempted_queues) > 0:
                queue: int = ordered_preempted_queues.popleft()
                self.preempted_queues.remove(queue)
        elif number_to_preempt > len(self.preempted_queues):
            ordered_running_queues: Deque[int] = self.eviction_order_policy.order_by_policy(
                running_queues,
                waiting_queues,
                running_queues,
                vtc_seq_group_user_relation,
                vtc_cost_by_user
            )
            while number_to_preempt > len(self.preempted_queues) and len(ordered_running_queues) > 0:
                running_queue: int = ordered_running_queues.pop()
                if queue_number_sent_to_run[running_queue] != 0:
                    self.preempted_queues.add(running_queue)
        # print('3. self.preempted_queues', self.preempted_queues)

        # print('Loop')
        running_seq_groups_to_run: Set[SequenceGroup] = set()
        for queue, queue_seq_groups in running_queues.items():
            if queue not in self.preempted_queues:
                remaining_to_collect: int = self.k - (running_by_lora[queue] if queue in running_by_lora else 0)
                if remaining_to_collect > 0:
                    ordered_queue_seq_groups: Deque[SequenceGroup] = self.requests_order_policy.order_seq_groups(
                        queue_seq_groups,
                        vtc_seq_group_user_relation,
                        vtc_cost_by_user
                    )
                    ordered_queue_seq_groups: Set[SequenceGroup] = set([seq_group for seq_group in ordered_queue_seq_groups][:remaining_to_collect])
                    running_seq_groups_to_run.update(ordered_queue_seq_groups)
                    # print('queue', queue, 'ordered_queue_seq_groups', len(ordered_queue_seq_groups))

        # print(f'Exiting {len(running_seq_groups_to_run)} seq groups to run')
        return running_seq_groups_to_run
