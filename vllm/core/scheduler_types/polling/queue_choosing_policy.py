import random
from collections import deque
from statistics import mean
from typing import Dict, List, Set, Deque

from vllm.core.scheduler_types.polling.request_order_policy import RequestOrderPolicy
from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)

logger = init_logger(__name__)


class QueueChoosingPolicy:  # TODO transform into interface

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        self.requests_order_policy = requests_order_policy

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        raise NotImplementedError()

    @classmethod
    def select_type(cls, _type: str, requests_order_policy: RequestOrderPolicy):
        if _type == 'random':
            return RandomQueueChoosingPolicy(requests_order_policy)
        if _type == 'lru':
            return LRUQueueChoosingPolicy(requests_order_policy)
        elif _type == 'fcfs-min':
            return FCFSMinQueueChoosingPolicy(requests_order_policy)
        elif _type == 'fcfs-min-userprio':
            return FCFSMinUserPrioQueueChoosingPolicy(requests_order_policy)
        elif _type == 'fcfs-average-min':
            return FCFSAverageMinQueueChoosingPolicy(requests_order_policy)
        elif _type == 'spt-min':
            return SPTMinQueueChoosingPolicy(requests_order_policy)
        elif _type == 'spt-sum-min':
            return SPTSumMinQueueChoosingPolicy(requests_order_policy)
        elif _type == 'spt-max-min':
            return SPTMaxMinQueueChoosingPolicy(requests_order_policy)
        elif 'spt-max-min_' in _type:
            return SPTMaxMinKQueueChoosingPolicy(_type, requests_order_policy)
        elif _type == 'fairness-min':
            return FairnessMinQueueChoosingPolicy(requests_order_policy)
        elif _type == 'fairness-average-min':
            return FairnessAverageMinQueueChoosingPolicy(requests_order_policy)
        else:
            raise NotImplementedError()


class RandomQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_indexes: List[int] = list(queues_to_order.keys())
        random.shuffle(queues_indexes)
        return deque(queues_indexes)


class LRUQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)
        self.running_queues_arrival: Deque[int] = deque()  # left -> the newest arrivals to running state
        self.waiting_queues_arrival: Deque[int] = deque()  # left -> the oldest arrivals to waiting state

    def order_by_policy(  # TODO optimize method
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_being_used: List[int] = list(running_queues.keys())
        random.shuffle(queues_being_used)
        for queue in queues_being_used:
            if queue not in self.running_queues_arrival:
                self.running_queues_arrival.appendleft(queue)
        new_running_queues_arrival = deque()
        for queue in self.running_queues_arrival:
            if queue in running_queues:
                new_running_queues_arrival.append(queue)
        self.running_queues_arrival = new_running_queues_arrival

        queues_not_being_used: List[int] = list(waiting_queues.keys())
        random.shuffle(queues_not_being_used)
        for queue in queues_not_being_used:
            if queue not in self.waiting_queues_arrival:
                self.waiting_queues_arrival.append(queue)
        new_waiting_queues_arrival = deque()
        for queue in self.waiting_queues_arrival:
            if queue in waiting_queues:
                new_waiting_queues_arrival.append(queue)
        self.waiting_queues_arrival = new_waiting_queues_arrival

        ordered_queues: Deque[int] = deque()
        for queue in self.waiting_queues_arrival:
            if queue in queues_to_order:
                ordered_queues.append(queue)
        for queue in self.running_queues_arrival:
            if queue in queues_to_order:
                ordered_queues.append(queue)

        return ordered_queues


class FCFSMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_min_arrival_times: Dict[int, int] = {}
        for queue, requests in queues_to_order.items():
            min_arrival_time = min([seq_group.metrics.arrival_time for seq_group in requests])
            queues_min_arrival_times[queue] = min_arrival_time
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_min_arrival_times.values(),
                        queues_min_arrival_times.keys()
                    ), key=lambda pair: pair[0]
                )
            ])


class FCFSMinUserPrioQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)
        self.default_policy = FCFSMinQueueChoosingPolicy(requests_order_policy)

    def order_by_policy(  # TODO update to use real user priority instead of transforming user_id to an integer
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        user_priority_queues: Dict[int, Set[SequenceGroup]] = {}
        not_user_priority_queues: Dict[int, Set[SequenceGroup]] = {}

        for queue, queue_seq_groups in queues_to_order.items():
            for queue_seq_group in queue_seq_groups:
                if int(vtc_seq_group_user_relation[queue_seq_group.request_id]) < 3:
                    user_priority_queues[queue] = queue_seq_groups
                    break
            else:
                not_user_priority_queues[queue] = queue_seq_groups

        user_priority_queues: Deque[int] = self.default_policy.order_by_policy(
            queues_to_order=user_priority_queues,
            waiting_queues=waiting_queues,
            running_queues=running_queues,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )
        not_user_priority_queues: Deque[int] = self.default_policy.order_by_policy(
            queues_to_order=not_user_priority_queues,
            waiting_queues=waiting_queues,
            running_queues=running_queues,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )

        return user_priority_queues + not_user_priority_queues


class FCFSAverageMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_average_arrival_times: Dict[int, int] = {}
        for queue, requests in queues_to_order.items():
            average_arrival_time = mean([seq_group.metrics.arrival_time for seq_group in requests])
            queues_average_arrival_times[queue] = average_arrival_time
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_average_arrival_times.values(),
                        queues_average_arrival_times.keys()
                    ), key=lambda pair: pair[0]
                )
            ])


class SPTMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def __estimate_output_length(self, seq_group: SequenceGroup) -> int:
        raise NotImplementedError("Output length estimation is not implemented")  # TODO implement

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_min_processing_times: Dict[int, int] = {}
        for queue, requests in queues_to_order.items():
            min_expected_output_len = min([self.__estimate_output_length(seq_group) for seq_group in requests])
            queues_min_processing_times[queue] = min_expected_output_len
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_min_processing_times.values(),
                        queues_min_processing_times.keys()
                    ), key=lambda pair: pair[0]
                )
            ])


class SPTSumMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def __estimate_output_length(self, seq_group: SequenceGroup) -> int:
        raise NotImplementedError("Output length estimation is not implemented")  # TODO implement

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_sum_processing_times: Dict[int, int] = {}
        for queue, requests in queues_to_order.items():
            sum_output_len = sum([self.__estimate_output_length(seq_group) for seq_group in requests])
            queues_sum_processing_times[queue] = sum_output_len
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_sum_processing_times.values(),
                        queues_sum_processing_times.keys()
                    ), key=lambda pair: pair[0]
                )
            ])


class SPTMaxMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)


    def __estimate_output_length(self, seq_group: SequenceGroup) -> int:
        raise NotImplementedError("Output length estimation is not implemented")  # TODO implement

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_max_processing_times: Dict[int, int] = {}
        for queue, requests in queues_to_order.items():
            max_output_len = max([self.__estimate_output_length(seq_group) for seq_group in requests])
            queues_max_processing_times[queue] = max_output_len
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_max_processing_times.values(),
                        queues_max_processing_times.keys()
                    ), key=lambda pair: pair[0]
                )
            ])


class SPTMaxMinKQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, _type: str, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)
        self.k: int = int(_type.split('_')[-1])
        if self.k <= 0:
            raise ValueError(f'k parameter should be an integer value over 0')

    def __estimate_output_length(self, seq_group: SequenceGroup) -> int:
        raise NotImplementedError("Output length estimation is not implemented")  # TODO implement

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_max_processing_times: Dict[int, int] = {}
        for queue, queue_seq_groups in queues_to_order.items():
            queue_seq_groups = self.requests_order_policy.order_seq_groups(
                seq_groups_to_order=queue_seq_groups,
                vtc_seq_group_user_relation=vtc_seq_group_user_relation,
                vtc_cost_by_user=vtc_cost_by_user
            )
            output_lens = []
            index = 0
            while queue_seq_groups and index < self.k:
                seq_group = queue_seq_groups.popleft()
                output_lens.append(self.__estimate_output_length(seq_group))
                index += 1
            max_output_len = max(output_lens)
            queues_max_processing_times[queue] = max_output_len
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_max_processing_times.values(),
                        queues_max_processing_times.keys()
                    ), key=lambda pair: pair[0]
                )
            ])


class FairnessMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_min_fairness: Dict[int, float] = {}
        queues_without_seq_groups: List[int] = []
        for queue, queue_seq_groups in queues_to_order.items():
            if len(queue_seq_groups) > 0:
                min_fairness = min([vtc_cost_by_user[vtc_seq_group_user_relation[seq_group.request_id]] for seq_group in queue_seq_groups])
                queues_min_fairness[queue] = min_fairness
            else:
                queues_without_seq_groups.append(queue)
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_min_fairness.values(),
                        queues_min_fairness.keys()
                    ), key=lambda pair: pair[0]
                )
            ] + queues_without_seq_groups)


class FairnessAverageMinQueueChoosingPolicy(QueueChoosingPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def order_by_policy(
            self,
            queues_to_order: Dict[int, Set[SequenceGroup]],
            waiting_queues: Dict[int, Set[SequenceGroup]],
            running_queues: Dict[int, Set[SequenceGroup]],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[int]:
        queues_average_fairness: Dict[int, float] = {}
        for queue, requests in queues_to_order.items():
            average_fairness = mean([vtc_cost_by_user[vtc_seq_group_user_relation[seq_group.request_id]] for seq_group in requests])
            queues_average_fairness[queue] = average_fairness
        return deque([
                queue
                for _, queue in sorted(
                    zip(
                        queues_average_fairness.values(),
                        queues_average_fairness.keys()
                    ), key=lambda pair: pair[0]
                )
            ])
