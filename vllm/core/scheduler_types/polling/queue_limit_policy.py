from typing import Dict, Set, Deque

from vllm.core.scheduler_types.polling.request_order_policy import RequestOrderPolicy
from vllm.sequence import (SequenceGroup)


class QueueLimitPolicy:  # TODO transform into interface

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        self.requests_order_policy = requests_order_policy

    def select_seq_groups_from_queue(
            self,
            seq_groups: Set[SequenceGroup],
            seq_groups_arrival_check: Dict[str, bool],
            already_sent_requests: int,
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Set[SequenceGroup]:
        raise NotImplementedError()

    @classmethod
    def select_type(cls, _type: str, requests_order_policy: RequestOrderPolicy):
        if _type == 'gated':
            return GatedQueueLimitPolicy(requests_order_policy)
        elif _type == 'exhaustive':
            return ExhaustiveQueueLimitPolicy(requests_order_policy)
        elif 'limited-gated_' in _type:
            return LimitedGatedQueueLimitPolicy(_type, requests_order_policy)
        elif 'limited-userprio-gated_' in _type:
            return LimitedUserPrioQueueLimitPolicy(_type, requests_order_policy, LimitedGatedQueueLimitPolicy(_type, requests_order_policy))
        elif 'limited-exhaustive_' in _type:
            return LimitedExhaustiveQueueLimitPolicy(_type, requests_order_policy)
        elif 'limited-userprio-exhaustive_' in _type:
            return LimitedUserPrioQueueLimitPolicy(_type, requests_order_policy, LimitedExhaustiveQueueLimitPolicy(_type, requests_order_policy))
        else:
            raise NotImplementedError()


class GatedQueueLimitPolicy(QueueLimitPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def select_seq_groups_from_queue(
            self,
            seq_groups: Set[SequenceGroup],
            seq_groups_arrival_check: Dict[str, bool],
            already_sent_requests: int,
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Set[SequenceGroup]:
        return {seq_group for seq_group in seq_groups if seq_groups_arrival_check[seq_group.request_id]}


class ExhaustiveQueueLimitPolicy(QueueLimitPolicy):

    def __init__(self, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)

    def select_seq_groups_from_queue(
            self,
            seq_groups: Set[SequenceGroup],
            seq_groups_arrival_check: Dict[str, bool],
            already_sent_requests: int,
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Set[SequenceGroup]:
        return seq_groups


class LimitedGatedQueueLimitPolicy(QueueLimitPolicy):

    def __init__(self, _type: str, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)
        self.k: int = int(_type.split('_')[-1])
        if self.k <= 0:
            raise ValueError(f'k parameter should be an integer value over 0')

    def select_seq_groups_from_queue(
            self,
            seq_groups: Set[SequenceGroup],
            seq_groups_arrival_check: Dict[str, bool],
            already_sent_requests: int,
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Set[SequenceGroup]:
        number_to_send = self.k - already_sent_requests
        selected_seq_groups: Set[SequenceGroup] = set()
        if number_to_send > 0:
            ordered_seq_groups: Deque[SequenceGroup] = self.requests_order_policy.order_seq_groups(
                seq_groups_to_order={seq_group for seq_group in seq_groups if seq_groups_arrival_check[seq_group.request_id]},
                vtc_seq_group_user_relation=vtc_seq_group_user_relation,
                vtc_cost_by_user=vtc_cost_by_user
            )
            while len(selected_seq_groups) < number_to_send and len(ordered_seq_groups) > 0:
                selected_seq_groups.add(ordered_seq_groups.popleft())
        return selected_seq_groups


class LimitedExhaustiveQueueLimitPolicy(QueueLimitPolicy):

    def __init__(self, _type: str, requests_order_policy: RequestOrderPolicy):
        super().__init__(requests_order_policy)
        self.k: int = int(_type.split('_')[-1])
        if self.k <= 0:
            raise ValueError(f'k parameter should be an integer value over 0')

    def select_seq_groups_from_queue(
            self,
            seq_groups: Set[SequenceGroup],
            seq_groups_arrival_check: Dict[str, bool],
            already_sent_requests: int,
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Set[SequenceGroup]:
        number_to_send = self.k - already_sent_requests
        selected_seq_groups: Set[SequenceGroup] = set()
        if number_to_send > 0:
            ordered_seq_groups: Deque[SequenceGroup] = self.requests_order_policy.order_seq_groups(
                seq_groups_to_order=seq_groups,
                vtc_seq_group_user_relation=vtc_seq_group_user_relation,
                vtc_cost_by_user=vtc_cost_by_user
            )
            while len(selected_seq_groups) < number_to_send and len(ordered_seq_groups) > 0:
                selected_seq_groups.add(ordered_seq_groups.popleft())
        return selected_seq_groups


class LimitedUserPrioQueueLimitPolicy(QueueLimitPolicy):

    def __init__(self, _type: str, requests_order_policy: RequestOrderPolicy, default_limit_policy: QueueLimitPolicy):
        super().__init__(requests_order_policy)
        self.default_policy = default_limit_policy

    def select_seq_groups_from_queue(  # TODO update to use real user priority instead of transforming user_id to an integer
            self,
            seq_groups: Set[SequenceGroup],
            seq_groups_arrival_check: Dict[str, bool],
            already_sent_requests: int,
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Set[SequenceGroup]:
        priority_user_seq_groups: Set[SequenceGroup] = set()
        not_priority_seq_groups: Set[SequenceGroup] = set()

        for seq_group in seq_groups:
            if int(vtc_seq_group_user_relation[seq_group.request_id]) < 3:
                priority_user_seq_groups.add(seq_group)
            else:
                not_priority_seq_groups.add(seq_group)
        not_priority_seq_groups: Set[SequenceGroup] = self.default_policy.select_seq_groups_from_queue(
            seq_groups=not_priority_seq_groups,
            seq_groups_arrival_check=seq_groups_arrival_check,
            already_sent_requests=already_sent_requests + len(priority_user_seq_groups),
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )

        return priority_user_seq_groups.union(not_priority_seq_groups)
