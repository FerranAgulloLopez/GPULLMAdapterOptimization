import heapq
from collections import deque
from typing import Dict, List, Set, Deque

from vllm.logger import init_logger
from vllm.sequence import (SequenceGroup)

logger = init_logger(__name__)


class RequestOrderPolicy:  # TODO transform into interface

    def __init__(self, max_model_len: int):
        self.max_model_len = max_model_len

    def order_seq_groups(
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:
        raise NotImplementedError()

    def reorder_by_starvation(
            self,
            ordered_seq_groups: Deque[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float],
            starvation_factor: int
    ) -> Deque[SequenceGroup]:
        # check starved users
        max_vtc = max(vtc_cost_by_user.values())
        starved_users: Dict[str, float] = {}
        for user_id, count in vtc_cost_by_user.items():
            if count < max_vtc - starvation_factor * self.max_model_len:
                starved_users[user_id] = count

        # apply starvation
        if len(starved_users) > 0:
            starved_users_seq_groups: Dict[str, deque] = {user_id: deque() for user_id in starved_users.keys()}
            ordered_seq_groups_new: deque[SequenceGroup] = deque()
            for seq_group in ordered_seq_groups:
                user_id = vtc_seq_group_user_relation[seq_group.request_id]
                if user_id in starved_users_seq_groups:
                    starved_users_seq_groups[user_id].appendleft(seq_group)
                else:
                    ordered_seq_groups_new.append(seq_group)
            starved_users_in_order: List[str] = [user_id for _, user_id in sorted(zip(starved_users.values(), starved_users.keys()), key=lambda pair: pair[0])]
            for user_id in reversed(starved_users_in_order):
                ordered_seq_groups_new.extendleft(starved_users_seq_groups[user_id])

            return ordered_seq_groups_new  # TODO refactor, do not transform, already use list structure instead
        else:
            return ordered_seq_groups

    @classmethod
    def select_type(cls, _type: str, max_model_len: int):
        if _type == 'fcfs':
            return FCFSRequestOrderPolicy(max_model_len)
        elif _type == 'fcfs-userprio':
            return FCFSUserPrioRequestOrderPolicy(max_model_len)
        elif _type == 'spt':
            return SPTRequestOrderPolicy(max_model_len)
        elif _type == 'fairness-fcfs':
            return FairnessRequestOrderPolicy(max_model_len, FCFSRequestOrderPolicy(max_model_len))
        elif _type == 'fairness-spt':
            return FairnessRequestOrderPolicy(max_model_len, SPTRequestOrderPolicy(max_model_len))
        elif 'fcfs-starvation_' in _type:
            return FCFSUserStarvationRequestOrderPolicy(_type, max_model_len)
        elif 'spt-starvation_' in _type:
            return SPTUserStarvationRequestOrderPolicy(_type, max_model_len)
        else:
            raise NotImplementedError()


class FCFSRequestOrderPolicy(RequestOrderPolicy):

    def __init__(self, max_model_len: int):
        super().__init__(max_model_len)

    def order_seq_groups(
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:
        return deque([
            seq_group
            for _, seq_group in sorted(
                [(seq_group.metrics.arrival_time, seq_group) for seq_group in seq_groups_to_order],
                key=lambda pair: pair[0]
            )
        ])


class FCFSUserPrioRequestOrderPolicy(RequestOrderPolicy):

    def __init__(self, max_model_len: int):
        super().__init__(max_model_len)
        self.default_policy = FCFSRequestOrderPolicy(max_model_len)

    def order_seq_groups(  # TODO update to use real user priority instead of transforming user_id to an integer
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:
        priority_user_requests: Set[SequenceGroup] = set()
        not_priority_user_requests: Set[SequenceGroup] = set()
        for seq_group in seq_groups_to_order:
            if int(vtc_seq_group_user_relation[seq_group.request_id]) < 3:
                priority_user_requests.add(seq_group)
            else:
                not_priority_user_requests.add(seq_group)

        priority_user_requests: Deque[SequenceGroup] = self.default_policy.order_seq_groups(
            seq_groups_to_order=priority_user_requests,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )
        not_priority_user_requests: Deque[SequenceGroup] = self.default_policy.order_seq_groups(
            seq_groups_to_order=not_priority_user_requests,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )
        return priority_user_requests + not_priority_user_requests


class SPTRequestOrderPolicy(RequestOrderPolicy):

    def __init__(self, max_model_len: int):
        super().__init__(max_model_len)

    def __estimate_output_length(self, seq_group: SequenceGroup) -> int:
        raise NotImplementedError("Output length estimation is not implemented")  # TODO implement

    def order_seq_groups(
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:
        return deque([
            seq_group
            for _, seq_group in sorted(
                [(self.__estimate_output_length(seq_group), seq_group) for seq_group in seq_groups_to_order],
                key=lambda pair: pair[0]
            )
        ])


class FairnessRequestOrderPolicy(RequestOrderPolicy):

    def __init__(self, max_model_len: int, request_order_policy: RequestOrderPolicy):
        super().__init__(max_model_len)
        self.request_order_policy = request_order_policy

    def order_seq_groups(
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:

        # perform default order
        ordered_seq_groups: Deque[SequenceGroup] = self.request_order_policy.order_seq_groups(
            seq_groups_to_order=seq_groups_to_order,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )

        # store seq groups by user
        seq_groups_by_user: Dict[str, Deque[SequenceGroup]] = {}
        for seq_group in ordered_seq_groups:
            user_id = vtc_seq_group_user_relation[seq_group.request_id]
            if user_id not in seq_groups_by_user:
                seq_groups_by_user[user_id] = deque([seq_group])
            else:
                seq_groups_by_user[user_id].append(seq_group)

        # create heap to get user with the lowest fairness faster
        updated_counter = []
        for user_id in seq_groups_by_user.keys():
            heapq.heappush(updated_counter, (vtc_cost_by_user[user_id], user_id))

        # perform fairness order
        ordered_seq_groups: Deque[SequenceGroup] = deque()
        for _ in range(len(seq_groups_to_order)):
            highest_priority_user_counter, highest_priority_user = heapq.heappop(updated_counter)
            highest_priority_seq_group = seq_groups_by_user[highest_priority_user].popleft()
            ordered_seq_groups.append(highest_priority_seq_group)
            if len(seq_groups_by_user[highest_priority_user]) > 0:
                highest_priority_user_counter += sum([seq.get_prompt_len() for seq in highest_priority_seq_group.get_seqs()])
                heapq.heappush(updated_counter, (highest_priority_user_counter, highest_priority_user))

        return ordered_seq_groups


class FCFSUserStarvationRequestOrderPolicy(FCFSRequestOrderPolicy):

    def __init__(self, _type: str, max_model_len: int):
        super().__init__(max_model_len)
        self.starvation_factor: int = int(_type.split('_')[-1])
        if self.starvation_factor < 0:
            raise ValueError(f'k parameter should be an integer value equal or over 0')

    def order_seq_groups(
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:
        ordered_seq_groups: Deque[SequenceGroup] = super().order_seq_groups(
            seq_groups_to_order=seq_groups_to_order,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )
        return super().reorder_by_starvation(
            ordered_seq_groups=ordered_seq_groups,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user,
            starvation_factor=self.starvation_factor
        )


class SPTUserStarvationRequestOrderPolicy(SPTRequestOrderPolicy):

    def __init__(self, _type: str, max_model_len: int):
        super().__init__(max_model_len)
        self.starvation_factor: int = int(_type.split('_')[-1])
        if self.starvation_factor < 0:
            raise ValueError(f'k parameter should be an integer value equal or over 0')

    def order_seq_groups(
            self,
            seq_groups_to_order: Set[SequenceGroup],
            vtc_seq_group_user_relation: Dict[str, str],
            vtc_cost_by_user: Dict[str, float]
    ) -> Deque[SequenceGroup]:
        ordered_seq_groups: Deque[SequenceGroup] = super().order_seq_groups(
            seq_groups_to_order=seq_groups_to_order,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user
        )
        return super().reorder_by_starvation(
            ordered_seq_groups=ordered_seq_groups,
            vtc_seq_group_user_relation=vtc_seq_group_user_relation,
            vtc_cost_by_user=vtc_cost_by_user,
            starvation_factor=self.starvation_factor
        )
