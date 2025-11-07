import time
from typing import Optional, Dict, Tuple, Set, List, Deque
from collections import deque

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.scheduler import SchedulingBudget, SchedulerPrefillOutputs, AllocStatus, ScheduledSequenceGroup
from vllm.core.scheduler_types.vtc_counter.vtc_counter import VTCCounter
from vllm.logger import init_logger

from vllm.sequence import (SequenceGroup, SequenceStatus)

logger = init_logger(__name__)


class FCFSScheduler(VTCCounter):

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)

    def _schedule_prefills(
            self,
            waiting_queue: deque,
            budget: SchedulingBudget,
            curr_loras: Optional[Set[int]],
            enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we break the loop (FCFS approach)
                    break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        prefills = SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

        for seq_group in prefills.seq_groups:
            seq_group = seq_group.seq_group  # Coming from vLLM bug, received object is ScheduledSequenceGroup instead
            if seq_group.request_id in self.vtc_seq_group_user_relation:
                user_id: str = self.vtc_seq_group_user_relation[seq_group.request_id]

                self.vtc_waiting_seq_groups_by_user[user_id] -= 1
                if self.vtc_waiting_seq_groups_by_user[user_id] == 0:
                    del self.vtc_waiting_seq_groups_by_user[user_id]

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
                del self.vtc_seq_group_user_relation[
                    seq_group.request_id]  # TODO check if not duplicate of abort_seq_group method

                self.vtc_waiting_seq_groups_by_user[user_id] -= 1
                if self.vtc_waiting_seq_groups_by_user[user_id] == 0:
                    del self.vtc_waiting_seq_groups_by_user[user_id]

        return waiting_queue, prefills
