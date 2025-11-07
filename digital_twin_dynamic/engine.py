from typing import List, Tuple

from digital_twin_dynamic.estimators.predictor_default_latency.predictor_default_latency import \
    PREFILL_CONSTANTS_PER_MODEL as DEFAULT_PREFILL_CONSTANTS_PER_MODEL,\
    DECODE_CONSTANTS_PER_MODEL as DEFAULT_DECODE_CONSTANTS_PER_MODEL,\
    default_latency_predictor_prefill,\
    default_latency_predictor_decode
'''from digital_twin_dynamic.estimators.predictor_default_latency.predictor_default_latency_grouped_per_step import \
    CONSTANTS_PER_MODEL as DEFAULT_CONSTANTS_PER_MODEL,\
    default_latency_predictor_grouped'''
'''from digital_twin_dynamic.estimators.predictor_latency_overhead.predictor_latency_overhead import \
    PREFILL_CONSTANTS_PER_MODEL as OVERHEAD_PREFILL_CONSTANTS_PER_MODEL,\
    DECODE_CONSTANTS_PER_MODEL as OVERHEAD_DECODE_CONSTANTS_PER_MODEL,\
    latency_overhead_predictor_prefill, \
    latency_overhead_predictor_decode'''
from digital_twin_dynamic.estimators.predictor_latency_overhead.predictor_latency_overhead_small import \
    DECODE_CONSTANTS_PER_MODEL as OVERHEAD_DECODE_CONSTANTS_PER_MODEL,\
    latency_overhead_predictor_decode
from digital_twin_dynamic.structures import Request, RunningBatch


class Engine:
    def __init__(
            self,
            adapter_slots: int,
            model: str,
            include_computation_overhead: bool
    ):
        self.adapter_slots = adapter_slots
        self.include_computation_overhead = include_computation_overhead

        self.default_latency_prefill_constants = DEFAULT_PREFILL_CONSTANTS_PER_MODEL[model]
        self.default_latency_decode_constants = DEFAULT_DECODE_CONSTANTS_PER_MODEL[model]
        '''self.default_latency_constants = DEFAULT_CONSTANTS_PER_MODEL[model]'''
        '''self.overhead_latency_prefill_constants = OVERHEAD_PREFILL_CONSTANTS_PER_MODEL[model]'''
        self.overhead_latency_decode_constants = OVERHEAD_DECODE_CONSTANTS_PER_MODEL[model]

    # Run a computation step, return computation elapsed time
    def run_step(self, running_batch: RunningBatch) -> float:
        '''# estimate default latency in prefills
        latency_prefills: float = default_latency_predictor_prefill(
            [
                running_batch.compute_prefill_tokens
            ],
            **self.default_latency_prefill_constants
        ) / 1000  # it is in milliseconds originally (we want it in seconds)'''
        latency_prefills = 0.0

        # estimate default latency in decodes
        latency_decodes: float = default_latency_predictor_decode(
            [
                running_batch.get_compute_decode_tokens()
            ],
            **self.default_latency_decode_constants
        ) / 1000  # it is in milliseconds originally (we want it in seconds)

        '''# estimate default latency in decodes
        latency: float = default_latency_predictor_grouped(
            [
                running_batch.get_batched_tokens()
            ],
            **self.default_latency_constants
        ) / 1000  # it is in milliseconds originally (we want it in seconds)'''

        # apply adapter slowdown
        if self.include_computation_overhead:
            '''# estimate latency overhead in prefills
            latency_prefills: float = latency_prefills * latency_overhead_predictor_prefill(
                [
                    running_batch.get_len_running_adapters(),
                    running_batch.get_maximum_adapter_size()
                ],
                **self.overhead_latency_prefill_constants
            ) / 100'''

            '''# estimate latency overhead in decodes
            latency: float = latency * latency_overhead_predictor_decode(
                [
                    running_batch.get_len_running_adapters(),
                    running_batch.get_maximum_adapter_size()
                ],
                **self.overhead_latency_decode_constants
            ) / 100'''

            # estimate latency overhead in decodes
            latency_decodes: float = latency_decodes * latency_overhead_predictor_decode(
                [
                    running_batch.get_len_running_adapters(),
                    running_batch.get_maximum_adapter_size()
                ],
                **self.overhead_latency_decode_constants
            ) / 100

        # mix latencies
        estimated_latency: float = max(latency_prefills, latency_decodes)
        #estimated_latency = latency
        assert estimated_latency > 0

        return estimated_latency
