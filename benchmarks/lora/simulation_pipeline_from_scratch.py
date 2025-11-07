import csv
import heapq
import argparse
import os
import re
import json
import glob
import random
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np
from multiprocessing import Pool
from multiprocessing.context import TimeoutError
from benchmarks.lora.benchmark_serving_by_time import assign_rates_to_adapters, create_adapter_prompts
from transformers import AutoTokenizer

from digital_twin_dynamic.manager import DynamicSimulatorManager


def assign_sizes_to_adapters(
        adapters: List[str],
        sizes_to_use: List[int]
) -> Tuple[List[str], List[int]]:
    assert len(adapters) >= len(sizes_to_use)
    adapters_sizes: List[int] = []

    index = 0
    while len(adapters_sizes) < len(adapters):
        adapters_sizes.append(sizes_to_use[index])
        index += 1
        if index >= len(sizes_to_use):
            index = 0
    aux_shuffled_list = list(zip(adapters, adapters_sizes))
    random.shuffle(aux_shuffled_list)
    adapters, adapters_sizes = zip(*aux_shuffled_list)

    values, counts = np.unique(adapters_sizes, return_counts=True)
    print(f"Adapter sizes. Values: {values}. Counts: {counts}")
    return adapters, adapters_sizes


def generate_adapter_requests(
    adapter_requests: List[Tuple[str, int, int]],
    adapter_rate: float,
    finish_time: float
) -> Tuple[str, int, int, float]:
    current_time: float = 0
    request_iterator = iter(adapter_requests)

    # send first request after interval
    interval = np.random.exponential(1.0 / adapter_rate)
    if interval >= finish_time:
        yield None, None, None, None
        return
    current_time += interval

    # send requests
    interval = np.random.exponential(1.0 / adapter_rate)
    for index, (prompt, prompt_len, output_len) in enumerate(request_iterator):
        yield prompt, prompt_len, output_len, current_time
        current_time += interval
        interval = np.random.exponential(1.0 / adapter_rate)

    raise ValueError('Task without enough requests to send')


def simulation_pipeline_from_scratch(
    total_time: float,  # total time to simulate
    model: str,  # model path to simulate
    adapter_slots: int,  # number of adapter slots in GPU (GPU adapters)
    served_adapters: int,  # number adapters to serve (CPU adapters)
    served_adapters_rates: List[float],  # list of adapter rates to distributed between served adapters
    served_adapters_sizes: List[int],  # list of adapter sizes to distribute between served adapters
    available_gpu_memory: int,  # max token capacity after loading model and adapter weights
    max_num_batched_tokens: int,  # max token number to batch together
    dataset_path: str,  # path to dataset to create prompts from
    random_seed: Optional[int] = 0,  # random seed for collecting prompts from datasets
    output_path: Optional[str] = None,  # path to store arrivals and adapter information
    mean_version: Optional[bool] = False,
    include_computation_overhead: Optional[bool] = True,
    include_preemption: Optional[bool] = False,
    without_offloading: Optional[bool] = True,
    include_network_collapse: Optional[int] = None,
    debug_path: Optional[str] = None,
    print_outcome: Optional[bool] = False,
    timeout: Optional[int] = None
) -> Dict[str, float]:
    simulation_tests: int = 1

    # set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # check without offloading case makes sense with adapter information
    if without_offloading:
        assert adapter_slots == served_adapters

    # create served adapters ids
    served_adapters: List[str] = [f'adapter_{index}' for index in range(served_adapters)]

    # equally distribute adapter rates between served adapters (and create arrays)
    served_adapters, served_adapters_rates = assign_rates_to_adapters(served_adapters, served_adapters_rates)

    # create adapter prompts
    tokenizer = AutoTokenizer.from_pretrained(model)
    adapters_prompts: List[List[Tuple[str, int, int]]] = create_adapter_prompts(
        total_time=total_time,
        adapters=served_adapters,
        adapters_rates=served_adapters_rates,
        dataset_name='sharegpt',
        dataset_path=dataset_path,
        sharegpt_output_len=None,
        tokenizer=tokenizer
    )

    # determine request arrivals
    request_arrivals: List[Tuple[float, Tuple[int, int, str]]] = []
    total_inputs: List[int] = []
    total_outputs: List[int] = []
    for index in range(len(served_adapters)):
        for (prompt, prompt_len, output_len, current_time) in generate_adapter_requests(
                adapters_prompts[index],
                served_adapters_rates[index],
                total_time
        ):
            if current_time is None or current_time >= total_time:
                break
            request_arrivals.append(
                (
                    current_time,
                    (
                        prompt_len,
                        output_len,
                        served_adapters[index]
                    )
                )
            )
            total_inputs.append(int(prompt_len))
            total_outputs.append(int(output_len))

    total_inputs: np.ndarray = np.asarray(total_inputs)
    total_outputs: np.ndarray = np.asarray(total_outputs)
    mean_inputs = np.mean(total_inputs)
    std_inputs = np.std(total_inputs)
    mean_outputs = np.mean(total_outputs)
    std_outputs = np.std(total_outputs)
    if mean_version:
        total_inputs: List[int] = []
        total_outputs: List[int] = []
        for index in range(len(request_arrivals)):
            input_tokens: int = max(1, round(np.random.normal(loc=mean_inputs, scale=std_inputs)))
            output_tokens: int = max(2, round(np.random.normal(loc=mean_outputs, scale=std_outputs)))
            request_arrivals[index] = (
                request_arrivals[index][0],
                (
                    input_tokens,
                    output_tokens,
                    request_arrivals[index][1][2]
                )
            )
            total_inputs.append(int(input_tokens))
            total_outputs.append(int(output_tokens))

    # store arrivals
    if output_path is not None:
        total_arrivals: int = len(request_arrivals)
        total_arrivals_input_tokens: int = sum(total_inputs)
        total_arrivals_output_tokens: int = sum(total_outputs)
        arrivals_info: Dict[str, Any] = {
            'total_arrivals': total_arrivals,
            'total_arrivals_input_tokens': total_arrivals_input_tokens,
            'total_arrivals_output_tokens': total_arrivals_output_tokens
        }
        with open(os.path.join(output_path, 'arrivals.json'), 'w') as file:
            json.dump(arrivals_info, file, indent=4)

    # equally distribute adapter sizes between served adapters (and create arrays)
    served_adapters, served_adapters_sizes = assign_sizes_to_adapters(served_adapters, served_adapters_sizes)

    # store adapter information
    if output_path is not None:
        with open(os.path.join(output_path, 'adapters.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['adapter_id', 'adapter_rate', 'adapter_size'])
            for index in range(len(served_adapters)):
                writer.writerow([served_adapters[index], served_adapters_rates[index], served_adapters_sizes[index]])

    # heapify arrivals
    heapq.heapify(request_arrivals)

    # run simulator
    model_id: str = os.path.basename(model)
    simulator: DynamicSimulatorManager = DynamicSimulatorManager(
        start_time=0,
        finish_time=total_time,
        model=model_id,
        adapter_slots=adapter_slots,
        served_adapters=served_adapters,
        served_adapters_sizes=served_adapters_sizes,
        available_gpu_memory=available_gpu_memory,
        max_num_batched_tokens=max_num_batched_tokens,
        request_arrivals=request_arrivals,
        print_outcome=print_outcome,
        include_computation_overhead=include_computation_overhead,
        include_loading_overhead=(not without_offloading),
        include_preemption=include_preemption,
        include_network_collapse=include_network_collapse,
        debug_path=debug_path
    )

    sum_simulation_output: Dict[str, float] = {}
    for _ in range(simulation_tests):
        simulation_output: Dict[str, float] = None
        if timeout is None:
            simulation_output = simulator.simulate()
        else:
            with Pool() as pool:
                result = pool.apply_async(simulator.simulate)
                try:
                    simulation_output = result.get(timeout=timeout)
                except TimeoutError:
                    print('Warn!!. Timeout found')
        if simulation_output:
            for key in simulation_output.keys():
                if key not in sum_simulation_output:
                    sum_simulation_output[key] = simulation_output[key]
                else:
                    sum_simulation_output[key] += simulation_output[key]

    for key in sum_simulation_output.keys():
        sum_simulation_output[key] /= simulation_tests

    return sum_simulation_output if len(sum_simulation_output) > 0 else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--total-time',
        type=float,
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--adapter-slots',
        type=int,
        required=True
    )
    parser.add_argument(
        '--served-adapters',
        type=int,
        required=True
    )
    parser.add_argument(
        '--served-adapters-rates',
        type=str,
        required=True
    )
    parser.add_argument(
        '--served-adapters-sizes',
        type=str,
        required=True
    )
    parser.add_argument(
        '--available-gpu-memory',
        type=int,
        required=True
    )
    parser.add_argument(
        '--max-num-batched_tokens',
        type=int,
        required=True
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        required=False
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        '--use-mean-version',
        default=False,
        action='store_true',
        help='Run simulator with mean version',
    )
    parser.add_argument(
        '--include-computation-overhead',
        default=False,
        action='store_true',
        help='Run simulator including computation overhead',
    )
    parser.add_argument(
        '--include-preemption',
        default=False,
        action='store_true',
        help='Run simulator including preemption',
    )
    parser.add_argument(
        '--without-offloading',
        default=False,
        action='store_true',
        help='Run simulator as the without offloading case',
    )
    parser.add_argument(
        '--include-network-collapse',
        default=None,
        type=int,
        help='Run simulator including network collapse',
    )
    parser.add_argument(
        '--debug-path',
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        '--print-outcome',
        default=True,
        action='store_true'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        required=False
    )

    args = parser.parse_args()

    args.served_adapters_rates = [float(item) for item in args.served_adapters_rates.split(' ')]
    args.served_adapters_sizes = [int(item) for item in args.served_adapters_sizes.split(' ')]

    simulation_output = simulation_pipeline_from_scratch(
        total_time=args.total_time,
        model=args.model,
        adapter_slots=args.adapter_slots,
        served_adapters=args.served_adapters,
        served_adapters_rates=args.served_adapters_rates,
        served_adapters_sizes=args.served_adapters_sizes,
        available_gpu_memory=args.available_gpu_memory,
        max_num_batched_tokens=args.max_num_batched_tokens,
        dataset_path=args.dataset_path,
        random_seed=args.random_seed,
        mean_version=args.use_mean_version,
        include_computation_overhead=args.include_computation_overhead,
        include_preemption=args.include_preemption,
        without_offloading=args.without_offloading,
        include_network_collapse=args.include_network_collapse,
        debug_path=args.debug_path,
        output_path=args.output_path,
        print_outcome=args.print_outcome,
        timeout=args.timeout
    )

    if simulation_output is not None and args.output_path is not None:
        with open(os.path.join(args.output_path, 'simulation_results.json'), 'w') as f:
            json.dump(simulation_output, f, indent=4)
