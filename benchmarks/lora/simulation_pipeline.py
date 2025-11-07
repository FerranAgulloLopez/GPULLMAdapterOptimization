import csv
import heapq
import argparse
import os
import re
import json
import glob
import psutil
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from multiprocessing import Pool
from multiprocessing.context import TimeoutError
import threading
import time

from digital_twin_dynamic.manager import DynamicSimulatorManager


def monitor_avg_usage(func, *args, sample_interval=0.5, **kwargs):
    process = psutil.Process(os.getpid())
    cpu_samples = []
    mem_samples = []
    running = True

    def sampler():
        while running:
            # CPU percent is relative to all CPUs, use interval=0 for instantaneous
            cpu_samples.append(process.cpu_percent(interval=None))
            mem_samples.append(process.memory_info().rss)
            time.sleep(sample_interval)

    # Start sampling thread
    t = threading.Thread(target=sampler)
    t.start()

    # Time and run the function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    # Stop sampling
    running = False
    t.join()

    # Compute averages
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    avg_mem = sum(mem_samples) / len(mem_samples) if mem_samples else 0

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Average CPU usage: {avg_cpu:.2f}%")
    print(f"Average memory usage: {avg_mem / 1024**2:.2f} MB")

    return result


def simulation_pipeline(
    experiment_path: str,
    mean_version: Optional[bool] = False,
    include_computation_overhead: Optional[bool] = True,
    include_preemption: Optional[bool] = False,
    without_offloading: Optional[bool] = True,
    include_network_collapse: Optional[bool] = True,
    debug_path: Optional[str] = None,
    print_outcome: Optional[bool] = False,
    timeout: Optional[int] = None
) -> Dict[str, float]:
    simulation_tests: int = 1

    # load metrics
    filenames: List[str] = glob.glob(os.path.join(experiment_path, 'openai-*.json'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {experiment_path}')
    with open(filenames[0]) as file:
        metrics: dict = json.load(file)

    # load server log
    filenames: List[str] = glob.glob(os.path.join(experiment_path, 'server_out.log'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {experiment_path}')
    with open(filenames[0]) as file:
        server_log: str = file.read()

    # extract model
    pattern = r"model='([^']*)'"
    found = re.findall(pattern, server_log)
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    model = os.path.basename(found[0])

    # extract adapter slots
    pattern = r'max_loras=(\d+)'
    found = re.findall(pattern, server_log)[-1]
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    adapter_slots = int(found)

    # extract max token capacity
    pattern = r'GPU KV cache size: ([0-9]+([,][0-9]*)?|[.][0-9]+) tokens'
    found = re.findall(pattern, server_log)[-1]
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    available_GPU_memory = int(found[0].replace(',', ''))

    # extract max batched tokens
    pattern = r'Chunked prefill is enabled with max_num_batched_tokens=(\d+).'
    found = re.findall(pattern, server_log)[-1]
    if found is None:
        raise ValueError(f'Metric pattern not found on result log')
    max_num_batched_tokens = int(found)

    # retrieve start and end times of benchmark
    start_time: float = float(metrics['start_time'])
    end_time: float = float(metrics['end_time'])

    # retrieve adapters information and transform rank to tokens
    served_adapters: List[str] = []
    served_adapters_sizes: List[int] = []
    with open(os.path.join(experiment_path, 'adapters.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for adapter_id, adapter_path in reader:
            served_adapters.append(adapter_id)
            adapter_rank = int(adapter_path.split('dummy_rank_')[1].replace('/', ''))
            served_adapters_sizes.append(adapter_rank)

    # check without offloading case makes sense with adapter information
    if without_offloading:
        assert adapter_slots == len(served_adapters)

    # retrieve arrivals information
    request_arrivals: List[Tuple[float, Tuple[int, int, str]]] = []
    if mean_version:
        total_inputs: List[int] = []
        total_outputs: List[int] = []
    with open(os.path.join(experiment_path, 'arrivals.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for arrival_time, input_tokens, output_tokens, adapter_id in reader:
            arrival_time = float(arrival_time)
            request_arrivals.append(
                (
                    arrival_time,
                    (
                        int(input_tokens),
                        int(output_tokens),
                        adapter_id
                    )
                )
            )
            if mean_version:
                total_inputs.append(int(input_tokens))
                total_outputs.append(int(output_tokens))
    if mean_version:
        total_inputs: np.ndarray = np.asarray(total_inputs)
        total_outputs: np.ndarray = np.asarray(total_outputs)
        mean_inputs = np.mean(total_inputs)
        std_inputs = np.std(total_inputs)
        mean_outputs = np.mean(total_outputs)
        std_outputs = np.std(total_outputs)
        '''total_new_inputs = []
        total_new_outputs = []'''
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
            '''total_new_inputs.append(input_tokens)
            total_new_outputs.append(output_tokens)'''

        '''import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, data in zip(axes.flat, [
            ('Real input', total_inputs),
            ('Sim input', total_outputs),
            ('Real output', np.asarray(total_new_inputs)),
            ('Sim output', np.asarray(total_new_outputs))
        ]):
            counts, bin_edges = np.histogram(data[1], bins=100)
            ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(data[0])
        plt.tight_layout()
        plt.savefig('/home/ferran/Downloads/lengths.png')'''

    heapq.heapify(request_arrivals)

    # retrieve maximum waiting queue
    if include_network_collapse:
        max_progress_num_waiting = int(np.max(np.load(os.path.join(experiment_path, 'num_waiting.npy'))))

    simulator: DynamicSimulatorManager = DynamicSimulatorManager(
        start_time=start_time,
        finish_time=end_time,
        model=model,
        adapter_slots=adapter_slots,
        served_adapters=served_adapters,
        served_adapters_sizes=served_adapters_sizes,
        available_gpu_memory=available_GPU_memory,
        max_num_batched_tokens=max_num_batched_tokens,
        request_arrivals=request_arrivals,
        print_outcome=print_outcome,
        include_computation_overhead=include_computation_overhead,
        include_loading_overhead=(not without_offloading),
        include_preemption=include_preemption,
        include_network_collapse=max_progress_num_waiting if include_network_collapse else None,
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
        '--experiment-path',
        type=str,
        required=True
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
        default=True,
        action='store_true',
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
    simulation_output = monitor_avg_usage(simulation_pipeline,
        experiment_path=args.experiment_path,
        mean_version=args.use_mean_version,
        include_computation_overhead=args.include_computation_overhead,
        include_preemption=args.include_preemption,
        without_offloading=args.without_offloading,
        include_network_collapse=args.include_network_collapse,
        debug_path=args.debug_path,
        print_outcome=args.print_outcome,
        timeout=args.timeout
    )

    if simulation_output is not None and args.output_path is not None:
        with open(os.path.join(args.output_path, 'simulation_results.json'), 'w') as f:
            json.dump(simulation_output, f, indent=4)
