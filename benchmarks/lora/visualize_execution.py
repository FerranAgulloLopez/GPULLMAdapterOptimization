import os
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import glob
import argparse
import re

from typing import List, Dict, Optional, Set

from benchmarks.lora.simulation_pipeline import simulation_pipeline, monitor_avg_usage


def main(
        experiment_path: str,
        output_path_real: str,
        output_path_simulated: str,
        simulate: bool,
        use_mean_version: bool,
        use_preemption: bool
):
    # extract start and end of benchmark
    filenames: List[str] = glob.glob(os.path.join(experiment_path, 'openai-*.json'))
    if len(filenames) != 1:
        raise ValueError(f'More than one output result file or none {filenames} for path {experiment_path}')
    with open(filenames[0]) as metrics_file:
        metrics: dict = json.load(metrics_file)
    start_time = metrics['start_time']
    end_time = metrics['end_time']

    # load time
    progress_time = np.load(os.path.join(experiment_path, 'time.npy'))

    # find start and end indexes
    start_index = 0
    end_index = len(progress_time)
    while progress_time[start_index] < start_time:
        start_index += 1
    while progress_time[end_index - 1] > end_time:
        end_index -= 1

    # waiting plot
    fig, ax = plt.subplots()
    progress_num_waiting = np.load(os.path.join(experiment_path, 'num_waiting.npy'))
    ax.plot(progress_time, progress_num_waiting, label='num waiting')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('requests (#)')
    ax.legend()
    fig.savefig(os.path.join(output_path_real, 'real_waiting'))
    print('Max waiting:', np.max(progress_num_waiting))

    # running plot
    fig, ax = plt.subplots()
    progress_num_running = np.load(os.path.join(experiment_path, 'num_running.npy'))
    ax.plot(progress_time, progress_num_running, label='num running')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('requests (#)')
    ax.legend()
    fig.savefig(os.path.join(output_path_real, 'real_running'))

    # running and waiting by adapter plot
    adapters: Set[int] = set()
    filenames: List[str] = glob.glob(os.path.join(experiment_path, 'running_by_adapter_*.npy'))
    for filename in filenames:
        adapter_id: int = int(filename.split('running_by_adapter_')[1].replace('.npy', ''))
        adapters.add(adapter_id)
    if len(adapters) > 0:
        fig, ax = plt.subplots()
        adapter_colors: Dict[int, str] = {}
        for adapter_id in adapters:
            try:
                progress_num_running = np.load(os.path.join(experiment_path, f'running_by_adapter_{adapter_id}.npy'))
                if adapter_id in adapter_colors:
                    ax.plot(progress_time, progress_num_running, label='num running', linestyle='solid', color=adapter_colors[adapter_id])
                else:
                    line = ax.plot(progress_time, progress_num_running, label='num running', linestyle='solid')[0]
                    adapter_colors[adapter_id] = line.get_color()
            except:
                continue
        for adapter_id in adapters:
            try:
                progress_num_waiting = np.load(os.path.join(experiment_path, f'waiting_by_adapter_{adapter_id}.npy'))
                if adapter_id in adapter_colors:
                    ax.plot(progress_time, progress_num_waiting, label='num waiting', linestyle='dotted', color=adapter_colors[adapter_id])
                else:
                    line = ax.plot(progress_time,progress_num_waiting, label='num waiting', linestyle='dotted')[0]
                    adapter_colors[adapter_id] = line.get_color()
            except:
                continue
        ax.set_xlabel('time (s)')
        ax.set_ylabel('requests (#)')
        legend_elements = []
        for style, label in zip(['solid', 'dotted'], ['running', 'waiting']):
            legend_elements.append(Line2D([], [], color='gray', linestyle=style, label=label))
        color_legend = [Line2D([], [], color=adapter_colors[aid], label=f'adapter {aid}') for aid in adapters]
        fig.legend(handles=legend_elements + color_legend, fontsize=10, loc='upper center', ncol=len(adapters) + 2, bbox_to_anchor=(0.5, 1.0))
        fig.savefig(os.path.join(output_path_real, 'real_running_waiting_by_adapter'))

    # finished plot
    fig, ax = plt.subplots()
    progress_num_finished = np.load(os.path.join(experiment_path, 'finished.npy'))
    ax.plot(progress_time, progress_num_finished, label='num finished (accumulated)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('requests (#)')
    ax.legend()
    fig.savefig(os.path.join(output_path_real, 'real_finished'))

    # preempted plot
    fig, ax = plt.subplots()
    progress_num_preempted = np.load(os.path.join(experiment_path, 'num_preempted.npy'))
    ax.plot(progress_time, progress_num_preempted, label='num preempted (accumulated)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('requests (#)')
    ax.legend()
    fig.savefig(os.path.join(output_path_real, 'real_preempted'))

    # kv cache plot
    fig, ax = plt.subplots()
    progress_kv_cache = np.load(os.path.join(experiment_path, 'gpu_cache_usage_perc.npy'))
    ax.plot(progress_time, progress_kv_cache, label='kv cache usage')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('kv cache usage (%)')
    ax.legend()
    fig.savefig(os.path.join(output_path_real, 'real_kv_cache_usage'))

    # arrivals
    arrivals_by_adapter: Dict[str, List[float]] = {}
    num_arrivals: int = 0
    with open(os.path.join(experiment_path, 'arrivals.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for arrival_time, input_tokens, output_tokens, adapter_id in reader:
            arrival_time = float(arrival_time)
            if adapter_id not in arrivals_by_adapter:
                arrivals_by_adapter[adapter_id] = []
            arrivals_by_adapter[adapter_id].append(arrival_time)
            num_arrivals += 1
    print('Num arrivals', num_arrivals)
    fig, ax = plt.subplots(figsize=(10, 4))
    limit: int = 10
    for index, (adapter_id, adapter_arrivals) in enumerate(arrivals_by_adapter.items()):
        if index > limit:
            break
        ax.plot(adapter_arrivals, [f'adapter {adapter_id}'] * np.shape(adapter_arrivals)[0], 'o')
    ax.set_xlabel('time')
    ax.set_ylabel('adapter')
    plt.title('Arrivals by adapter')
    plt.xticks(rotation=45)
    fig.savefig(os.path.join(output_path_real, 'arrivals_by_adapter.png'))
    fig, ax = plt.subplots(figsize=(10, 4))
    limit: int = 10
    time_bins = None
    for index, (adapter_id, adapter_arrivals) in enumerate(arrivals_by_adapter.items()):
        if index > limit:
            break
        adapter_arrivals = np.asarray(adapter_arrivals)
        if time_bins is None:
            time_bins = np.arange(adapter_arrivals.min() - 1, adapter_arrivals.max() + 1, 300)
        adapter_arrivals_counts, _ = np.histogram(adapter_arrivals, bins=time_bins)
        ax.plot(time_bins[:-1], adapter_arrivals_counts, drawstyle='steps-post', label='Request rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of requests')
    plt.title('Server Request Arrival Rate Over Time')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_path_real, 'arrivals_by_adapter_bins.png'))

    if simulate:
        assert args.output_path_simulated is not None

        # load server log
        filenames: List[str] = glob.glob(os.path.join(experiment_path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {experiment_path}')
        with open(filenames[0]) as file:
            server_log: str = file.read()

        # extract adapter slots
        pattern = 'max_loras=(\d+)'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        adapter_slots = int(found)

        # extract served adapters
        pattern = 'max_cpu_loras=(\d+)'
        found = re.findall(pattern, server_log)
        if not found:
            served_adapters = None
        else:
            served_adapters = int(found[-1])

        monitor_avg_usage(simulation_pipeline,
            experiment_path,
            mean_version=use_mean_version,
            without_offloading=False,# if served_adapters is None or served_adapters == adapter_slots else False,
            include_computation_overhead=True,
            include_preemption=use_preemption,
            include_network_collapse=True,
            debug_path=output_path_simulated,
            print_outcome=True,
            timeout=None
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output-path-real',
        type=str,
        required=True
    )
    parser.add_argument(
        '--use-simulator',
        default=False,
        action='store_true',
        help='Run simulator',
    )
    parser.add_argument(
        '--output-path-simulated',
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
        '--use-preemption',
        default=False,
        action='store_true',
        help='Run simulator with preemption',
    )
    args = parser.parse_args()
    main(
        args.experiment_path,
        args.output_path_real,
        args.output_path_simulated,
        args.use_simulator,
        args.use_mean_version,
        args.use_preemption
    )
