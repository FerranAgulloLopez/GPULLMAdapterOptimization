import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def extract_results_latency(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str) -> Dict[str, float]:
        output: Dict[str, float] = {}
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        for i in range(len(filenames) - 1, -1, -1):
            if 'intermediate' in filenames[i]:
                del filenames[i]
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as metrics_file:
            log: str = metrics_file.read()

        # compute output latency
        output_tokens: int = round(int(metrics['total_output_tokens']) / float(metrics['num_prompts']))
        output['output_latency'] = (output_tokens - 1) * float(metrics['mean_tpot_ms'])

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank', 'run']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            m: int = int(folder.split('_')[1])
            rank: int = int(folder.split('_')[2])
            run: int = int(folder.split('run_')[1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except ValueError:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                with open(os.path.join(path, folder, 'server_err.log')) as f:
                    error_log: str = f.read()
                    if 'ValueError: No available memory for the cache blocks' in error_log:
                        error_message += 'Not enough memory'
                    elif 'torch.cuda.OutOfMemoryError: CUDA out of memory' in error_log:
                        error_message += 'Not enough memory'
                    elif 'ValueError: The model\'s max seq len (4096) is larger than the maximum number of tokens that can be stored in KV cache' in error_log:
                        error_message += 'Not enough memory'
                    elif 'RuntimeError: CUDA error: uncorrectable ECC error encountered' in error_log:
                        error_message += 'ECC error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif 'RuntimeError: CUDA error: an illegal memory access was encountered' in error_log:
                        error_message += 'Memory access error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif '[Errno 98] error while attempting to bind on address' in error_log:
                        error_message += 'Port bind error'
                        rerun_errors.append(os.path.join(path, folder))
                    else:
                        error_message += 'Unknown error'
                        unknown_errors += 1
                # print(error_message)
                metrics = {}
            metrics['m'] = m
            metrics['rank'] = rank
            metrics['run'] = run
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def extract_results_throughput(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str) -> Dict[str, float]:
        output: Dict[str, float] = {}
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        for i in range(len(filenames) - 1, -1, -1):
            if 'intermediate' in filenames[i]:
                del filenames[i]
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as metrics_file:
            log: str = metrics_file.read()

        # compute throughput
        output['output_tokens'] = int(metrics['total_output_tokens'])

        # compute duration
        output['duration'] = float(metrics['duration'])

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank', 'prompts', 'run']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            m: int = int(folder.split('_')[1])
            rank: int = int(folder.split('_')[2])
            prompts: int = int(folder.split('__')[1].split('_')[0])
            run: int = int(folder.split('run_')[1])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
            except ValueError:
                error_message: str = f'WARNING! Error while extracting results -> {os.path.join(path, folder)}. '
                with open(os.path.join(path, folder, 'server_err.log')) as f:
                    error_log: str = f.read()
                    if 'ValueError: No available memory for the cache blocks' in error_log:
                        error_message += 'Not enough memory'
                    elif 'torch.cuda.OutOfMemoryError: CUDA out of memory' in error_log:
                        error_message += 'Not enough memory'
                    elif 'ValueError: The model\'s max seq len (4096) is larger than the maximum number of tokens that can be stored in KV cache' in error_log:
                        error_message += 'Not enough memory'
                    elif 'RuntimeError: CUDA error: uncorrectable ECC error encountered' in error_log:
                        error_message += 'ECC error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif 'RuntimeError: CUDA error: an illegal memory access was encountered' in error_log:
                        error_message += 'Memory access error'
                        rerun_errors.append(os.path.join(path, folder))
                    elif '[Errno 98] error while attempting to bind on address' in error_log:
                        error_message += 'Port bind error'
                        rerun_errors.append(os.path.join(path, folder))
                    else:
                        error_message += 'Unknown error'
                        unknown_errors += 1
                print(error_message)
                metrics = {}
            metrics['m'] = m
            metrics['rank'] = rank
            metrics['prompts'] = prompts
            metrics['run'] = run
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return results


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, (x_values, y_values) in output_tmp.items():
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        x_line.sort()
        output.append(
            (
                key,
                x_line,
                y_line
            )
        )
    output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def merge_results(results: List[Dict[str, float]], id_metrics: List[str], keys_to_merge: Set[str]) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    output_tmp: Dict[str, Dict[str, float]] = {}
    for item in results:
        found_all: bool = True
        for metric_key in keys_to_merge:
            if metric_key not in item:
                found_all = False
                break
        if not found_all:
            continue

        _id: str = create_id(item, id_metrics)
        if _id not in output_tmp:
            output_tmp[_id] = {key: 0 for key in keys_to_merge}
            for metric_key in id_metrics:
                output_tmp[_id][metric_key] = item[metric_key]
            output_tmp[_id]['runs'] = 0
        output_tmp[_id]['runs'] += 1
        for metric_key in keys_to_merge:
            output_tmp[_id][metric_key] += item[metric_key]

    output: List[Dict[str, float]] = []
    for item in output_tmp.values():
        new_item: Dict[str, float] = {}
        for metric_key in id_metrics:
            new_item[metric_key] = item[metric_key]
        new_item['runs'] = item['runs']
        for metric_key in keys_to_merge:
            new_item[metric_key] = item[metric_key] / item['runs']
        output.append(new_item)

    return output


def plot_latency_overhead(
        latency_results: Dict[str, List[Dict[str, float]]],
        loading_results: Dict[str, Dict[str, Dict[str, float]]],
        path: str,
        title: str,
) -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.0,
        'mathtext.default': 'regular',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4,
        'figure.figsize': (13, 3)  # Adjusted for balanced horizontal layout
    })
    processed_latency_results_from_disk = []
    processed_latency_results_from_cpu = []
    for dataset, results in latency_results.items():
        for exp in results:
            loading_time: float = \
                (
                        loading_results[f'rank_{exp["rank"]}']['first_load']['disk->CPUMem'] + \
                        loading_results[f'rank_{exp["rank"]}']['first_load']['CPUMem->GPUMem']
                ) * 1000
            exp['loading_time_from_disk'] = loading_time
            loading_time: float = \
                (
                        loading_results[f'rank_{exp["rank"]}']['consecutive_loads']['disk->CPUMem'] + \
                        loading_results[f'rank_{exp["rank"]}']['consecutive_loads']['CPUMem->GPUMem']
                ) * 1000
            exp['loading_time_from_cpu'] = loading_time
        processed_latency_results_from_disk.append(
            (
                dataset,
                __prepare_lines(
                    results,
                    'loading_time_from_disk',
                    'output_latency',
                    'rank'
                )
            )
        )
        processed_latency_results_from_cpu.append(
            (
                dataset,
                __prepare_lines(
                    results,
                    'loading_time_from_cpu',
                    'output_latency',
                    'rank'
                )
            )
        )

    new_labels = {
        'SmallRequest': 'SmallRequest',
        'MediumRequest': 'MediumRequest',
        'LargeRequest': 'LargeRequest',
    }

    fig, ax = plt.subplots(nrows=1, ncols=2, layout='constrained')
    fig.subplots_adjust(wspace=0.23)
    width = 0.25  # the width of the bars
    x_line_labels = ['size 8', 'size 16', 'size 32']
    x_line = np.arange(len(x_line_labels))
    for index_x, (column_title, processed_latency_results) in enumerate(
            [
                ('From disk', processed_latency_results_from_disk),
                ('From CPU', processed_latency_results_from_cpu)
            ]
    ):
        legend_elements = []
        multiplier = 0
        for label_results, results in {
            'SmallRequest': processed_latency_results[0][1],
            'MediumRequest': processed_latency_results[1][1],
            'LargeRequest': processed_latency_results[2][1],
        }.items():
            y_line = []
            for rank, loading_time, output_latency in results:
                loading_time = loading_time[0]
                output_latency = output_latency[0]
                y_line.append(loading_time / output_latency * 100)
            offset = width * multiplier
            rects = ax[index_x].bar(x_line + offset, y_line, width, label=new_labels[label_results])
            legend_elements.append(rects)
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        if index_x == 0:
            ax[index_x].set_ylabel('relative load impact (%)')
        # ax.set_xlabel('')
        # ax.set_title('')
        ax[index_x].set_title(column_title)
        ax[index_x].set_xticks(x_line + width, x_line_labels)

    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.16))
    plt.savefig(os.path.join(path, f'loading_overhead.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    model = 'llama-2-7b'

    results_p25: List[Dict[str, float]] = merge_results(extract_results_latency(f'latency/{model}/p25_dataset'), ['m', 'rank'], {'output_latency'})
    results_mean: List[Dict[str, float]] = merge_results(extract_results_latency(f'latency/{model}/mean_dataset'), ['m', 'rank'], {'output_latency'})
    results_p75: List[Dict[str, float]] = merge_results(extract_results_latency(f'latency/{model}/p75_dataset'), ['m', 'rank'], {'output_latency'})
    latency_results: Dict[str, List[Dict[str, float]]] = {
        'SmallRequest': results_p25,
        'MediumRequest': results_mean,
        'LargeRequest': results_p75
    }

    with open('loading/results.json') as loading_file:
        loading_results: Dict[str, Dict[str, Dict[str, float]]] = json.load(loading_file)

    plot_latency_overhead(
        latency_results,
        loading_results,
        '.',
        ''
    )


if __name__ == '__main__':
    main()
