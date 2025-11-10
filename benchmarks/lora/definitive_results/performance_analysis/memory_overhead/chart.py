import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


def extract_experiment_metric(path: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
    for i in range(len(filenames) - 1, -1, -1):
        if 'intermediate' in filenames[i]:
            del filenames[i]
    if len(filenames) == 1:
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

        # compute throughput
        output['total_throughput'] = float(metrics['input_throughput'] + metrics['output_throughput'])

        # compute itl
        output['itl'] = float(metrics['mean_itl_ms'])
    else:
        raise ValueError(f'More than one output result file or none {filenames} for path {path}')

    # compute max batch size
    num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
    output['max_batch_size'] = np.max(num_running)

    # compute mean batch size
    num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
    output['mean_batch_size'] = np.mean(num_running)

    return output


def extract_results(path: str) -> List[Dict[str, float]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['m', 'rank']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if folder == '_':
                continue
            m: int = int(folder.split('_')[1])
            rank: int = int(folder.split('_')[2])
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


def plot_memory_overhead_throughput_small(
        model_results: Dict[str, Dict[str, List[Dict[str, float]]]],
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

    model_throughput_results = []
    for label_model, model_results_small in model_results.items():
        results_throughput_list = []
        for label_results, results in model_results_small.items():
            if label_results == 'SmallRequest':
                continue
            results_throughput = __prepare_lines(
                results,
                'm',
                'total_throughput',
                'rank'
            )
            results_throughput_list.append((label_results, results_throughput))
        model_throughput_results.append((label_model, results_throughput_list))

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.3, 1], wspace=0.)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1, sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)

    axs = [ax1, ax2, ax3]

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    color_by_rank = {}
    linestyles = {
        'Llama-2-7B': 'solid',
        'Llama-2-13B': 'dotted'
    }
    for label_model, results_throughput_list in model_throughput_results:
        for index_x, (label_results, results_throughput) in enumerate(results_throughput_list):
            axs[index_x].add_line(Line2D([], [], color="none", label=label_model))
            for rank, x_line, y_line in results_throughput:
                if rank not in color_by_rank:
                    line = axs[index_x].plot(
                        x_line,
                        y_line,
                        marker='o',
                        linestyle=linestyles[label_model],
                        label=f'rank {rank}'
                    )[0]
                    color_by_rank[rank] = line.get_color()
                else:
                    axs[index_x].plot(
                        x_line,
                        y_line,
                        marker='o',
                        linestyle=linestyles[label_model],
                        label=f'rank {rank}',
                        color=color_by_rank[rank]
                    )
                x_final_point = x_line[-1] + (x_line[-1] - x_line[-2])
                y_final_point = y_line[-1] + (y_line[-1] - y_line[-2])
                axs[index_x].plot(
                    x_line[-1:] + [x_final_point],
                    y_line[-1:] + [y_final_point],
                    marker='x',
                    markevery=[1],
                    linestyle=linestyles[label_model],
                    color=color_by_rank[rank]
                )
            if index_x == 0:
                axs[index_x].set_ylabel('t. throughput (tokens/s)')
            '''if index_x == 1:
                print('AAAAAAA')
                axs[index_x].set_xticks([])
                axs[index_x].set_xticklabels([])'''
            axs[index_x].set_xlabel('adapters in GPU (#)')
            axs[index_x].set_title(label_results)
    plt.setp(ax2.get_yticklabels(), visible=False)

    results_batch_size = __prepare_lines(
        model_results['Llama-2-7B']['MediumRequest'],
        'm',
        'mean_batch_size',
        'rank'
    )
    index_x = 2
    for rank, x_line, y_line in results_batch_size:
        line = axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'rank {rank}'
        )[0]
        x_final_point = x_line[-1] + (x_line[-1] - x_line[-2])
        y_final_point = y_line[-1] + (y_line[-1] - y_line[-2])
        axs[index_x].plot(
            x_line[-1:] + [x_final_point],
            y_line[-1:] + [y_final_point],
            marker='x',
            markevery=[1],
            linestyle='solid',
            color=line.get_color()
        )
    axs[index_x].set_ylabel('mean batch (reqs)')
    axs[index_x].set_xlabel('adapters in GPU (#)')
    axs[index_x].set_title('MediumRequest')
    # axs.set_title(label_results)
    # axs[index_x].legend(loc='upper right')

    legend_elements = []
    for model, linestyle in linestyles.items():
        legend_elements.append(Line2D([], [], color='gray', linestyle=linestyle, label=model))
    for rank, color in color_by_rank.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f'rank {rank}'))
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.15))

    plt.savefig(os.path.join(path, f'memory_overhead.pdf'), format='pdf', bbox_inches='tight', dpi=400)


def main():
    model_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    model_str = {
        'llama-2-7b': 'Llama-2-7B',
        'llama-2-13b': 'Llama-2-13B',
    }
    for model in ['llama-2-7b', 'llama-2-13b']:
        results_p25: List[Dict[str, float]] = extract_results(f'{model}/p25_dataset')
        results_mean: List[Dict[str, float]] = extract_results(f'{model}/mean_dataset')
        results_p75: List[Dict[str, float]] = extract_results(f'{model}/p75_dataset')
        model_results[model_str[model]] = {
            'SmallRequest': results_p25,
            'MediumRequest': results_mean,
            'LargeRequest': results_p75
        }

    plot_memory_overhead_throughput_small(
        model_results,
        '.',
        ''
    )


if __name__ == '__main__':
    main()
