import copy
import random
import glob
import json
import os
import re
from collections import deque
from typing import List, Tuple, Dict, Set
#from matplotlib.patches import Patch

#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.lines import Line2D
#from scipy.optimize import curve_fit


DECODE_CONSTANTS_PER_MODEL = {
    'llama-3.1-8b-instruct': {
        'constant_1': 1.93011757,
        'constant_2': 200.01448221,
        'constant_3': 1.
    },
    'qwen-2.5-7b-instruct': {
        'constant_1': 1.66851571,
        'constant_2': 177.69399315,
        'constant_3': 1.
    }
}


def latency_overhead_predictor_decode(
        x,
        constant_1=DECODE_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_1'],
        constant_2=DECODE_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_2'],
        constant_3=DECODE_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_3']
):
    loras_in_batch = x[0]
    rank = x[1]  # not used
    # assert np.all(mean_batch_size > 0)
    # return constant_1 / (1 + np.exp(-constant_2 * (loras_in_batch - constant_3)))
    return constant_1 * loras_in_batch + constant_2


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str, filter_in: Tuple[str, str] = None, filter_out: Tuple[str, List[str]] = None, add_all_info: bool = False) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        selection_id = item[selection]
        if filter_in is not None and str(item[filter_in[0]]) != filter_in[1]:
            continue
        if filter_out is not None and str(item[filter_out[0]]) in filter_out[1]:
            continue
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
            if add_all_info:
                output_tmp[selection_id] = ([], [], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
        if add_all_info:
            output_tmp[selection_id][2].append(item)
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, values in output_tmp.items():
        if not add_all_info:
            x_values, y_values = values
        else:
            x_values, y_values, z_values = values
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        if add_all_info:
            z_line = [z_value for index, z_value in enumerate(z_values) if x_values[index] is not None and y_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        if add_all_info:
            z_line = [z_value for _, z_value in sorted(zip(x_line, z_line))]
        x_line.sort()
        if not add_all_info:
            output.append(
                (
                    key,
                    x_line,
                    y_line
                )
            )
        else:
            output.append(
                (
                    key,
                    x_line,
                    y_line,
                    z_line
                )
            )
    output = [value for _, value in sorted(zip([value[0] for value in output], output))]

    return output


def extract_results(path: str, _type: str) -> Tuple[float, List[Dict[str, float]]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def cut_timewise(start_time: float, end_time: float, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        assert np.shape(times)[0] == np.shape(values)[0]
        start_index = 0
        end_index = np.shape(times)[0]
        while times[start_index] < start_time:
            start_index += 1
        while times[end_index - 1] > end_time:
            end_index -= 1
        return values[start_index:end_index]

    def extract_experiment_metric(path: str) -> Dict[str, float]:
        output: Dict[str, float] = {}
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none')

        # load summary file
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

        # load server log
        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            log: str = file.read()

        # extract start time
        start_time: float = float(metrics['start_time'])

        # extract end time
        end_time: float = float(metrics['end_time'])

        # compute prompts
        output['num_prompts'] = int(metrics['num_prompts'])

        # compute itl
        output['mean_itl_ms'] = float(metrics['mean_itl_ms'])

        # compute ttft
        output['mean_ttft_ms'] = float(metrics['mean_ttft_ms'])

        # compute throughput
        output['throughput'] = (float(metrics['total_input_tokens']) + float(metrics['total_output_tokens'])) / float(metrics['duration'])

        # compute batch size progress
        progress_time: np.ndarray = np.load(os.path.join(path, 'time.npy'))
        progress_num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        progress_num_running = cut_timewise(start_time, end_time, progress_time, progress_num_running)

        # compute max batch size
        if np.shape(progress_num_running)[0] > 0:
            output['max_batch_size'] = float(np.max(progress_num_running))
        else:
            output['max_batch_size'] = 0

        # compute mean batch size
        if np.shape(progress_num_running)[0] > 0:
            output['mean_batch_size'] = float(np.mean(progress_num_running))
        else:
            output['mean_batch_size'] = 0

        # compute mean input tokens per batch
        output['total_input_tokens'] = float(metrics['total_input_tokens'])

        # compute output tokens per request
        output['req_output_length'] = int(float(metrics['total_output_tokens']) / int(metrics['num_prompts']))

        # compute duration
        output['duration'] = float(metrics['duration'])

        # compute scheduler metrics (scheduler time and mean loras by batch)
        pattern = f'Timestamp: ((\d+\.\d+)(e-\d+)?). Scheduler time: ((\d+\.\d+)(e-\d+)?) seconds. Unique adapters by batch: ((\d+\.(\d+)?)(e-\d+)?)'
        progress_data = re.findall(pattern, log)
        if progress_data is None:
            raise ValueError(f'Metric pattern not found on result log')
        progress_time: List[float] = []
        progress_scheduler: List[float] = []
        progress_loras: List[int] = []
        for data_point in progress_data:
            progress_time.append(float(data_point[0]))
            progress_scheduler.append(float(data_point[3]))
            progress_loras.append(round(float(data_point[6])))
        progress_scheduler: np.ndarray = cut_timewise(
            start_time,
            end_time,
            np.asarray(progress_time),
            np.asarray(progress_scheduler)
        )
        progress_loras: np.ndarray = cut_timewise(
            start_time,
            end_time,
            np.asarray(progress_time),
            np.asarray(progress_loras)
        )
        output['scheduler_time'] = float(np.sum(progress_scheduler))
        output['mean_loras_by_batch'] = float(np.mean(progress_loras))

        return output

    total_duration: float = 0
    collected_ids: Set[str] = set()
    results = []
    errors = 0
    if _type == 'initial_points':
        id_metrics = ['set_batch_size', 'set_input_length', 'set_output_length']
    elif _type == 'overhead':
        id_metrics = ['set_batch_size', 'rank', 'set_input_length', 'set_output_length', 'cpu_loras']
    else:
        raise NotImplementedError
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            if _type == 'initial_points':
                set_batch_size: int = int(folder.split('_')[1])
                set_input_length: int = int(folder.split('_')[3])
                set_output_length: int = int(folder.split('_')[4])
            elif _type == 'overhead':
                set_batch_size: int = int(folder.split('_')[1])
                rank: int = int(folder.split('_')[3])
                set_input_length: int = int(folder.split('_')[5])
                set_output_length: int = int(folder.split('_')[6])
                cpu_loras: int = int(folder.split('_')[7])
            else:
                raise NotImplementedError

            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
                total_duration += metrics['duration']
            except ValueError as e:
                print(os.path.join(path, folder), e)
                errors += 1
                metrics = {}
            if _type == 'initial_points':
                metrics['set_batch_size'] = set_batch_size
                metrics['set_input_length'] = set_input_length
                metrics['set_output_length'] = set_output_length
            elif _type == 'overhead':
                metrics['set_batch_size'] = set_batch_size
                metrics['rank'] = rank
                metrics['set_input_length'] = set_input_length
                metrics['set_output_length'] = set_output_length
                metrics['cpu_loras'] = cpu_loras
            else:
                raise NotImplementedError

            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {errors}. Should be zero.')
    return total_duration, results


def plot_relationship(
        results_output_initial_points: List[Dict[str, float]],
        results_output_overheads: List[Dict[str, float]],
        path: str,
        title: str,
) -> None:
    all_results = {
        'decode': results_output_overheads
    }
    all_metrics_to_show = {
        'decode': [
            'mean_itl_ms',
            'mean_itl_ms-overhead',
            'throughput',
            'mean_batch_size',
            'mean_loras_by_batch',
            'duration',
            'scheduler_time'
        ]
    }

    # convert initial points to dict with itl
    initial_points_output_reference: Dict[str, Dict[str, float]] = {}
    for result in results_output_initial_points:
        _id: str = f'{result["set_batch_size"]}_{result["set_input_length"]}_{result["set_output_length"]}'
        if 'mean_itl_ms' not in result:
            raise ValueError('Missing mandatory results')
        initial_points_output_reference[_id] = {}
        for metric in all_metrics_to_show['decode']:
            initial_points_output_reference[_id][metric.replace('-overhead', '')] = result[metric.replace('-overhead', '')]
    del results_output_initial_points
    initial_points_reference = {
        'decode': initial_points_output_reference
    }

    # get different rank and batch values
    batch_values_decode: Set[int] = set()
    rank_values: Set[int] = set()
    for results in results_output_overheads:
        rank_values.add(results['rank'])
        batch_values_decode.add(results['set_batch_size'])
    batch_values: Dict[str, List[int]] = {
        'decode': sorted(batch_values_decode)
    }
    rank_values: List[int] = sorted(rank_values)

    # plot
    color_by_rank = {}
    for label_results, metrics_to_show in all_metrics_to_show.items():
        nrows = len(metrics_to_show)
        ncols = max(len(batch_values[label_results]), 2)
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True, sharey='row')
        fig.subplots_adjust(hspace=0, wspace=0)

        line_style_by_type = {}
        available_line_styles = deque()
        available_line_styles.append('solid')
        available_line_styles.append('dotted')
        available_line_styles.append('dashed')

        for index_y, metric_to_show in enumerate(metrics_to_show):
            for index_x, set_batch_size in enumerate(batch_values[label_results]):
                for rank in rank_values:
                    processed_results = __prepare_lines(
                        [results for results in all_results[label_results] if results['set_batch_size'] == set_batch_size],
                        'cpu_loras',
                        metric_to_show.replace('-overhead', ''),
                        'set_input_length' if label_results == 'prefill' else 'set_output_length',
                        filter_in=('rank', str(rank)),
                        add_all_info=True
                    )
                    for type_value, x_line, y_line, z_line in processed_results:
                        # add initial point
                        _id: str = f'{z_line[0]["set_batch_size"]}_{z_line[0]["set_input_length"]}_{z_line[0]["set_output_length"]}'
                        x_line = [0] + x_line
                        y_line = [initial_points_reference[label_results][_id][metric_to_show.replace('-overhead', '')]] + y_line

                        # add overhead if required
                        if 'overhead' in metric_to_show:
                            y_line = [(y_line[index_value] / y_line[0]) * 100 for index_value in range(len(y_line))]

                        # plot
                        if type_value not in line_style_by_type:
                            line_style_by_type[type_value] = available_line_styles.popleft()
                        if rank not in color_by_rank:
                            line = axs[index_y, index_x].plot(
                                x_line,
                                y_line,
                                marker='o',
                                linestyle=line_style_by_type[type_value]
                            )[0]
                            color_by_rank[rank] = line.get_color()
                        else:
                            axs[index_y, index_x].plot(
                                x_line,
                                y_line,
                                marker='o',
                                linestyle=line_style_by_type[type_value],
                                color=color_by_rank[rank]
                            )

                axs[index_y, index_x].set_xlabel('CPU adapters (#)', fontsize=10)
                if index_y == 0:
                    axs[index_y, index_x].set_title(f'Batch size {set_batch_size}')

            axs[index_y, 0].set_ylabel(metric_to_show, fontsize=10)

        legend_elements = []
        color_legend = [Line2D([], [], color=color, label=f'rank {rank}') for rank, color in color_by_rank.items()]
        for label, style in line_style_by_type.items():
            legend_elements.append(Line2D([], [], color='gray', linestyle=style, label=label))
        fig.legend(handles=legend_elements + color_legend, fontsize=10, loc='upper center', ncol=len(color_legend) + len(line_style_by_type), bbox_to_anchor=(0.5, 0.92))
        plt.savefig(os.path.join(path, f'relationship_{title}_{label_results}'.replace('.', '-')), bbox_inches='tight')


def plot_predictor(
        results_output_initial_points: List[Dict[str, float]],
        results_output_overheads: List[Dict[str, float]],
        path: str,
        title: str,
) -> None:
    all_results = {
        'decode': results_output_overheads
    }
    all_metrics_to_show = {
        'decode': [
            'mean_itl_ms-overhead'
        ]
    }

    # convert initial points to dict with itl
    initial_points_output_reference: Dict[str, Dict[str, float]] = {}
    for result in results_output_initial_points:
        _id: str = f'{result["set_batch_size"]}_{result["set_input_length"]}_{result["set_output_length"]}'
        if 'mean_itl_ms' not in result:
            raise ValueError('Missing mandatory results')
        initial_points_output_reference[_id] = {}
        for metric in all_metrics_to_show['decode']:
            initial_points_output_reference[_id][metric.replace('-overhead', '')] = result[metric.replace('-overhead', '')]
    del results_output_initial_points
    initial_points_reference = {
        'decode': initial_points_output_reference
    }

    # get different rank and batch values
    batch_values_decode: Set[int] = set()
    rank_values: Set[int] = set()
    for results in results_output_overheads:
        rank_values.add(results['rank'])
        batch_values_decode.add(results['set_batch_size'])
    batch_values: Dict[str, List[int]] = {
        'decode': sorted(batch_values_decode)
    }

    # plot
    color_by_rank = {}
    for label_results, metrics_to_show in all_metrics_to_show.items():
        nrows = len(metrics_to_show)
        ncols = 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True, sharey='row')
        fig.subplots_adjust(hspace=0, wspace=0)

        marker_style_by_type = {}
        available_marker_styles = deque()
        available_marker_styles.append('o')
        available_marker_styles.append('*')
        available_marker_styles.append('x')

        xdata = []
        ydata = []
        max_x_line = None
        for index_y, metric_to_show in enumerate(metrics_to_show):
            for set_batch_size in batch_values[label_results]:
                if set_batch_size not in marker_style_by_type:
                    marker_style_by_type[set_batch_size] = available_marker_styles.popleft()
                for rank in rank_values:
                    processed_results = __prepare_lines(
                        [results for results in all_results[label_results] if results['set_batch_size'] == set_batch_size],
                        'mean_loras_by_batch',
                        metric_to_show.replace('-overhead', ''),
                        'set_input_length' if label_results == 'prefill' else 'set_output_length',
                        filter_in=('rank', str(rank)),
                        add_all_info=True
                    )
                    for type_value, x_line, y_line, z_line in processed_results:
                        if max_x_line is None or max(x_line) > max_x_line:
                            max_x_line = max(x_line)

                        # add initial point
                        _id: str = f'{z_line[0]["set_batch_size"]}_{z_line[0]["set_input_length"]}_{z_line[0]["set_output_length"]}'
                        x_line = [0] + x_line
                        y_line = [initial_points_reference[label_results][_id][metric_to_show.replace('-overhead', '')]] + y_line

                        # add overhead if required
                        if 'overhead' in metric_to_show:
                            y_line = [(y_line[index_value] / y_line[0]) * 100 for index_value in range(len(y_line))]

                        # remove first point
                        x_line = x_line[1:]
                        y_line = y_line[1:]

                        # plot
                        if rank not in color_by_rank:
                            line = axs.plot(
                                x_line,
                                y_line,
                                marker=marker_style_by_type[set_batch_size],
                                linestyle='None'
                            )[0]
                            color_by_rank[rank] = line.get_color()
                        else:
                            axs.plot(
                                x_line,
                                y_line,
                                marker=marker_style_by_type[set_batch_size],
                                linestyle='None',
                                color=color_by_rank[rank]
                            )

                        for index_y in range(len(x_line)):
                            xdata.append([x_line[index_y], rank])
                            ydata.append(y_line[index_y])

                axs.set_xlabel('Unique adapters by batch (#)', fontsize=10)

            axs.set_ylabel(metric_to_show, fontsize=10)

        aux_list = list(zip(xdata, ydata))
        random.shuffle(aux_list)
        xdata, ydata = zip(*aux_list)
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        xdata = np.transpose(xdata)

        predictor = latency_overhead_predictor_decode
        popt_exponential, _ = curve_fit(predictor, xdata, ydata, maxfev=15000)
        print(f'{label_results}. Learnt constants for model {title}: {popt_exponential}')

        for rank in rank_values:
            x_line = list(range(1, round(max_x_line)))
            axs.plot(
                x_line,
                predictor(
                    np.asarray([x_line, [rank] * len(x_line)]),
                    **{f'constant_{index + 1}': value for index, value in enumerate(popt_exponential)}
                ),
                linestyle='dotted',
                color=color_by_rank[rank]
            )

        legend_elements = []
        color_legend = [Patch(facecolor=color, edgecolor='black', label=f'rank {rank}') for rank, color in color_by_rank.items()]
        for label, style in marker_style_by_type.items():
            legend_elements.append(Line2D([], [], color='gray', marker=style, linestyle='None', label=f'batch {label}'))
        legend_elements.append(Line2D([], [], color='gray', linestyle='dotted', label=f'predicted'))
        fig.legend(handles=legend_elements + color_legend, fontsize=10, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
        plt.savefig(os.path.join(path, f'predictor_{title}_{label_results}'.replace('.', '-')), bbox_inches='tight')


def main():
    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print('#####', model)
        total_duration: float = 0

        duration, results_output_initial_points = extract_results(f'{model}/initial_points_small/output', 'initial_points')
        total_duration += duration
        duration, results_output_overheads = extract_results(f'{model}/overhead_small/output', 'overhead')
        total_duration += duration

        plot_relationship(
            results_output_initial_points,
            results_output_overheads,
            '',
            f'{model}_small'
        )

        plot_predictor(
            results_output_initial_points,
            results_output_overheads,
            '',
            f'{model}_small'
        )

        print(f'Total duration by model {model}: {total_duration}')


if __name__ == '__main__':
    main()
