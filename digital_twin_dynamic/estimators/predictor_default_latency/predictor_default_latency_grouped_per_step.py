import copy
import glob
import json
import os
import random
from typing import List, Tuple, Dict, Set, Any
from matplotlib.patches import Patch
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


CONSTANTS_PER_MODEL = {
    'llama-3.1-8b-instruct': {
        'constant_1': None,
        'constant_2': None
    },
    'qwen-2.5-7b-instruct': {
        'constant_1': 0.02877584,
        'constant_2': 14.673757
    }
}


def default_latency_predictor_grouped(
        x,
        constant_1=CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_1'],
        constant_2=CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_2']
):
    batched_tokens = x[0]
    # assert np.all(mean_batch_size > 0)
    return constant_1 * batched_tokens + constant_2


def __prepare_lines(results: List[Dict[str, float]], x_axis: str, y_axis: str, selection: str, filter_in: Tuple[str, str] = None, additional_line: str = None) -> List[
    Tuple[str, List[int], List[float]]]:
    output_tmp: Dict[str, Tuple[List[int], List[float]]] = {}
    for item in results:
        if selection not in item:
            continue
        selection_id = item[selection]
        if filter_in is not None and str(item[filter_in[0]]) != filter_in[1]:
            continue
        if selection_id not in output_tmp:
            output_tmp[selection_id] = ([], [])
            if additional_line is not None:
                output_tmp[selection_id] = ([], [], [])
        if x_axis not in item:
            output_tmp[selection_id][0].append(None)
        else:
            output_tmp[selection_id][0].append(item[x_axis])
        if y_axis not in item:
            output_tmp[selection_id][1].append(None)
        else:
            output_tmp[selection_id][1].append(item[y_axis])
        if additional_line is not None:
            if additional_line not in item:
                output_tmp[selection_id][2].append(None)
            else:
                output_tmp[selection_id][2].append(item[additional_line])
    output: List[Tuple[str, List[int], List[float]]] = []
    for key, values in output_tmp.items():
        if additional_line is None:
            x_values, y_values = values
        else:
            x_values, y_values, z_values = values
        x_line = [x_value for index, x_value in enumerate(x_values) if
                  x_value is not None and y_values[index] is not None]
        y_line = [y_value for index, y_value in enumerate(y_values) if
                  y_value is not None and x_values[index] is not None]
        if additional_line is not None:
            z_line = [z_value for index, z_value in enumerate(z_values) if
                  z_value is not None and x_values[index] is not None]
        y_line = [y_value for _, y_value in sorted(zip(x_line, y_line))]
        if additional_line is not None:
            z_line = [z_value for _, z_value in sorted(zip(x_line, z_line))]
        x_line.sort()
        if additional_line is None:
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


def extract_results(path: str) -> List[Dict[str, float]]:
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

    def extract_experiment_metric(path: str) -> List[Dict[str, Any]]:
        output: List[Dict[str, float]] = []
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')

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

        # compute scheduler metrics (scheduler time and mean loras by batch)
        pattern = f'Timestamp: ((\d+\.\d+)(e-\d+)?). Step time: ((\d+\.\d+)(e-\d+)?) seconds. Batched tokens: ((\d+\.?(\d+)?)?(e-\d+)?)'
        progress_data = re.findall(pattern, log)
        if len(progress_data) == 0:
            raise ValueError(f'Metric pattern not found on result log')
        progress_time: List[float] = []
        progress_step_time: List[float] = []
        progress_batched_tokens: List[int] = []
        for data_point in progress_data:
            progress_time.append(float(data_point[0]))
            progress_step_time.append(float(data_point[3]))
            progress_batched_tokens.append(round(float(data_point[6])))
        progress_step_time: np.ndarray = cut_timewise(
            start_time,
            end_time,
            np.asarray(progress_time),
            np.asarray(progress_step_time)
        )
        progress_batched_tokens: np.ndarray = cut_timewise(
            start_time,
            end_time,
            np.asarray(progress_time),
            np.asarray(progress_batched_tokens)
        )
        for index in range(len(progress_step_time)):
            if progress_step_time[index] < 0.1:
                output.append(
                    {
                        'step_time': progress_step_time[index] * 1000,
                        'batched_tokens': progress_batched_tokens[index],
                        'aux': 0
                    }
                )
        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['set_batch_size', 'set_input_length', 'set_output_length']
    results = []
    errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            _id = create_id(
                {
                    'set_batch_size': int(folder.split('_')[1]),
                    'set_input_length': int(folder.split('_')[3]),
                    'set_output_length': int(folder.split('_')[4])
                },
                id_metrics
            )
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            try:
                results += extract_experiment_metric(os.path.join(path, folder))
            except ValueError as e:
                print(e)
                errors += 1
    print(f'Errors: {errors}. Should be zero.')
    return results


def plot_relationship(
        results: List[Dict[str, float]],
        path: str,
        title: str,
) -> None:
    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))

    processed_results = __prepare_lines(
        results,
        'batched_tokens',
        'step_time',
        'aux'
    )
    processed_results = processed_results[0]
    x_line = processed_results[1]
    y_line = processed_results[2]
    axs.scatter(x_line, y_line, label='data')

    axs.set_ylabel('inter-token latency (ms)', fontsize=10)
    axs.set_xlabel('batched tokens (toks)', fontsize=10)
    axs.legend(fontsize=10, loc='upper right')

    plt.savefig(os.path.join(path, f'relationship_{title}'.replace('.', '-')), bbox_inches='tight')


def plot_predictor(
        results: List[Dict[str, float]],
        path: str,
        title: str,
) -> None:
    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))

    processed_results = __prepare_lines(
        results,
        'batched_tokens',
        'step_time',
        'aux'
    )
    processed_results = processed_results[0]
    x_line = processed_results[1]
    y_line = processed_results[2]
    axs.scatter(x_line, y_line, label='data')

    xdata = []
    ydata = []
    for index in range(len(x_line)):
        xdata.append([x_line[index]])
        ydata.append(y_line[index])
    aux_list = list(zip(xdata, ydata))
    random.shuffle(aux_list)
    xdata, ydata = zip(*aux_list)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    xdata = np.transpose(xdata)

    popt_exponential, _ = curve_fit(default_latency_predictor_grouped, xdata, ydata, maxfev=15000)
    print(f'Learnt constants for model {title}: {popt_exponential}')

    z_line = default_latency_predictor_grouped(
        np.asarray([x_line]),
        popt_exponential[0],
        popt_exponential[1]
    )
    axs.plot(
        x_line,
        z_line,
        linestyle='dotted',
        label='model'
    )

    axs.set_ylabel('inter-token latency (ms)', fontsize=10)
    axs.set_xlabel('batched tokens (toks)', fontsize=10)
    axs.legend(fontsize=10, loc='upper right')

    plt.savefig(os.path.join(path, f'predictor_{title}'.replace('.', '')), bbox_inches='tight')


def main():
    # 'llama-3.1-8b-instruct'
    for model in ['qwen-2.5-7b-instruct']:
        results = extract_results(f'{model}/grouped_per_step')

        plot_relationship(
            results,
            '',
            f'{model}_grouped_per_step'
        )

        plot_predictor(
            results,
            '',
            f'{model}_grouped_per_step'
        )


if __name__ == '__main__':
    main()
