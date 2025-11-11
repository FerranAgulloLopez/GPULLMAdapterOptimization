import csv
import heapq
import pickle
import os
import re
import json
import glob
from typing import List, Tuple, Dict, Set, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
    _id: str = ''
    for metric_key in id_metrics:
        _id += f'{metrics[metric_key]}_'
    return _id


def extract_results_test(path: str) -> List[Dict[str, Any]]:

    def extract_experiment_metric(path: str) -> Dict[str, Any]:
        output: Dict[str, float] = {}

        # load server log
        filenames: List[str] = glob.glob(os.path.join(path, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            server_log: str = file.read()

        # load metrics
        filenames: List[str] = glob.glob(os.path.join(path, 'openai-*.json'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')
        with open(filenames[0]) as file:
            metrics: dict = json.load(file)

        # extract adapter slots
        pattern = r'max_loras=(\d+)'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['adapter_slots'] = int(found)

        # extract served adapters and sizes
        pattern = r'max_cpu_loras=(\d+)'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        output['served_adapters'] = int(found)

        # extract adapter rates
        folder_name: str = os.path.basename(os.path.dirname(path))
        output['rates'] = folder_name.split('_')[1].replace('-', ' ')

        # extract adapter sizes
        output['sizes'] = folder_name.split('_')[3].replace('-', ' ')

        # extract total time
        total_time: float = float(metrics['duration'])

        # extract total throughput
        output['total_throughput']: float = float(metrics['input_throughput']) + float(metrics['output_throughput'])

        # compute ideal total throughput
        received_input_tokens: int = 0
        received_output_tokens: int = 0
        with open(os.path.join(path, 'arrivals.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for arrival_time, input_tokens, output_tokens, adapter_id in reader:
                received_input_tokens += int(input_tokens)
                received_output_tokens += int(output_tokens)
        output['ideal_total_throughput'] = (received_input_tokens + received_output_tokens) / total_time

        # populate simulation metrics
        for metric_key, metric_value in metrics.items():
            output[metric_key] = metric_value

        return output

    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['served_adapters', 'adapter_slots', 'rates', 'sizes']
    results = []
    errors: int = 0
    for parent_dir in os.listdir(path):
        parent_dir_path = os.path.join(path, parent_dir)
        if os.path.isdir(parent_dir_path):
            for folder in os.listdir(parent_dir_path):
                folder_path = os.path.join(parent_dir_path, folder)
                if os.path.isdir(folder_path):
                    try:
                        metrics = extract_experiment_metric(folder_path)
                        metrics['path'] = folder_path
                        _id = create_id(metrics, id_metrics)
                        if _id in collected_ids:
                            raise ValueError('Repeated results')
                        collected_ids.add(_id)
                        results.append(metrics)
                    except Exception as e:
                        print(e)
                        errors += 1
    print(f'Extraction errors: {errors}. Should be zero.')
    return results


def find_max_point(x_line: List[int], y_line: List[int], y_line_ideal: List[float]) -> int:
    assert len(x_line) > 0
    highlight_index: int = -1
    while (highlight_index + 1) < len(x_line) and (1 - (y_line[highlight_index + 1] / y_line_ideal[highlight_index + 1])) < 0.1:
        highlight_index += 1
    if highlight_index == -1:
        highlight_index = 0
    return highlight_index


def plot_sample(
        all_results: List[Dict[str, Any]],
        path: str,
        title: str
) -> None:
    # define constants
    sample_size: int = 5
    x_axis: str = 'served_adapters'
    y_axis: str = 'total_throughput'
    y_axis_ideal: str = 'ideal_total_throughput'

    # processed results (group by rates and sizes)
    id_metrics: List[str] = ['rates', 'sizes']
    processed_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in all_results:
        result_id: str = create_id(result, id_metrics)
        if result_id not in processed_results:
            processed_results[result_id] = []
        processed_results[result_id].append(result)

    # choose sample
    if sample_size < len(processed_results):
        random_sample = random.sample(list(processed_results.values()), sample_size)
    else:
        random_sample = processed_results.values()

    # create lines
    lines_per_config = []
    for sample_results in random_sample:
        # merge results by adapter slots
        aux_merged_results: Dict[int, List[Dict[str, Any]]] = {}
        for result in sample_results:
            adapter_slots: int = result['adapter_slots']
            if result['adapter_slots'] not in aux_merged_results:
                aux_merged_results[adapter_slots] = []
            aux_merged_results[adapter_slots].append(result)
        # create lines
        lines = []
        for adapter_slots, results in aux_merged_results.items():
            x_line = []
            y_line = []
            y_line_ideal = []
            for result in results:
                x_line.append(result[x_axis])
                y_line.append(result[y_axis])
                y_line_ideal.append(result[y_axis_ideal])
            y_line = [item for _, item in sorted(zip(x_line, y_line))]
            y_line_ideal = [item for _, item in sorted(zip(x_line, y_line_ideal))]
            x_line = sorted(x_line)
            lines.append((f'slots {adapter_slots}', x_line, y_line, y_line_ideal))
        label = f'rates {sample_results[0]["rates"]} sizes {sample_results[0]["sizes"]}'
        lines_per_config.append((label, lines))

    # plot
    nrows = 1
    ncols = sample_size
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), sharex=True, sharey=True)
    # fig.subplots_adjust(wspace=0)

    for index_x, (title_subplot, result_lines) in enumerate(lines_per_config):
        global_maximum_x = None
        global_maximum_y = None
        global_maximum_color = None

        for label, x_line, y_line, y_line_ideal in result_lines:
            # line
            line = axs[index_x].plot(
                x_line,
                y_line,
                marker='o',
                linestyle='solid',
                label=label
            )[0]

            # max point
            max_x_point: int = find_max_point(x_line, y_line, y_line_ideal)
            if max_x_point == -1:
                print('Warn!!! Max point not found')
            else:
                axs[index_x].scatter(x_line[max_x_point], y_line[max_x_point], 10, marker='x', linewidths=20, color=line.get_color())

                if global_maximum_y is None or global_maximum_y < y_line[max_x_point]:
                    global_maximum_x = x_line[max_x_point]
                    global_maximum_y = y_line[max_x_point]
                    global_maximum_color = line.get_color()

        if global_maximum_y is not None:
            axs[index_x].scatter(global_maximum_x, global_maximum_y, 10, marker='o', linewidths=20, color=global_maximum_color, label='Global Max')

        axs[index_x].set_xlabel(x_axis, fontsize=10)
        axs[index_x].set_ylabel(y_axis, fontsize=10)
        axs[index_x].set_title(title_subplot, fontsize=10)
        axs[index_x].legend(loc='upper right', fontsize=10)

    plt.savefig(os.path.join(path, f'sample_plot_{title.replace(".", "")}'), bbox_inches='tight')


def create_dataset(
        all_results: List[Dict[str, Any]],
        path: str,
        title: str
) -> None:
    # define constants
    x_axis: str = 'served_adapters'
    y_axis: str = 'total_throughput'
    y_axis_ideal: str = 'ideal_total_throughput'

    # processed results (group by rates and sizes)
    id_metrics: List[str] = ['rates', 'sizes']
    processed_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in all_results:
        result_id: str = create_id(result, id_metrics)
        if result_id not in processed_results:
            processed_results[result_id] = []
        processed_results[result_id].append(result)

    # create dataset (extract features)
    dataset: List[Dict[str, Any]] = []
    for set_of_results in processed_results.values():
        features: Dict[str, Any] = {}
        sample_result: Dict[str, Any] = set_of_results[0]

        # rate features
        rates: str = sample_result['rates']
        rates: np.ndarray = np.asarray([float(item) for item in rates.split(' ')])
        features['max_rate'] = float(np.max(rates))
        features['min_rate'] = float(np.min(rates))
        features['mean_rate'] = float(np.mean(rates))
        features['std_rate'] = float(np.std(rates))

        # rank features
        sizes: str = sample_result['sizes']
        sizes: np.ndarray = np.asarray([float(item) for item in sizes.split(' ')])
        features['max_size'] = float(np.max(sizes))
        features['min_size'] = float(np.min(sizes))
        features['mean_size'] = float(np.mean(sizes))
        features['std_size'] = float(np.std(sizes))

        # max features
        # merge results by adapter slots
        aux_merged_results: Dict[int, List[Dict[str, Any]]] = {}
        for result in set_of_results:
            adapter_slots: int = result['adapter_slots']
            if result['adapter_slots'] not in aux_merged_results:
                aux_merged_results[adapter_slots] = []
            aux_merged_results[adapter_slots].append(result)
        max_served_adapters = None
        max_adapter_slots = None
        max_throughput = None
        for adapter_slots, results in aux_merged_results.items():
            x_line = []
            y_line = []
            y_line_ideal = []
            for result in results:
                x_line.append(result[x_axis])
                y_line.append(result[y_axis])
                y_line_ideal.append(result[y_axis_ideal])
            y_line = [item for _, item in sorted(zip(x_line, y_line))]
            y_line_ideal = [item for _, item in sorted(zip(x_line, y_line_ideal))]
            x_line = sorted(x_line)
            max_x_point: int = find_max_point(x_line, y_line, y_line_ideal)
            if max_x_point != -1:
                if max_throughput is None or max_throughput < y_line[max_x_point]:
                    max_throughput = y_line[max_x_point]
                    max_served_adapters = x_line[max_x_point]
                    max_adapter_slots = adapter_slots
        if max_throughput is None:
            print('Warn!!! Max point not found')
        else:
            features['max_served_adapters'] = max_served_adapters
            features['max_total_throughput'] = max_throughput
            features['max_adapter_slots'] = max_adapter_slots

        # reference features
        features['path'] = os.path.dirname(sample_result['path'])

        dataset.append(features)

    # save dataset
    with open(os.path.join(path, f'dataset_{title.replace(".", "")}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        header: List[str] = list(dataset[0].keys())
        writer.writerow(header)
        for dataset_row in dataset:
            writer.writerow([dataset_row[key] for key in header])


def main():
    # set random seed
    random.seed(0)
    np.random.seed(0)

    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        print(model)

        path: str = os.path.join('', model)
        all_results: List[Dict[str, Any]] = extract_results_test(path)

        '''plot_sample(
            all_results,
            '',
            model,
        )'''

        create_dataset(
            all_results,
            '.',
            model,
        )


if __name__ == '__main__':
    main()
