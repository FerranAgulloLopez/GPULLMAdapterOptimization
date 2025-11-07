import copy
import glob
import json
import os
import random
from typing import List, Tuple, Dict, Set
#from matplotlib.patches import Patch

#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.lines import Line2D
#from scipy.optimize import curve_fit


PREFILL_CONSTANTS_PER_MODEL = {
    'llama-3.1-8b-instruct': {
        'constant_1': 1.31633234e-02,
        'constant_2': 3.53299914e+03
    },
    'qwen-2.5-7b-instruct': {
        'constant_1': 1.38739477e-02,
        'constant_2': 8.81103463e+01
    }

}

DECODE_CONSTANTS_PER_MODEL = {
    'llama-3.1-8b-instruct': {
        'constant_1': 0.0437994,
        'constant_2': 10.54174836
    },
    'qwen-2.5-7b-instruct': {
        'constant_1': 0.04894489,
        'constant_2': 10.61679297
    }
}


def default_latency_predictor_prefill(
        x,
        constant_1=PREFILL_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_1'],
        constant_2=PREFILL_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_2']
):
    total_input_tokens = x[0]
    # assert np.all(total_input_tokens > 0)
    return constant_1 * total_input_tokens + constant_2


def default_latency_predictor_decode(
        x,
        constant_1=DECODE_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_1'],
        constant_2=DECODE_CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']['constant_2']
):
    mean_batch_size = x[0]
    # assert np.all(mean_batch_size > 0)
    return constant_1 * mean_batch_size + constant_2


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


def extract_results(path: str) -> Tuple[float, List[Dict[str, float]]]:
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
        for i in range(len(filenames) - 1, -1, -1):
            if 'intermediate' in filenames[i]:
                del filenames[i]
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {path}')

        # load summary file
        with open(filenames[0]) as metrics_file:
            metrics: dict = json.load(metrics_file)

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

        # compute batch size progress
        progress_num_running: np.ndarray = np.load(os.path.join(path, 'num_running.npy'))
        progress_time: np.ndarray = np.load(os.path.join(path, 'time.npy'))
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

        return output

    total_duration: float = 0
    collected_ids: Set[str] = set()
    id_metrics: List[str] = ['set_batch_size', 'set_input_length', 'set_output_length']
    results = []
    rerun_errors: List[str] = []
    unknown_errors: int = 0
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            set_batch_size: int = int(folder.split('_')[1])
            set_input_length: int = int(folder.split('_')[3])
            set_output_length: int = int(folder.split('_')[4])
            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
                total_duration += metrics['duration']
            except ValueError as e:
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
                        print(e)
                # print(error_message)
                metrics = {}
            metrics['set_batch_size'] = set_batch_size
            metrics['set_input_length'] = set_input_length
            metrics['set_output_length'] = set_output_length
            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {unknown_errors}. Should be zero.')
    print(f'Rerun errors: {len(rerun_errors)}. Should be zero. Full list: {rerun_errors}')
    return total_duration, results


def plot_relationship(
        results_input: List[Dict[str, float]],
        results_output: List[Dict[str, float]],
        path: str,
        title: str,
) -> None:
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    # fig.subplots_adjust(wspace=0)

    # prefill
    index_x = 0
    processed_results_input = __prepare_lines(
        results_input,
        'total_input_tokens',
        'mean_ttft_ms',
        'set_batch_size'
    )
    for set_batch_size, x_line, y_line in processed_results_input:
        axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'batch size {set_batch_size}'
        )

    axs[index_x].set_ylabel('average time to first token (ms)', fontsize=10)
    axs[index_x].set_xlabel('total input tokens (toks)', fontsize=10)
    axs[index_x].legend(fontsize=10, loc='upper right')
    axs[index_x].set_title('Prefill')

    # decode
    index_x = 1
    processed_results_output = __prepare_lines(
        results_output,
        'mean_batch_size',
        'mean_itl_ms',
        'req_output_length'
    )
    for output_length, x_line, y_line in processed_results_output:
        axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'output {output_length}'
        )

    axs[index_x].set_ylabel('average inter-token latency (ms)', fontsize=10)
    axs[index_x].set_xlabel('average batch size (toks)', fontsize=10)
    axs[index_x].legend(fontsize=10, loc='upper right')
    axs[index_x].set_title('Decode')

    plt.savefig(os.path.join(path, f'relationship_{title}'.replace('.', '-')), bbox_inches='tight')


def plot_predictor(
        results_input: List[Dict[str, float]],
        results_output: List[Dict[str, float]],
        path: str,
        title: str,
) -> None:
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    # fig.subplots_adjust(wspace=0)

    # prefill
    index_x = 0
    processed_results_input = __prepare_lines(
        results_input,
        'total_input_tokens',
        'mean_ttft_ms',
        'set_batch_size'
    )
    color = {}
    legend_elements = []
    xdata = []
    ydata = []
    for set_batch_size, x_line, y_line in processed_results_input:
        line = axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'batch size {set_batch_size}'
        )[0]
        color[set_batch_size] = line.get_color()
        legend_elements.append(Patch(facecolor=line.get_color(), edgecolor='black', label=line.get_label()))
        for index_y in range(len(x_line)):
            xdata.append([x_line[index_y]])
            ydata.append(y_line[index_y])
    aux_list = list(zip(xdata, ydata))
    random.shuffle(aux_list)
    xdata, ydata = zip(*aux_list)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    xdata = np.transpose(xdata)

    popt_exponential, _ = curve_fit(default_latency_predictor_prefill, xdata, ydata, maxfev=15000)
    print(f'Prefill. Learnt constants for model {title}: {popt_exponential}')

    for set_batch_size, x_line, y_line in processed_results_input:
        axs[index_x].plot(
            x_line,
            default_latency_predictor_prefill(
                np.asarray([x_line]),
                popt_exponential[0],
                popt_exponential[1]
            ),
            marker='o',
            color=color[set_batch_size],
            linestyle='dotted'
        )

    axs[index_x].set_ylabel('average time to first token (ms)', fontsize=10)
    axs[index_x].set_xlabel('total input tokens (toks)', fontsize=10)
    axs[index_x].legend(handles=legend_elements, fontsize=10, loc='upper right')
    axs[index_x].set_title('Prefill')

    # decode
    index_x = 1
    processed_results_output = __prepare_lines(
        results_output,
        'mean_batch_size',
        'mean_itl_ms',
        'req_output_length'
    )
    color = {}
    legend_elements = []
    xdata = []
    ydata = []
    for output_length, x_line, y_line in processed_results_output:
        line = axs[index_x].plot(
            x_line,
            y_line,
            marker='o',
            linestyle='solid',
            label=f'output {output_length}'
        )[0]
        color[output_length] = line.get_color()
        legend_elements.append(Patch(facecolor=line.get_color(), edgecolor='black', label=line.get_label()))
        for index_y in range(len(x_line)):
            xdata.append([x_line[index_y]])
            ydata.append(y_line[index_y])
    aux_list = list(zip(xdata, ydata))
    random.shuffle(aux_list)
    xdata, ydata = zip(*aux_list)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    xdata = np.transpose(xdata)

    popt_exponential, _ = curve_fit(default_latency_predictor_decode, xdata, ydata, maxfev=15000)
    print(f'Decode. Learnt constants for model {title}: {popt_exponential}')

    for output_length, x_line, y_line in processed_results_output:
        axs[index_x].plot(
            x_line,
            default_latency_predictor_decode(
                np.asarray([x_line]),
                popt_exponential[0],
                popt_exponential[1]
            ),
            marker='o',
            linestyle='dotted',
            color=color[output_length]
        )

    axs[index_x].set_ylabel('average inter-token latency (ms)', fontsize=10)
    axs[index_x].set_xlabel('average batch size (toks)', fontsize=10)
    axs[index_x].legend(handles=legend_elements, fontsize=10, loc='upper right')
    axs[index_x].set_title('Decode')

    legend_elements = []
    legend_elements.append(Line2D([], [], marker='o', linestyle='solid', color='gray', label='Real results'))
    legend_elements.append(Line2D([], [], marker='o', linestyle='dotted', color='gray', label='Predicted results'))
    fig.legend(handles=legend_elements, fontsize=10, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))

    plt.savefig(os.path.join(path, f'predictor_{title}'.replace('.', '')), bbox_inches='tight')


def main():
    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        total_duration_input, results_input = extract_results(f'{model}/input')
        total_duration_output, results_output = extract_results(f'{model}/output')

        plot_relationship(
            results_input,
            results_output,
            '',
            model
        )

        plot_predictor(
            results_input,
            results_output,
            '',
            model
        )

        print(f'Total duration by model {model}: {total_duration_input + total_duration_output}')


if __name__ == '__main__':
    main()
