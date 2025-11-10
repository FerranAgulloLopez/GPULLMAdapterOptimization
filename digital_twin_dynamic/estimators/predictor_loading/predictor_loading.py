import glob
import json
import os
import re
from typing import List, Tuple, Dict, Set, Any


CONSTANTS_PER_MODEL = {  # model - max_lora_rank - adapter_rank
    'llama-3.1-8b-instruct': {
        'constant_1': {
            8: {
                8: 6.612409031949937
            },
            16: {
                8: 7.772141913883388,
                16: 6.383920940570533
            },
            32: {
                8: 8.076073373667896,
                16: 7.868956197053195,
                32: 6.367266979068518
            }
        }
    },
    'qwen-2.5-7b-instruct': {
        'constant_1': {
            8: {
                8: 3.060487743932754
            },
            16: {
                8: 3.3986456133425236,
                16: 2.9900502949021757
            },
            32: {
                8: 3.3807663433253765,
                16: 3.506520129740238,
                32: 3.0435188487172127
            }
        }
    }
}


def loading_time_predictor(
        x,
        constant_1=CONSTANTS_PER_MODEL['llama-3.1-8b-instruct']
):
    max_lora_rank = x[0]
    adapter_rank = x[1]
    return constant_1[max_lora_rank][adapter_rank]


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


def extract_results(path: str, adapter_rank: int) -> Tuple[float, List[Dict[str, Any]]]:
    def create_id(metrics: Dict[str, float], id_metrics: List[str]) -> str:
        _id: str = ''
        for metric_key in id_metrics:
            _id += f'{metrics[metric_key]}_'
        return _id

    def extract_experiment_metric(path: str) -> Dict[str, Any]:
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

        # extract duration
        output['duration']: float = float(metrics['duration'])

        # extract num prompts
        num_prompts: float = float(metrics['num_prompts'])

        # extract max cpu loras
        pattern = r'max_cpu_loras=(\d+)'
        found = re.findall(pattern, log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        max_cpu_loras = int(found)

        # extract disk loads
        pattern = r'Total loads from disk: (\d+). Total loading time from disk: ((\d+\.(\d+)?)(e-\d+)?) seconds'
        data = re.findall(pattern, log)
        if data is None:
            raise ValueError(f'Metric pattern not found on result log')
        number_disk_loads: int = len(data)
        assert number_disk_loads == max_cpu_loras

        # extract memory loads
        pattern = r'Total loads from memory: (\d+). Total loading time from memory: ((\d+\.(\d+)?)(e-\d+)?) seconds'
        data = re.findall(pattern, log)
        if data is None:
            raise ValueError(f'Metric pattern not found on result log')
        number_memory_loads: int = len(data)
        assert number_memory_loads == (number_disk_loads + num_prompts)

        # extract activation times
        activation_times: List[float] = [float(item[1]) for item in data]

        # extract total activation time minus init
        total_activation_time: float = activation_times[number_memory_loads - 1] - activation_times[number_disk_loads - 1]

        # extract mean activation time
        output['mean_activation_time'] = total_activation_time / num_prompts

        return output

    total_duration: float = 0
    collected_ids: Set[str] = set()
    results = []
    errors = 0
    id_metrics = ['adapter_rank', 'max_lora_rank']
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            max_lora_rank: int = int(folder.replace('_', ''))

            try:
                metrics = extract_experiment_metric(os.path.join(path, folder))
                total_duration += metrics['duration']
            except ValueError as e:
                print(os.path.join(path, folder), e)
                errors += 1
                metrics = {}
            metrics['adapter_rank'] = adapter_rank
            metrics['max_lora_rank'] = max_lora_rank

            _id = create_id(metrics, id_metrics)
            if _id in collected_ids:
                raise ValueError('Repeated results')
            collected_ids.add(_id)
            results.append(metrics)
    print(f'Unknown extraction errors: {errors}. Should be zero.')
    return total_duration, results


def plot_relationship(
        results: List[Dict[str, Any]],
        path: str,
        title: str,
) -> None:
    color_by_adapter_rank = {}

    # plot
    nrows = 1
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    for item in results:
        mean_activation_time: float = item['mean_activation_time'] * 1000  # from seconds to ms
        adapter_rank: int = item['adapter_rank']
        max_lora_rank: int = item['max_lora_rank']

        if adapter_rank in color_by_adapter_rank:
            axs.plot(
                max_lora_rank,
                mean_activation_time,
                marker='o',
                label=f'adapter rank {adapter_rank}',
                color=color_by_adapter_rank[adapter_rank]
            )
        else:
            line = axs.plot(
                max_lora_rank,
                mean_activation_time,
                label=f'adapter rank {adapter_rank}',
                marker='o'
            )[0]
            color_by_adapter_rank[adapter_rank] = line.get_color()

    axs.set_xlabel('Max LoRA rank', fontsize=10)
    axs.set_ylabel('average loading CPU->GPU (ms)', fontsize=10)
    legend_elements = [Line2D([], [], color=color, marker='o', linestyle='None', label=f'adapter rank {adapter_rank}') for adapter_rank, color in color_by_adapter_rank.items()]
    fig.legend(handles=legend_elements, fontsize=10, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.0))
    plt.savefig(os.path.join(path, f'relationship_{title}'.replace('.', '-')), bbox_inches='tight')


def main():
    for model in ['llama-3.1-8b-instruct', 'qwen-2.5-7b-instruct']:
        total_duration: float = 0
        all_results = []

        duration, results = extract_results(f'{model}/adapter_rank_8', 8)
        all_results += results
        total_duration += duration

        duration, results = extract_results(f'{model}/adapter_rank_16', 16)
        all_results += results
        total_duration += duration

        duration, results = extract_results(f'{model}/adapter_rank_32', 32)
        all_results += results
        total_duration += duration

        plot_relationship(
            all_results,
            '',
            model
        )

        final_results: Dict[int, Dict[int, float]] = {}
        for item in all_results:
            mean_activation_time: float = item['mean_activation_time'] * 1000  # from seconds to ms
            adapter_rank: int = item['adapter_rank']
            max_lora_rank: int = item['max_lora_rank']
            if max_lora_rank not in final_results:
                final_results[max_lora_rank] = {}
            if adapter_rank in final_results[max_lora_rank]:
                raise ValueError('Repeated results')
            final_results[max_lora_rank][adapter_rank] = mean_activation_time

        print(f'Final results for {model}')
        print(json.dumps(final_results, indent=4))
        print(f'Total duration by model {model}: {total_duration}')
        with open(os.path.join('', f'{model}_constants.json'), 'w') as file:
            json.dump(final_results, file, indent=4)


if __name__ == '__main__':
    # visualization imports here, to not run them during DT execution
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    main()
