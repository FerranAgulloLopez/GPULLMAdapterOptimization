from __future__ import print_function
import argparse
import json
import os.path
import random
import subprocess
import sys
from copy import deepcopy
from itertools import combinations, combinations_with_replacement
from typing import List, Dict, Set, Any
import os
import glob
import re

# path to root code directory in host and container
EXP_HOME_CODE_DIR = '.'

# path to Slurm executable
EXP_SLURM_EXECUTABLE = 'benchmarks/lora/deployment/slurm/slurm_simulator.sh'

# path to container image
EXP_CONTAINER_IMAGE = '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/images/vllm_0.8.5.sif'


def get_gpu_memory_availability(path: str) -> Dict[str, Dict[int, Dict[int, float]]]:  # model, adapter slots, adapter size
    dirs_with_results: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.startswith('openai-') and filename.endswith('.json'):
                dirs_with_results.add(dirpath)
                break  # No need to check more files in this directory

    gpu_memory_availability: Dict[str, Dict[int, Dict[int, float]]] = {}
    for dir_with_results in dirs_with_results:
        # load server log
        filenames: List[str] = glob.glob(os.path.join(dir_with_results, 'server_out.log'))
        if len(filenames) != 1:
            raise ValueError(f'More than one output result file or none {filenames} for path {dir_with_results}')
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

        # extract adapter size
        pattern = r'max_lora_rank=(\d+)'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        adapter_size = int(found)

        # extract max token capacity
        pattern = r'GPU KV cache size: ([0-9]+([,][0-9]*)?|[.][0-9]+) tokens'
        found = re.findall(pattern, server_log)[-1]
        if found is None:
            raise ValueError(f'Metric pattern not found on result log')
        available_GPU_memory = int(found[0].replace(',', ''))

        # include data in output dict
        if model not in gpu_memory_availability:
            gpu_memory_availability[model] = {}
        if adapter_slots not in gpu_memory_availability[model]:
            gpu_memory_availability[model][adapter_slots] = {}
        if adapter_size in gpu_memory_availability[model][adapter_slots]:
            raise ValueError('Repeated results')
        gpu_memory_availability[model][adapter_slots][adapter_size] = available_GPU_memory

    return gpu_memory_availability


def schedule_job(
        user: str,
        queue: str,
        specific_name: str,
        default_env_vars: dict,
        exp_output_path: str,
        exp_run_command: str,
        exp_max_duration: str,
        exclusive: bool,
        slurm_executable: str,
        no_effect: bool
) -> None:
    global EXP_HOME_CODE_DIR, EXP_SLURM_EXECUTABLE, EXP_CONTAINER_IMAGE

    os.makedirs(exp_output_path, exist_ok=True)

    env = os.environ.copy()
    env['EXP_NAME'] = specific_name
    env['EXP_MAX_DURATION_SECONDS'] = exp_max_duration
    env['EXP_OUTPUT_PATH'] = exp_output_path
    env['EXP_RUN_COMMAND'] = exp_run_command
    env['EXP_CONTAINER_IMAGE'] = EXP_CONTAINER_IMAGE

    # running env vars
    str_env_vars: str = ''
    default_env_vars['PYTHONPATH'] = os.path.abspath(EXP_HOME_CODE_DIR)
    default_env_vars['TOKENIZERS_PARALLELISM'] = 'false'
    env['EXP_ENV_VARS'] = str_env_vars

    if not no_effect:
        command = f'cat {slurm_executable} | envsubst > {exp_output_path}/launcher.sh'
        subprocess.run(command, env=env, shell=True)

        if exclusive:
            command = f'sbatch -A {user} -q {queue} --exclusive {exp_output_path}/launcher.sh'
        else:
            command = f'sbatch -A {user} -q {queue} {exp_output_path}/launcher.sh'

        subprocess.run(command, shell=True)
    else:
        print(exp_run_command)


def main(
        user: str,
        queue: str,
        multiple_running: int,
        model: str,
        dataset_path: str,
        default_arguments: Dict[str, Any],
        adapters_list: List[int],
        adapter_rates_list: List[float],
        adapter_sizes_list: List[int],
        max_num_batched_tokens: int,
        seconds_to_simulate: float,
        gpu_memory_availability_exps: str,
        output_path: str,
        default_env_vars: dict,
        exp_max_duration: str,
        exclusive: bool,
        slurm_executable: str,
        no_effect: bool
) -> None:
    model_id: str = os.path.basename(model)
    if 'without_offloading' not in default_arguments:
        raise ValueError('Missing mandatory default argument')  # TODO refactor
    without_offloading: bool = default_arguments['without_offloading']

    gpu_memory_availability: Dict[str, Dict[int, Dict[int, float]]] = get_gpu_memory_availability(gpu_memory_availability_exps)

    all_combinations_number: int = 0
    dataset_combinations: Set[str] = set()
    pending_paths_to_run: List[str] = []
    runner_id: int = 0
    for index_served_adapters, served_adapters in enumerate(adapters_list):

        if without_offloading:
            possible_adapter_slots: List[int] = [served_adapters]
        else:
            possible_adapter_slots: List[int] = [adapters_list[index_adapter_slots] for index_adapter_slots in range(0, index_served_adapters + 1)]

        for adapter_slots in possible_adapter_slots:

            for adapter_rate_combination in combinations(adapter_rates_list, 3):
                adapter_rate_combination: str = ' '.join(str(item) for item in adapter_rate_combination)

                for adapter_size_combination in combinations_with_replacement(adapter_sizes_list, 3):

                    # check GPU memory constraint
                    correct_combination: bool = True
                    max_adapter_size: int = max(adapter_size_combination)
                    if model_id not in gpu_memory_availability:
                        print('Warn!! Missing model')
                        correct_combination = False
                    elif adapter_slots not in gpu_memory_availability[model_id]:
                        correct_combination = False
                    elif max_adapter_size not in gpu_memory_availability[model_id][adapter_slots]:
                        correct_combination = False

                    # remove bias towards 32 rank (which reduces max GPU memory even if only one is present)
                    number_of_32s: int = 0
                    for size in adapter_size_combination:
                        if size == 32:
                            number_of_32s += 1
                    if number_of_32s != 3:
                        correct_combination = False

                    adapter_size_combination: str = ' '.join(str(item) for item in adapter_size_combination)

                    # run
                    if correct_combination:
                        # determine name and output path
                        dir_name: str = f'' \
                                        f'adapters_{served_adapters}' \
                                        f'_slots_{adapter_slots}' \
                                        f'_rate_{adapter_rate_combination.replace(" ", "-")}' \
                                        f'_size_{adapter_size_combination.replace(" ", "-")}'
                        exp_output_path: str = os.path.join(
                            output_path,
                            'results',
                            f'rate_{adapter_rate_combination.replace(" ", "-")}_size_{adapter_size_combination.replace(" ", "-")}/',
                            dir_name
                        )

                        # count
                        all_combinations_number += 1
                        dataset_combinations.add(f'{adapter_rate_combination}_{adapter_size_combination}')

                        # define arguments
                        arguments: Dict[str, Any] = {
                            'total_time': seconds_to_simulate,
                            'model': model,
                            'adapter_slots': adapter_slots,
                            'served_adapters': served_adapters,
                            'served_adapters_rates': [float(item) for item in adapter_rate_combination.split(' ')],
                            'served_adapters_sizes': [int(item) for item in adapter_size_combination.split(' ')],
                            'available_gpu_memory': gpu_memory_availability[model_id][adapter_slots][max_adapter_size],
                            'max_num_batched_tokens': max_num_batched_tokens,
                            'dataset_path': dataset_path,
                            'output_path': exp_output_path,
                            'print_outcome': True
                        }

                        # include default arguments
                        for default_argument_key, default_argument_value in default_arguments.items():
                            if default_argument_key in arguments:
                                raise ValueError('Illegal default argument')
                            arguments[default_argument_key] = default_argument_value

                        # create output directory
                        os.makedirs(exp_output_path, exist_ok=True)

                        # save arguments
                        with open(os.path.join(exp_output_path, 'arguments.json'), 'w') as file:
                            json.dump(arguments, file, indent=4)

                        # add to pending paths to run
                        pending_paths_to_run.append(exp_output_path)

                        # schedule if needed
                        if len(pending_paths_to_run) >= multiple_running:
                            # define run name
                            run_name: str = f'{runner_id}_running_{len(pending_paths_to_run)}'
                            runner_id += 1

                            # define run command
                            run_command: str = f'python3 benchmarks/lora/simulation_pipeline_from_scratch_multiple.py --paths \''
                            for index_path, path in enumerate(pending_paths_to_run):
                                if ' ' in path:
                                    raise Exception('Path must not include spaces')
                                if index_path == 0:
                                    run_command += path
                                else:
                                    run_command += f' {path}'
                            run_command += '\''
                            pending_paths_to_run = []

                            # define runner output path
                            runner_output_path: str = os.path.join(
                                output_path,
                                'runners',
                                run_name
                            )

                            # schedule
                            schedule_job(
                                user,
                                queue,
                                run_name,
                                default_env_vars,
                                runner_output_path,
                                run_command,
                                exp_max_duration,
                                exclusive,
                                slurm_executable,
                                no_effect
                            )
    # schedule if still some pending
    if len(pending_paths_to_run) > 0:
        # define run name
        run_name: str = f'{runner_id}_running_{len(pending_paths_to_run)}'
        runner_id += 1

        # define run command
        run_command: str = f'python3 benchmarks/lora/simulation_pipeline_from_scratch_multiple.py --paths \''
        for index_path, path in enumerate(pending_paths_to_run):
            if ' ' in path:
                raise Exception('Path must not include spaces')
            if index_path == 0:
                run_command += path
            else:
                run_command += f' {path}'
        run_command += '\''
        pending_paths_to_run = []

        # define runner output path
        runner_output_path: str = os.path.join(
            output_path,
            'runners',
            run_name
        )

        # schedule
        schedule_job(
            user,
            queue,
            run_name,
            default_env_vars,
            runner_output_path,
            run_command,
            exp_max_duration,
            exclusive,
            slurm_executable,
            no_effect
        )
    print('GPU memory availability', gpu_memory_availability)
    print('All combinations', all_combinations_number)
    print('Dataset combinations', len(dataset_combinations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher of vllm benchmarking experiments on Kubernetes')
    parser.add_argument('--user', type=str, help='Slurm user', required=True)
    parser.add_argument('--queue', type=str, help='Slurm queue', required=True)
    parser.add_argument('--multiple-running', type=int, default=1, help='Every job runs the desired number of simulations', required=False)
    parser.add_argument('--max-duration', type=str, default='00:05:00', help='Slurm queue')
    parser.add_argument('--model', type=str,  help='Model path to use in simulation', required=True)
    parser.add_argument('--dataset-path', type=str, help='Dataset to use in simulation', required=True)
    parser.add_argument('--adapters', type=str, help='List of adapters to use.', required=True)
    parser.add_argument('--adapter-rates', type=str, help='Adapter rates to choose from.', required=True)
    parser.add_argument('--adapter-sizes', type=str, help='Adapter sizes to choose from.', required=True)
    parser.add_argument('--seconds-to-simulate', type=float, help='Seconds to simulate', required=True)
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048, help='Max num batched tokens', required=False)
    parser.add_argument('--gpu-memory-availability-exps', type=str, help='Path to GPU memory availability exps', required=True)
    parser.add_argument('--output-path', type=str, help='Path to store results', required=True)
    parser.add_argument('--default-arguments', type=str, default='', help='Dictionary with simulation default arguments', required=False)
    parser.add_argument('--exclusive', action='store_true', default=False, help='Run the experiments in exclusive mode')
    parser.add_argument('--slurm-executable', type=str, default=EXP_SLURM_EXECUTABLE, help='Define the slurm script to run')
    parser.add_argument('--no-effect', action='store_true', default=False, help='Do everything except the step of launching the experiment')
    parser.add_argument('--default-env-vars', type=str, help='Dictionary with the default env vars')
    args = parser.parse_args()

    default_arguments = json.loads(args.default_arguments.replace('\'', '"'))
    if args.default_env_vars is not None:
        default_env_vars = json.loads(args.default_env_vars.replace('\'', '"'))
    else:
        default_env_vars = {}

    args.adapters = [int(item) for item in args.adapters.split(' ')]
    args.adapter_rates = [float(item) for item in args.adapter_rates.split(' ')]
    args.adapter_sizes = [int(item) for item in args.adapter_sizes.split(' ')]

    os.makedirs(args.output_path, exist_ok=True)
    config_path = os.path.join(args.output_path, f'config-{str(random.randint(0, 100000))}.txt')
    with open(config_path, 'w') as config_file:
        config = 'PYTHONPATH=. python3 ' + ' '.join(sys.argv).replace('{', '"{').replace('}', '}"') + '\n'
        config_file.write(config)

    main(
        user=args.user,
        queue=args.queue,
        multiple_running=args.multiple_running,
        model=args.model,
        dataset_path=args.dataset_path,
        default_arguments=default_arguments,
        adapters_list=args.adapters,
        adapter_rates_list=args.adapter_rates,
        adapter_sizes_list=args.adapter_sizes,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seconds_to_simulate=args.seconds_to_simulate,
        gpu_memory_availability_exps=args.gpu_memory_availability_exps,
        output_path=args.output_path,
        default_env_vars=default_env_vars,
        exp_max_duration=args.max_duration,
        exclusive=args.exclusive,
        slurm_executable=args.slurm_executable,
        no_effect=args.no_effect
    )
