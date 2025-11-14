from __future__ import print_function
import argparse
import json
import os.path
import random
import subprocess
import sys
from copy import deepcopy
from typing import List
import os
import re

# path to root code directory in host and container
EXP_HOME_CODE_DIR = '.'

# path to Slurm executable
EXP_SLURM_EXECUTABLE = 'benchmarks/lora/deployment/slurm/slurm_simulator.sh'

# path to container image
EXP_CONTAINER_IMAGE = '/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/images/vllm_0.8.5.sif'


def schedule_job(
        user: str,
        queue: str,
        specific_name: str,
        default_env_vars: dict,
        output_path: str,
        exp_run_command: str,
        exp_max_duration: str,
        exclusive: bool,
        slurm_executable: str,
        no_effect: bool,
) -> None:
    global EXP_HOME_CODE_DIR, EXP_SLURM_EXECUTABLE, EXP_CONTAINER_IMAGE

    exp_output_path = os.path.join(output_path)
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
        arguments: str,
        results_path: str,
        output_path: str,
        default_env_vars: dict,
        exp_max_duration: str,
        exclusive: bool,
        slurm_executable: str,
        no_effect: bool,
        test: bool
) -> None:
    if \
            '--debug-path' in arguments \
            or '--timeout' in arguments \
            or '--print-outcome' in arguments \
            or '--experiment-path' in arguments \
            or '--output-path' in arguments:
        raise ValueError('Input arguments are not allowed')

    dirs_with_results = set()
    for dirpath, dirnames, filenames in os.walk(results_path):
        for filename in filenames:
            if filename.startswith('openai-') and filename.endswith('.json'):
                dirs_with_results.add(dirpath)
                break  # No need to check more files in this directory

    for index, directory_with_results in enumerate(dirs_with_results):
        if test and index > 0:
            break

        run_name = os.path.basename(directory_with_results)
        exp_output_path = os.path.join(output_path, directory_with_results.replace(results_path, '')[1:])
        run_command = f'' \
                      f'python3 benchmarks/lora/simulation_pipeline.py' \
                      f' --experiment-path {directory_with_results}' \
                      f' --output-path {exp_output_path}' \
                      f' --print-outcome'
        run_command += ' ' + arguments

        schedule_job(
            user,
            queue,
            run_name,
            default_env_vars,
            exp_output_path,
            run_command,
            exp_max_duration,
            exclusive,
            slurm_executable,
            no_effect
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher of vllm benchmarking experiments on Kubernetes')
    parser.add_argument('--user', type=str, help='Slurm user', required=True)
    parser.add_argument('--queue', type=str, help='Slurm queue', required=True)
    parser.add_argument('--arguments', type=str, default='', help='Arguments to use in simulation', required=False)
    parser.add_argument('--results-path', type=str, help='Path to collect what to simulate', required=True)
    parser.add_argument('--output-path', type=str, help='Path to store results', required=True)
    parser.add_argument('--exclusive', action='store_true', default=False, help='Run the experiments in exclusive mode')
    parser.add_argument('--max-duration', type=str, default='00:05:00', help='Slurm queue')
    parser.add_argument('--slurm-executable', type=str, default=EXP_SLURM_EXECUTABLE, help='Define the slurm script to run')
    parser.add_argument('--no-effect', action='store_true', default=False, help='Do everything except the step of launching the experiment')
    parser.add_argument('--test', action='store_true', default=False, help='Run just one')
    parser.add_argument('--default-env-vars', type=str, help='Dictionary with the default env vars')
    args = parser.parse_args()

    if args.default_env_vars is not None:
        default_env_vars = json.loads(args.default_env_vars.replace('\'', '"'))
    else:
        default_env_vars = {}

    os.makedirs(args.output_path, exist_ok=True)
    config_path = os.path.join(args.output_path, f'config-{str(random.randint(0, 100000))}.txt')
    with open(config_path, 'w') as config_file:
        config = 'PYTHONPATH=. python3 ' + ' '.join(sys.argv).replace('{', '"{').replace('}', '}"') + '\n'
        config_file.write(config)

    main(
        args.user,
        args.queue,
        args.arguments.replace('\'', ''),
        args.results_path,
        args.output_path,
        default_env_vars,
        args.max_duration,
        args.exclusive,
        args.slurm_executable,
        args.no_effect,
        args.test
    )
