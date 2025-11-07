from __future__ import print_function

import argparse
import os
import os.path
import random
import subprocess
import sys

# path to root code directory in host
EXP_HOME_CODE_DIR = '.'

# path to Slurm executable
EXP_SLURM_EXECUTABLE = 'benchmarks/lora/deployment/slurm/slurm_ml_training.sh'

# path to script
EXP_RUNNING_SCRIPT = 'benchmarks/lora/train_ml.py'

# conda environment
EXP_CONDA_ENV = 'MLTrain'


def schedule_job(
        user: str,
        queue: str,
        specific_name: str,
        output_path: str,
        exp_run_command: str,
        exp_max_duration: str,
        exclusive: bool,
        no_effect: bool,
) -> None:
    global EXP_SLURM_EXECUTABLE, EXP_CONDA_ENV

    exp_output_path = os.path.join(output_path)
    os.makedirs(exp_output_path, exist_ok=True)

    env = os.environ.copy()
    env['EXP_NAME'] = specific_name
    env['EXP_MAX_DURATION_SECONDS'] = exp_max_duration
    env['EXP_OUTPUT_PATH'] = exp_output_path
    env['EXP_RUN_COMMAND'] = exp_run_command
    env['EXP_CONDA_ENV'] = EXP_CONDA_ENV

    if not no_effect:
        command = f'cat {EXP_SLURM_EXECUTABLE} | envsubst > {exp_output_path}/launcher.sh'
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
        output_path: str,
        arguments: str,
        exp_max_duration: str,
        exclusive: bool,
        no_effect: bool,
) -> None:
    global EXP_HOME_CODE_DIR, EXP_RUNNING_SCRIPT

    run_name = os.path.basename(output_path)
    run_command = f'PYTHONPATH={os.path.abspath(EXP_HOME_CODE_DIR)}' \
                  f' python3 {os.path.abspath(EXP_RUNNING_SCRIPT)}' \
                  f' --output-path {output_path}'
    run_command += ' ' + arguments

    schedule_job(
        user=user,
        queue=queue,
        specific_name=run_name,
        output_path=output_path,
        exp_run_command=run_command,
        exp_max_duration=exp_max_duration,
        exclusive=exclusive,
        no_effect=no_effect,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher of vllm benchmarking experiments on Kubernetes')
    parser.add_argument('--user', type=str, help='Slurm user', required=True)
    parser.add_argument('--queue', type=str, help='Slurm queue', required=True)
    parser.add_argument('--output-path', type=str, help='Path to store results', required=True)
    parser.add_argument('--exclusive', action='store_true', default=False, help='Run the experiments in exclusive mode')
    parser.add_argument('--max-duration', type=str, default='00:05:00', help='Slurm queue')
    parser.add_argument('--no-effect', action='store_true', default=False, help='Do everything except the step of launching the experiment')
    args, unknown = parser.parse_known_args()

    os.makedirs(args.output_path, exist_ok=False)
    config_path = os.path.join(args.output_path, f'config-{str(random.randint(0, 100000))}.txt')
    with open(config_path, 'w') as config_file:
        config = 'PYTHONPATH=. python3 ' + ' '.join(sys.argv) + '\n'
        config_file.write(config)

    main(
        user=args.user,
        queue=args.queue,
        output_path=args.output_path,
        arguments=' '.join([f'\'{item}\'' if item[0] != '-' else item for item in unknown]),
        exp_max_duration=args.max_duration,
        exclusive=args.exclusive,
        no_effect=args.no_effect,
    )
