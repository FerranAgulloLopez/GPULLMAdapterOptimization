# A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving
_This repository was created through a fork of the [vLLM repository](https://github.com/vllm-project/vllm) as a result of the workshop manuscript [A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving](https://arxiv.org/abs/2508.08343)_

### Introduction
This repository includes the code modifications made to the original vLLM repository to support the experiments presented in the mentioned manuscript. It also contains all the results and code used to generate the manuscript's tables and figures.

The sections below detail:
- The specific code changes
- The required steps for setting up the environment
- The required steps to run the experiments
- The results of the manuscript and how to create the tables and charts

### Code changes
This repository includes modifications both to the benchmarking and server components of the vLLM serving system. Our main changes in these two components are the following:
- Benchmark component: 
  - An updated benchmark script, _benchmarks/benchmark_serving.py_, on top of the original online benchmark script that includes:
    - Adapter management: It is able to collect the adapters deployed in the server and send requests to them.
    - Metrics logging: It collects the metrics from the vLLM Prometheus endpoint. This is handled in _benchmarks/concurrent_metrics_checker.py_ and can be disabled using the `--disable-log-stats` flag.
    - Integrated server launching: It can launch the server automatically using the `--launch-server` flag. Server arguments can be passed with `--server-args`. This simplifies the process of running a benchmark from a single entry point.
  - An updated benchmark script, _benchmarks/benchmark_serving_by_time.py_, which is build on top of the previous script but with the following changes:
    - Fixed time window: The script behaviour is changed to reproduce a time window of a client-server paradigm, instead of just sending a fixed number of requests that always creates an unrealistic scenario at the end of the exeuction.
    - More adapter management: Allows more management of the rates and sizes of the adapters being deployed and served.
- Server component
  - HPC Slurm deployment: We included the required steps and data to deploy and run the server and benchmark with Slurm in HPC environments (singularity images) in the directory _benchmarks/deployment/slurm_.
  - Batch running: It includes the script _benchmarks/deployment/slurm/launcher_ that acts as a launcher to run multiple experiments with different configurations transparent from Slurm.
  - More logging: The server code is updated to provide more logging, like the scheduler time by step or the loading time of adapters. 
  - Easier usage: The server code is updated to includes new input parameters for easier usage, like `--dummy-lora-modules` to serve a desired set of adapter replicas.

We also include all the code for the proposed Digital Twin:
- The main core of the component can be found under _digital_twin_dynamic/_
- The predictive performance models can be found under _digital_twin_dynamic/estimators/_
- The required steps to run and deploy it with Slurm in HPC environments (singularity images) are available in the directory _benchmarks/deployment/slurm_.

We also made a few minor updates to the _.gitignore_ file, adding new rules to prevent certain newly generated output files from being tracked, and removing specific rules to allow the inclusion of selected result files that we wanted to upload to the repository.

### How to set up
We show how to reproduce the experiments that appear in the paper, where we use Singularity and Slurm, which must be installed prior to execution. Nevertheless, as the original vLLM code, everything can also be run with docker or plainly with Python. Follow the required steps:
- Create the base Docker image as the foundation for the Singularity image: `VLLM_INSTALL_PUNICA_KERNELS=1 docker build -f docker/Dockerfile --target vllm-openai -t vllm .`
- If having problems with the wheel size, you can reduce the amount of supported through build arg `torch_cuda_arch_list` or disable the wheel check through build arg `RUN_WHEEL_CHECK` (or just comment it in the Dockerfile directly)
- Create the Singularity image. If willing to use both online and offline modes, it is recommended to use the definition file found in the _benchmarks_simple_ deployment directory as appears in the following commands: `sudo singularity build vllm.sif benchmarks_simple/deployment/SingularityBenchmark.def`
- If having problems with memory errors, you can include swap space like following:
```
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### How to run
Once the Singularity image has been built, you can easily reproduce the experiments from the paper using the commands provided in the .txt configuration files. For every run set of experiments we include a config*.txt file with the command that was used ot run it (it is created automatically). These commands invoke the launcher that does all effort in running the desired experiments in the Slurm cluster transparent to the user.

For instance, the directory _benchmarks/lora/definitive_results/ml_modelling/test_dataset/llama-3.1-8b-instruct/rate_0.1-0.00625-0.003125_size_16-16-16/_ contains the file _config-61127.txt_ with the ready-to-run command that was used to run some of the Llama-3.1-8B experiments to validate the ML modeling. Simply execute it from the root directory of the repository, and the corresponding experiments will be submitted to the Slurm queue for execution.

In this example we have the following:
```
PYTHONPATH=. python3 benchmarks/lora/deployment/slurm/launcher.py \
 --user bsc98 \
 --queue acc_bsccs  \
 --max-duration 01:45:00 \
 --results-path benchmarks/lora/definitive_results/finding_maximum/test_dataset/with_offloading/llama-3.1-8b-instruct/rate_0.1-0.00625-0.003125_size_16-16-16 \
 --default-server-args "{'--disable-log-requests': '', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-3.1-8b-instruct', '--enable-lora': '', '--dummy-lora-modules': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-3.1-8b-instruct/lora/lora-finance_dummy_rank_16/ /gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-3.1-8b-instruct/lora/lora-finance_dummy_rank_16/ /gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-3.1-8b-instruct/lora/lora-finance_dummy_rank_16/', '--max-num-seqs': '2048', '--scheduler-cls': 'vllm.v1.core.sched.scheduler.Scheduler', '--no-enable-prefix-caching': ''}" \ 
 --default-benchmark-args "{'--backend': 'openai', '--dataset-path': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/ShareGPT_V3_unfiltered_cleaned_split.json', '--endpoint': '/v1/completions', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-3.1-8b-instruct', '--save-result': '', '--total-time': '3600', '--adapter-rates': '0.1 0.00625 0.003125', '--server-init-time': '1800'}" \
 --test-server-args "{'--max-loras': ['16', '32', '64', '96', '128', '160', '192', '256', '320', '384'], '--max-cpu-loras': ['16', '32', '64', '96', '128', '160', '192', '256', '320', '384'], '--max-lora-rank': ['16']}" \
 --test-benchmark-args "{}" \
 --benchmark-executable benchmarks/lora/benchmark_serving_by_time.py \
 --check-condition-cpu-loras \
```

As shown, the command invokes the launcher for the _benchmark_serving_by_time_. While the full list of arguments can be found in the corresponding launcher script, here is a summary of the key components defined in this example:

- `PYTHONPATH`: Set to the root of the repository to define the Python working directory.
- `--user`: Specifies the Slurm user account.
- `--queue`: Specifies the Slurm queue or partition.
- `--max-duration`: Sets the maximum allowed runtime for the Slurm job.
- `--results-path`: Path where all experiment results will be saved.
- `--default-server-args`: Arguments passed to the vLLM server for all experiments.
- `--default-benchmark-args`: Common benchmark arguments used across all experiments.
- `--test-server-args`: Server arguments that will vary across different experiments.
- `--test-benchmark-args`: Benchmark arguments that will vary across different experiments.

The launcher will automatically initiate an experiment for every combination of server and benchmark arguments defined via the `--test-server-args` and `--test-benchmark-args` flags. Each of these experiments will also include the default arguments specified by the `--default-server-args` and `--default-benchmark-args` flags. Every experiment is executed through the appropriate benchmark script, which also handles launching of the vLLM server.

Take into account that these commands use input arguments for defining the locations of the datasets and models between others. They will need to be change for them to work properly.

### Manuscript results
All results of the manuscript appear in the directory _benchmarks/lora/definitive_results_ divided by figures or tables from the manuscript. In every subfolder there appear the outcome of the run experiments with Slurm, along a chart.py python script that reads these experiment outcomes and produces the corresponding figure or table.

Specifically, we have the following:
- performance_analysis
  - memory_overhead: Figure 2
  - compute_overhead: Figure 3
  - loading_overhead: Figure 4
- motivation_figure: Figure 1
- digital_twin_modelling: Table 1 and Figure 7
- ml_modelling: Table 1, Table 2 and Figure 8

### How to cite
If using these code modifications please cite this paper:
```
Agulló, F., Oliveras, J., Wang, C., Gutiérrez-Torre, A., Tardieu, O., Youssef, A., Torres, J., & Berral, J. Ll. (2025). A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving. In NeurIPS 2025 Machine Learning Systems (MLSys) Workshop.
```