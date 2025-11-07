- Use vLLM default repository, code changes from modified version are mounted when running the container
- Create base docker image: `DOCKER_BUILDKIT=1 VLLM_INSTALL_PUNICA_KERNELS=1 docker build --build-arg VLLM_MAX_SIZE_MB=800 --build-arg TORCH_CUDA_ARCH_LIST='8.9 9.0+PTX' -f docker/Dockerfile -t vllm . --target vllm-openai`
- Create singularity image with additional requirements to run benchmark: `sudo singularity build vllm-benchmark.sif benchmarks/lora/deployment/slurm/SingularityBenchmark.def`
- Schedule different benchmarks e.g., `PYTHONPATH=. python3 benchmarks/lora/deployment/slurm/launcher.py --user bsc98 --queue acc_debug --results-path /home/bsc/bsc098069/llm_benchmarking/results/ --default-server-args "{'--disable-log-stats': '', '--disable-log-requests': '', '--model': '/home/bsc/bsc098069/llm_benchmarking/models/llama/llama-2-7b'}" --default-benchmark-args "{'--disable-log-stats': '', '--backend': 'openai', '--dataset': '/home/bsc/bsc098069/llm_benchmarking/data/ShareGPT_V3_unfiltered_cleaned_split.json', '--endpoint': '/v1/completions', '--model': '/home/bsc/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--save-result':''}" --test-server-args "{}" --test-benchmark-args "{'--num-prompts': ['10']}"`

We show how to reproduce the experiments that appear in the paper, where we use Singularity and Slurm, which must be installed prior to execution. Nevertheless, as the original vLLM code, everything can also be run with docker or plainly with Python. Follow the required steps:
- Create the base Docker image as the foundation for the Singularity image: `docker build -f docker/Dockerfile --target vllm-openai -t vllm .`
- Optionally specifies: `--build-arg max_jobs=8 --build-arg nvcc_threads=2`. Check official documentation at https://docs.vllm.ai/en/stable/deployment/docker.html#building-vllms-docker-image-from-source
- If having problems with the wheel size, you can reduce the amount of supported through build arg `torch_cuda_arch_list` or disable the wheel check through build arg `RUN_WHEEL_CHECK` (or just comment it in the Dockerfile directly)
- Create the Singularity image. If willing to use both online and offline modes, it is recommended to use the definition file found in the _benchmarks_simple_ deployment directory as appears in the following commands:
  - Default: `sudo singularity build vllm.sif benchmarks_simple/deployment/SingularityBenchmark.def`
  - With Nsight: `sudo singularity build vllm-nsight.sif benchmarks_simple/deployment/SingularityBenchmark_nsight.def`
- If having problems with memory errors, you can include swap space like following:
```
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```
- [OPT] If willing to use Nvidia Nsight Systems / Compute or MPS, the desired technologies should be installed as well prior to execution in the corresponding nodes.