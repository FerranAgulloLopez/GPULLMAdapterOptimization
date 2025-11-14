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