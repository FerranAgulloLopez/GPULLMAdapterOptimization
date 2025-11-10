#!/bin/bash
#SBATCH --job-name=create_dummy_datasets
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/create_dummy_datasets/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/create_dummy_datasets/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
module load singularity
singularity exec --nv --env PYTHONPATH=. vllm-benchmark.sif python3 benchmarks/lora/create_dummy_dataset.py
