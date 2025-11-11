#!/bin/bash
#SBATCH --job-name=_8_192_16_
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/lora/definitive_results/model_with_offloading/table/simulation_results_mean/qwen-2.5-7b-instruct/rank_16_rates_1.6_0.8_0.4/_8_192_16_/log_%j.out
#SBATCH --error=benchmarks/lora/definitive_results/model_with_offloading/table/simulation_results_mean/qwen-2.5-7b-instruct/rank_16_rates_1.6_0.8_0.4/_8_192_16_/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=00:05:00
module load singularity
singularity exec --nv  /gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm_0.8.5.sif python3 benchmarks/lora/simulation_pipeline.py --experiment-path benchmarks/lora/definitive_results/model_with_offloading/table/real_results/qwen-2.5-7b-instruct/rank_16_rates_1.6_0.8_0.4/_8_192_16_ --output-path benchmarks/lora/definitive_results/model_with_offloading/table/simulation_results_mean/qwen-2.5-7b-instruct/rank_16_rates_1.6_0.8_0.4/_8_192_16_ --print-outcome --include-computation-overhead --include-network-collapse --use-mean-version --include-preemption