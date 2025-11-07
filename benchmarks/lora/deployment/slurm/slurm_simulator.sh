#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=${EXP_OUTPUT_PATH}/log_%j.out
#SBATCH --error=${EXP_OUTPUT_PATH}/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=${EXP_MAX_DURATION_SECONDS}
module load singularity
singularity exec --nv ${EXP_ENV_VARS} ${EXP_CONTAINER_IMAGE} ${EXP_RUN_COMMAND}