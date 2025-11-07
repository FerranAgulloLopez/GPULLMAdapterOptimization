#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=${EXP_OUTPUT_PATH}/log_%j.out
#SBATCH --error=${EXP_OUTPUT_PATH}/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --time=${EXP_MAX_DURATION_SECONDS}
module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${EXP_CONDA_ENV}
${EXP_RUN_COMMAND}