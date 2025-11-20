#!/bin/bash
#SBATCH --job-name=sim_ratio
#SBATCH --output=logs/sim_ratio_%A_%a.out
#SBATCH --error=logs/sim_ratio_%A_%a.err

#SBATCH -p gpu                # GPU partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --mem=50G              # memory request 50 GB
#SBATCH --array=1-30           # seeds 1–50
#SBATCH --time=3:00:00        # adjust as needed
#SBATCH --cpus-per-task=2      # adjust if more CPU cores needed

# Load modules or activate environment
module load gcc/14.2.0
module load openblas/0.3.28
module load cuda/12.8
module load conda/miniforge3/24.11.3-0
conda activate torch
export CUBLAS_WORKSPACE_CONFIG=:16:8

RATIO=${RATIO}
CASEONLY=${CASEONLY}
SEED=${SLURM_ARRAY_TASK_ID}

echo "Running simulation with ratio=${RATIO}, seed=${SEED}, caseonly=${CASEONLY}"
python dosim.py --seed ${SEED} --ratio ${RATIO} ${CASEONLY}

