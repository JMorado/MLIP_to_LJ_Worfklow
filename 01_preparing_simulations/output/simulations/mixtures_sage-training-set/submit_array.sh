#!/bin/bash
#SBATCH --job-name=mix
#SBATCH --account=comet_ngammcb
#SBATCH --partition=default_free
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-1031%16

export OMP_NUM_THREADS=16
export OPENMM_CPU_THREADS=16

# Format folder name with 4-digit zero padding
RUN_ID=$(printf "run_%04d" ${SLURM_ARRAY_TASK_ID})

echo "Running task ${SLURM_ARRAY_TASK_ID}"
echo "Entering folder: $RUN_ID"

cd $RUN_ID || { echo "Folder $RUN_ID not found!"; exit 1; }

# Run your python job
python -u run*.py > output 2> errors