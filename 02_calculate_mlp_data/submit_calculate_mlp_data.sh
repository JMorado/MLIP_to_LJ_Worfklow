#!/bin/bash
#SBATCH --job-name=mlp_calc
#SBATCH --account=comet_ngammcb
#SBATCH --partition=default_free
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --ntasks=1
#SBATCH --array=751-1031%128

# Format folder name with 4-digit zero padding
RUN_ID=$(printf "run_%04d" ${SLURM_ARRAY_TASK_ID})

echo "Running task ${SLURM_ARRAY_TASK_ID}"
echo "Entering folder: $RUN_ID"

cd $RUN_ID || { echo "Folder $RUN_ID not found!"; exit 1; }

# Run your python job
export OMP_NUM_THREADS=32
export OPENMM_CPU_THREADS=32

# Run your python job
python calculate_mlp_data.py --mlp-name mace-off24-medium --mlp-device cpu --n-frames 1 > compute_mlp_data.output 2> compute_mlp_data.errors