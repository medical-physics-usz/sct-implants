#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/home/nzala/data/results/synCT/Ph_augmentation/%j.out

# Load modules
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

DONOR_PATH="/shares/tanadini-lang.physik.uzh/sCT/raw_data/patients_with_hip_implant/" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant
RECEIVER_PATH="/shares/tanadini-lang.physik.uzh/sCT/raw_data/patients_without_implant/" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_without_implant
OUTPUT_PATH="/home/nzala/scratch/datasets/raw_data/patients_augmented_balgrist_TEST" # eg. /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant_augmented
CONFIG_PATH="/home/nzala/scratch/datasets/raw_data/patients_augmented_balgrist_TEST" # Path to directory with data_augmentation_configurations.xlsx

REPO_ROOT="/home/nzala/code/paper_repository"

# ====================

# Log job metadata
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
conda info --envs | grep '*'

# Run augmentation pipeline
echo "--- Start Preprocessing Pipeline ---"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python -m sct_metal_implants.data_augmentation.physics_guided.run_Ph_augmentation \
                --donor_path "$DONOR_PATH" \
                --receiver_path "$RECEIVER_PATH" \
                --output_path "$OUTPUT_PATH" \
                --config_path "$CONFIG_PATH"

echo "Finished at: $(date)"

