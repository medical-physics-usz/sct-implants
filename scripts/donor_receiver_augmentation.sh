#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/home/nzala/data/results/synCT/DR_augmentation/%j.out

# Load modules
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

DONOR_PATH="/shares/tanadini-lang.physik.uzh/sCT/raw_data/patients_with_hip_implant" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant
RECEIVER_PATH="/shares/tanadini-lang.physik.uzh/sCT/raw_data/patients_without_implant" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_without_implant
OUTPUT_PATH="/home/nzala/scratch/datasets/raw_data/patients_augmented_TEST" # eg. /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant_augmented
PLOT_PATH="/home/nzala/scratch/datasets/raw_data/patients_augmented_TEST/plot" # Output path of generated 3D plots of (augmented) implants
CONFIG_PATH="/home/nzala/scratch/datasets/raw_data/patients_augmented_TEST" # Path to directory with data_augmentation_configurations.xlsx

# ====================

# Log job metadata
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
conda info --envs | grep '*'

# Run augmentation pipeline
echo "--- Start Preprocessing Pipeline ---"
python ../sct_metal_implants/data_augmentation/donor_receiver/run_data_augmentation.py \
                --donor_path "$DONOR_PATH" \
                --receiver_path "$RECEIVER_PATH" \
                --output_path "$OUTPUT_PATH" \
                --plot_path "$PLOT_PATH" \
                --config_path "$CONFIG_PATH"

echo "Finished at: $(date)"

