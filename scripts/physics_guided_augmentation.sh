#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/PATH-TO-OUTPUT-LOG/%j.out

# Load modules
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

DONOR_PATH="PATH-TO-DICOM-DATA-OF-PATIENTS-WITH-IMPLANT" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant
RECEIVER_PATH="PATH-TO-DICOM-DATA-OF-PATIENTS-WITHOUT-IMPLANT" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_without_implant
OUTPUT_PATH="OUTPUT-PATH-FOR-AUGMENTED-DATA" # eg. /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant_augmented_Ph
CONFIG_PATH="PATH-TO-DONOR-RECEIVER-CONFIG-DIR" # Path to directory with data_augmentation_configurations.xlsx

REPO_ROOT="PATH-TO-REPOSITORY-ROOT" # Path to root of sct-metal-implant-paper repository, eg., "/home/USERNAME/code/paper_repository"

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

