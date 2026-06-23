#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/PATH-TO-OUTPUT-LOG/%j.out

# Load conda environment
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

# Define path to synthRAD 2023 patient data
INPUT_ROOT="PATH-TO-INPUT-DATA" # eg. /home/USERNAME/scratch/datasets/synthRAD_data/with_hip_implant

# Define path to save intermediate data
INTERMEDIATE_ROOT="PATH-TO-INPUT-DATA" # eg. /home/USERNAME/scratch/datasets/synthRAD_data/intermediate_DICOM/with_hip_implant

# Define path to save processed synthRAD 2023 patient data
INPUT_ROOT="PATH-TO-OUTPUT-DATA" # eg. /home/USERNAME/scratch/datasets/synthRAD_data/processed/with_hip_implant

# ====================

# Log job metadata
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
conda info --envs | grep '*'

# Run pipeline
echo "--- Start Preprocessing Pipeline ---"
python ../sct_metal_implants/preprocessing/prepare_synthRAD_data.py \
            --input-root "$INPUT_ROOT" \
            --intermediate-root "$INTERMEDIATE_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --add-derived-contours

echo "Finished at: $(date)"
