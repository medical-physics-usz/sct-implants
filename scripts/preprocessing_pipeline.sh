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

# Define one or more input DICOM datasets depending on the experiment
# (e.g., for a mixed dataset also add a path to /patients_without_implant)
RAW_DICOM_DATA_1="PATH-TO-DICOM-DATA"   # e.g., /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant

# Define where preprocessed data should be saved
ROOT_PREPROCESSED="PATH-TO-OUTPUT"      # e.g., /home/USERNAME/scratch/datasets/processed_data/patients_with_hip_implant

# Define which modalities should be processed
MODALITIES="CT,MR_in" # comma-separated string of modalities "CT,MR_in,MR_opp,MR_W,MR_F"

# Define maximal CT intensity for clipping
MAX_CT_INTENSITY=3000

# ====================


# Log job metadata
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
conda info --envs | grep '*'

# Run pipeline
echo "--- Start Preprocessing Pipeline ---"
python ../sct_metal_implants/preprocessing/01_preprocessing.py \
                --path_raw_dicom_data "$RAW_DICOM_DATA_1" \
                --path_root_preprocessed "$ROOT_PREPROCESSED" \
                --modalities "$MODALITIES"
echo "Done 01_preprocessing.py"

python ../sct_metal_implants/preprocessing/02_resampling_and_resizing.py \
                --path_root_preprocessed "$ROOT_PREPROCESSED" \
                --modalities "$MODALITIES"
echo "Done 02_resampling_and_resizing.py"

python ../sct_metal_implants/preprocessing/03_normalization.py \
                --path_root_preprocessed "$ROOT_PREPROCESSED" \
                --modalities "$MODALITIES" \
                --max_ct_intensity "$MAX_CT_INTENSITY"
echo "Done 03_normalization.py"

python ../sct_metal_implants/preprocessing/04_dataset_creation.py \
                --path_root_preprocessed "$ROOT_PREPROCESSED" \
                --modalities "$MODALITIES"
echo "Done 04_dataset_creation.py"

echo "Finished at: $(date)"
