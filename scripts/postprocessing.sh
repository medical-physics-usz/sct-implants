#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/nzala/data/results/synCT/postprocessing/%j.out

# Load conda environment
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

PATH_ORIGINAL_DICOM_DATA="/shares/tanadini-lang.physik.uzh/sCT/raw_data/patients_with_hip_implant" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implants
PATH_FAKE_NIFTI_DATA="/home/nzala/data/results/synCT" # eg., /home/USERNAME/data/results/synCT

DATASET="patients_TEST"  # Dataset name eg., patients_mixed OR patients_mixed_augmented
MODEL="pix2pix"          # Model name eg. pix2pix, cyclecan, cut
NET_G="resnet_9blocks"   # Generator eg., resnet_9blocks | unet_256 | resnet_spade | resnet_attention | resnet_weighted
INPUT_MODALITIES="MR_in" # Input modalities, comma-seperated eg., MR_in,MR_opp,MR_W
SPLIT="split2"           # Current split for CV

INPUT_MODALITIES_NAME="${INPUT_MODALITIES//,/_}"
NAME="${INPUT_MODALITIES_NAME}_${NET_G}_${SPLIT}" # Name of the experiment, eg., MR_in_resnet_9blocks

PATH_EXCEL="/home/nzala/scratch/datasets/processed_data/${DATASET}/Excel" # eg. folder containing patient_info.xlsx

# ====================

# Log job metadata
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
conda info --envs | grep '*'

# Run postprocessing
echo "--- Start Postprocessing Pipeline ---"
python ../sct_metal_implants/postprocessing/NIFTI_to_DICOM_resampling.py \
                --path_original_dicom_data "$PATH_ORIGINAL_DICOM_DATA" \
                --path_fake_nifti_data "$PATH_FAKE_NIFTI_DATA" \
                --path_excel "$PATH_EXCEL" \
                --dataset "$DATASET" \
                --model "$MODEL" \
                --model_name "$NAME"

echo "Finished at: $(date)"
