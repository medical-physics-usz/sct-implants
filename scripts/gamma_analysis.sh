#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:30:00
#SBATCH --output=/PATH-TO-OUTPUT-LOG/%j.out

# Load modules
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

MODEL="pix2pix"
NET_G="resnet_9blocks"      # Generator eg., resnet_9blocks | unet_256 | resnet_spade | resnet_attention | resnet_weighted
DATA="patients_TEST"        # Dataset name eg., patients_mixed OR patients_mixed_augmented
INPUT_MODALITIES="MR_in"    # Input modalities, comma-seperated eg., MR_in,MR_opp,MR_W

INPUT_MODALITIES_NAME="${INPUT_MODALITIES//,/_}"
NAME="${INPUT_MODALITIES_NAME}_${NET_G}" # Name of the experiment, eg., MR_in_resnet_9blocks

RESOLUTION="2mm" # Resolution of dosimetric evaluation
ALL_ANGLES=false # False for partial-arc, True for full-arc

if [ "$ALL_ANGLES" = true ]; then
  CONFIGURATION="RTDose_${RESOLUTION}_all_angles" # Full-arcs (all angles)
else
  CONFIGURATION="RTDose_${RESOLUTION}" # Partial-arcs (avoidance angles)
fi

PATH_DATA="PATH-TO-DOSE-FOLDER-ROOT" # Path to root folder where RTDose files are saved from dosimetric_evaluation.sh eg., /home/USERNAME/data/results/synCT/dosimetric_evaluation
PATH_RESULT="OUTPUT-PATH-FOR-GAMMA-ANALYSIS" # eg., /home/USERNAME/data/results/synCT/gamma_analysis
DICOM_PATH="PATH-TO-DICOM-DATA-OF-PATIENTS" # eg., /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant
PATIENT_INFO_EXCEL_PATH="PATH-TO-FOLDER-WITH-PATIENT-INFO-EXCEL/${DATA}/Excel" # Path to Excel containing patient information (patient_info.xlsx)

# ====================

# Derived Variables
PATH_DATA_DOSE="${PATH_DATA}/${DATA}/${MODEL}/${NAME}/${CONFIGURATION}" # Directory containing RTDose files from DVH eval
PATH_GAMMA_RESULT="${PATH_RESULT}/${DATA}/${MODEL}/${NAME}/${CONFIGURATION}" # Directory where gamma anylsis results should be saved

# ====================

# Log job metadata
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
conda info --envs | grep '*'

# Run postprocessing
echo "--- Start Postprocessing Pipeline ---"
python ../sct_metal_implants/evaluation/gamma_analysis/gamma_analysis.py \
                --path_RTDose_files "$PATH_DATA_DOSE" \
                --path_output "$PATH_GAMMA_RESULT" \
                --path_original_dicom_data "$DICOM_PATH" \
                --path_excel "$PATIENT_INFO_EXCEL_PATH"

echo "Finished at: $(date)"
