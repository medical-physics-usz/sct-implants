#!/usr/bin/bash -l
#SBATCH --job-name=matrad_demo
#SBATCH --time=15:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/home//nzala/data/results/matlab/dosimetric_evaluation/%j.out
#SBATCH --constraint=INTEL
#SBATCH --exclude=u24-chi0000-401

module load matlab

# Variables
MODEL="pix2pix"
NET_G="resnet_9blocks"      # Generator eg., resnet_9blocks | unet_256 | resnet_spade | resnet_attention | resnet_weighted
DATA="patients_TEST"        # Dataset name eg., patients_mixed OR patients_mixed_augmented
INPUT_MODALITIES="MR_in"    # Input modalities, comma-seperated eg., MR_in,MR_opp,MR_W

INPUT_MODALITIES_NAME="${INPUT_MODALITIES//,/_}"
NAME="${INPUT_MODALITIES_NAME}_${NET_G}" # Name of the experiment, eg., MR_in_resnet_9blocks

RESOLUTION="10" # [mm] Resolution of dosimetric evaluation
ALL_ANGLES=false # False for partial-arc, True for full-arc

# Paths
PATH_MATRAD="/home/nzala/matlab/matRad" # Path to matRad repository

# Path to matLab script to execute
if [ "$ALL_ANGLES" = true ]; then
  PATH_SCRIPT="/home/nzala/matlab/matRad/examples/dosimtric_evaluation_hip_implants_all_angles.m"
else
  PATH_SCRIPT="/home/nzala/matlab/matRad/examples/dosimtric_evaluation_hip_implants.m"
fi

PATH_REAL_DATA="/home/nzala/matlab/dosimetric_calculation/dicom_data_real/" # Path to rCT (processed to match sCT) and /RTstructs
PATH_FAKE_DATA="/home/nzala/data/results/synCT"  # Path to sCT
PATH_OUTPUT="/home/nzala/data/results/synCT/dosimetric_evaluation" # Path where DVH results and RTDose files should be saved
PATIENT_INFO_EXCEL_PATH="/home/nzala/scratch/datasets/processed_data/${DATA}/Excel" # Path to Excel containing patient information

# Derived Paths
PATH_FAKE_DATA_FULL="${PATH_FAKE_DATA}/${DATA}/${MODEL}/test/"
PATH_OUTPUT_FULL="${PATH_OUTPUT}/${DATA}/${MODEL}/${NAME}/"

# Definition of MatLab Command

MATLAB_CMD="warning('off','all'); \
addpath('$PATH_MATRAD'); \
path_real_data = '$PATH_REAL_DATA'; \
path_fake_data = '$PATH_FAKE_DATA_FULL'; \
path_output = '$PATH_OUTPUT_FULL'; \
path_excel = '$PATIENT_INFO_EXCEL_PATH'; \
modelName = string('$NAME'); \
resolution = string('$RESOLUTION'); \
run('$PATH_SCRIPT'); \
exit;"

# Log some job info
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

echo "${DATA} ${MODEL} ${NAME}"
# Run the MATLAB script
matlab -nodisplay -nosplash -nodesktop -r "$MATLAB_CMD"

# Delete unwanted RTDose files
find "$PATH_OUTPUT_FULL" -type f -name 'RTDose_*' ! -name 'RTDose_1_physicalDose.dcm' -delete

echo "Finished at:     $(date)"
