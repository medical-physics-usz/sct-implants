#!/usr/bin/bash -l
#SBATCH --job-name=matrad_demo
#SBATCH --time=15:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/PATH-TO-OUTPUT-LOG/%j.out
#SBATCH --constraint=INTEL

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
PATH_MATRAD="PATH-TO-MATRAD-REPOSITORY" # Path to cloned matRad repository: eg., /home/USERNAME/matlab/matRad
PATH_SCRIPT="PATH-TO-MATLAB-SCRIPT" # Path to matlab file: eg., /home/USERNAME/matlab/matRad/examples/

# Path to matLab script to execute
if [ "$ALL_ANGLES" = true ]; then
  PATH_SCRIPT_FULL="${PATH_SCRIPT}/dosimtric_evaluation_hip_implants_all_angles.m" # Full-arcs (all angles)
else
  PATH_SCRIPT_FULL="${PATH_SCRIPT}/dosimtric_evaluation_hip_implants.m" # Partial-arcs (avoidance angles)
fi

PATH_REAL_DATA="PATH-TO-REAL-DICOM-DATA" # Path to rCT (processed to match sCT) and /RTstructs
PATH_FAKE_DATA="PATH-TO-FAKE-DICOM-DATA" # Path to synthetic DICOM data (sCT postprocessed)
PATH_OUTPUT="PATH-TO-OUTPUT-FOLDER"      # Path where DVH results and RTDose files should be saved
PATIENT_INFO_EXCEL_PATH="PATH-TO-FOLDER-WITH-PATIENT-INFO-EXCEL/${DATA}/Excel" # Path to Excel containing patient information (patient_info.xlsx)

# Derived Paths
PATH_FAKE_DATA_FULL="${PATH_FAKE_DATA}/${DATA}/${MODEL}/test/"
PATH_OUTPUT_FULL="${PATH_OUTPUT}/${DATA}/${MODEL}/${NAME}/"
PATIENT_INFO_EXCEL_PATH_FULL

# Definition of MatLab Command
MATLAB_CMD="warning('off','all'); \
addpath('$PATH_MATRAD'); \
path_real_data = '$PATH_REAL_DATA'; \
path_fake_data = '$PATH_FAKE_DATA_FULL'; \
path_output = '$PATH_OUTPUT_FULL'; \
path_excel = '$PATIENT_INFO_EXCEL_PATH'; \
modelName = string('$NAME'); \
resolution = string('$RESOLUTION'); \
run('PATH_SCRIPT_FULL'); \
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
