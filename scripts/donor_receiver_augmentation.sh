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
OUTPUT_PATH="OUTPUT-PATH-FOR-AUGMENTED-DATA" # eg. /home/USERNAME/scratch/datasets/raw_data/patients_with_hip_implant_augmented_DR
PLOT_PATH="OUTPUT-PATH-FOR-GENERATED-3D-PLOTS" # Output path of generated 3D plots of (augmented) implants
CONFIG_PATH="PATH-TO-DONOR-RECEIVER-CONFIG-DIR" # Path to directory with data_augmentation_configurations.xlsx

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

