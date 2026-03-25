#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:30:00
#SBATCH --output=/PATH-TO-OUTPUT-LOG/%j.out

# Load modules
module load cuda
module load miniforge3
source activate sct-metal-implants

# ====================
#  Variable definition
# ====================

MODEL="pix2pix"
NET_G="resnet_9blocks"           # Generator eg., resnet_9blocks | unet_256 | resnet_spade | resnet_attention | resnet_weighted
DATA="patients_with_hip_implant" # Dataset name eg., patients_mixed OR patients_mixed_augmented
PATH_DATA="PATH-TO-PREPROCESSED-DATA-ROOT" # eg., /home/USERNAME/scratch/datasets/processed_data
PATH_RESULT="PATH-TO-RESULTS-ROOT" # eg., /home/USERNAME/data/results/synCT
INPUT_MODALITIES="MR_in"         # Input modalities, comma-seperated eg., MR_in,MR_opp,MR_W
OUTPUT_MODALITIES="CT"           # Output modalities
PSEUDO_3D_WINDOW=0               # 0 = pure 2D, >0 = pseudo-3D window-size (left  and right context of adjacent slices)
SPLIT="split1"                   # Current split for CV

INPUT_MODALITIES_NAME="${INPUT_MODALITIES//,/_}"
NAME="${INPUT_MODALITIES_NAME}_${NET_G}_${SPLIT}" # Name of the experiment, eg., MR_in_resnet_9blocks

# ====================

# Derived Variables
PATH_DATA_PREPROCESSED="${PATH_DATA}/${DATA}" # Preprocessed data found via PATH-TO-PREPROCESSED-DATA/DATA-SET
PATH_DATA_SET="${PATH_DATA}/${DATA}/dataset"
PATH_RESULT_INFERENCE="${PATH_RESULT}/${DATA}/${MODEL}/test/${NAME}"
CHECKPOINT_DIR="${PATH_RESULT}/${DATA}/${MODEL}/train"

# ====================

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Check which environment is active
conda info --envs | grep '*'

# Model Training
python ../external_models/pytorch-CycleGAN-and-pix2pix/train.py \
                --dataroot "$PATH_DATA_SET" \
                --name "$NAME" \
                --checkpoints_dir "$CHECKPOINT_DIR"\
                --model "$MODEL" \
                --direction AtoB \
                --dataset_mode aligned_implant \
                --input_modalities "$INPUT_MODALITIES" \
                --output_modalities "$OUTPUT_MODALITIES" \
                --pseudo3D_window "$PSEUDO_3D_WINDOW" \
                --preprocess None \
                --n_epochs 2 \
                --n_epochs_decay 0 \
                --save_epoch_freq 2 \
                --display_id -1 \
                --netG "$NET_G" \
                --patient_list_excel "${PATH_DATA_PREPROCESSED}/Excel/patient_info.xlsx" \
                --current_split "$SPLIT"

echo "Finished Training at: $(date)"

# Model Inference
python ../external_models/pytorch-CycleGAN-and-pix2pix/test.py \
                --phase test \
                --dataroot "$PATH_DATA_SET" \
                --name "$NAME" \
                --checkpoints_dir "$CHECKPOINT_DIR" \
                --path_root_results "$PATH_RESULT_INFERENCE" \
                --model "$MODEL" \
                --dataset_mode aligned_implant \
                --direction AtoB \
                --input_modalities "$INPUT_MODALITIES" \
                --output_modalities "$OUTPUT_MODALITIES" \
                --pseudo3D_window "$PSEUDO_3D_WINDOW" \
                --preprocess none \
                --netG "$NET_G" \
                --max_ct_intensity 3000 \
                --patient_list_excel "${PATH_DATA_PREPROCESSED}/Excel/patient_info.xlsx" \
                --current_split "$SPLIT"

echo "Finished Inference at: $(date)"

# Model Evaluation
python ../sct_metal_implants/evaluation/image_similarity_evaluation/evaluation.py \
                --path_root_preprocessed "$PATH_DATA_PREPROCESSED" \
                --path_root_inference "$PATH_RESULT_INFERENCE" \
                --path_output "$PATH_RESULT_INFERENCE"

echo "Finished at: $(date)"
