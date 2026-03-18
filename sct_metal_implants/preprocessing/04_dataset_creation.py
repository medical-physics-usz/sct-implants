import argparse
import os
import numpy as np
import nibabel as nib
from preprocessing_helpers import create_folder
import glob
import pandas as pd

# Adapted from https://github.com/medical-physics-usz/synthetic_CT_generation with thesis-specific modifications
def create_slices(path_root_preprocessed, modalities_to_process):
    """Convert 3D NIfTI volumes into 2D slices for dataset.
    Creates train/test folders, processes modalities and masks, and saves
    slices per patient based on Excel split definition.

    Parameters:
        path_root_preprocessed (str): root path for preprocessed data
        modalities_to_process (list[str]): list of modalities to process (e.g., ["CT","MR_in"])
    """
    # Definition of metal masks to process
    masks_to_process = ["MR_in_masks_metal", "CT_masks_metal"]

    # Path definition of input path (normalization) and output path (dataset)
    path_dataset = os.path.join(path_root_preprocessed, "dataset")
    path_normalization_before = os.path.join(path_root_preprocessed, "normalization", "before")
    path_normalization_after = os.path.join(path_root_preprocessed, "normalization", "after")

    # Load Excel with info about train / test split
    path_excel_patient_info = os.path.join(path_root_preprocessed, "Excel", "patient_info.xlsx")
    df_all = pd.read_excel(path_excel_patient_info, engine="openpyxl", dtype={"StudyID": str})

    # Create Train / Test subfolders
    if not os.path.exists(path_excel_patient_info):
        print(f"{path_excel_patient_info} does NOT exist!")
        return

    # Create Dataset Folders
    for modality in modalities_to_process + masks_to_process:
        modality_dir = os.path.join(path_dataset, modality)
        create_folder(modality_dir, f"{modality} directory", force=True)

    # Patient Folders
    path_data = os.path.join(path_root_preprocessed, "data_exported")
    patient_folders = [name for name in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, name))]

    # Loop through patients in excel
    for index, line in df_all.iterrows():
        # Get patient number and split
        split = line.Split
        patient_nr = line.StudyID

        if not split in ["train", "test"]:
            continue

        if patient_nr not in patient_folders:
            print(f"{patient_nr} not in Preprocessed Data")
            continue

        # Loop through NIFTI files to process per patient (modality + masks)
        for modality in modalities_to_process + masks_to_process:

            # dataset/train/modality OR dataset/test/modality
            # path_slices = os.path.join(path_dataset, split, modality)
            path_slices = os.path.join(path_dataset, modality)


            # Get NIFTI files from normalized modality folder
            if "mask" in modality:
                path_modality_normalized = os.path.join(path_normalization_before, modality)
                image_path = glob.glob(os.path.join(path_modality_normalized, f"mask_{patient_nr}*.nii"))
            else:
                path_modality_normalized = os.path.join(path_normalization_after, modality)
                image_path = glob.glob(os.path.join(path_modality_normalized, f"{patient_nr}*.nii"))

            if image_path is None:
                print(f"{path_modality_normalized} has not NIFTI files.")
                continue

            # Load NIFTI 3D volume
            image = nib.load(image_path[0])
            image_array = np.array(image.get_fdata())

            # Loop through individual slices of 3D NIFTI volume
            for i in range(0, image_array.shape[2]):
                # Load NIFTI
                im = image.slicer[..., i:(i + 1)]
                path_save = os.path.join(path_slices, patient_nr + "-" + str(i) + ".nii")
                im_data = im.get_fdata(caching="unchanged")
                im = nib.Nifti1Image(im_data, image.affine)

                # Save NIFTI
                if os.path.exists(path_save):
                    os.remove(path_save)
                nib.save(im, path_save)
            print(f"Saved {image_array.shape[2]} slices for {patient_nr}{modality}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument('--path_root_preprocessed', type=str, required=True, help="Root path for processed data output")
    parser.add_argument('--modalities', type=str, required=False, default="CT,MR_in,MR_opp",
                        help="Comma-separated list of modalities to process (e.g., CT,MR_in,MR_opp)")

    args = parser.parse_args()
    modalities_list = [m.strip() for m in args.modalities.split(',') if m.strip()]

    create_slices(args.path_root_preprocessed, modalities_list)

