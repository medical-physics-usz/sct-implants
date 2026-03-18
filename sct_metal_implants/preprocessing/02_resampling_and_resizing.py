import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel import processing

from preprocessing_helpers import create_folder
from resampling_and_resizing_helpers import affine_no_rotation, crop_or_pad


# --- CONSTANTS ---
modality_background = {
    "CT": -1024,
    "MR_in": 0,
    "MR_opp": 0,
    "MR_W": 0,
    "MR_F": 0,
    "mask": 0
}

crop_x_desired=256
crop_y_desired=256
desired_voxel_size_x = 1.6304
desired_voxel_size_y = 1.6304
desired_voxel_size_z = 2

# Additional files to process for certain modalities are defined here (body/metal masks)
resampling_plan = {
    "CT": ["mask_PatXXX_3D_body.nii", "mask_PatXXX_3D_metal.nii"],
    "MR_in": ["mask_PatXXX_3D_metal.nii"]
}

# Adapted from https://github.com/medical-physics-usz/synthetic_CT_generation with thesis-specific modifications
def resample_and_resize(path_root_preprocessed, modalities_to_process):
    """Resample and resize NIfTI volumes.
    Resamples modalities and masks to target voxel size and image dimensions.

    Parameters:
        path_root_preprocessed (str): root path for preprocessed data
        modalities_to_process (list[str]): list of modalities (e.g., ["CT","MR_in"])
    """

    # Define paths
    path_normalization = os.path.join(path_root_preprocessed, "normalization", "before")
    path_excel = os.path.join(path_root_preprocessed, "Excel")
    excel_metadata = os.path.join(path_excel, "metadata_CT_MR.xlsx")
    excel_patient_info = os.path.join(path_excel, "patient_info.xlsx")

    # Create folders for resampled modalities
    for modality in modalities_to_process + ["masks"]:
        path = os.path.join(path_normalization, modality)
        create_folder(path, "resampling folder", force=True)

        # Create extra folders for modalities with metal masks (CT, MR_in)
        if modality.startswith("MR_in") or modality.startswith("CT"):
            path = os.path.join(path_normalization, modality + "_masks_metal")
            create_folder(path, "resampling folder metal mask", force=True)

    # Read excels containing DICOM and patient data
    df_all = pd.read_excel(excel_metadata, index_col=0, engine="openpyxl")  # to read important DICOM metadata
    df_patient_info = pd.read_excel(excel_patient_info, engine="openpyxl")  # to read info about corrupted slices

    # Loop metadata excel generated from 01_preprocessing.py
    for index, line in df_all.iterrows():
        if not (line.ModalityFolder in modalities_to_process and line.PathNIFTI != ""):
            continue

        if not os.path.exists(line.PathNIFTI):
            print("no nifti for Patient = {} nifti_paths = {}".format(line.PatientNr, line.PathNIFTI))
            continue

        # Read data from excel row
        patient_nr = line.PatientNr
        modality = line.ModalityFolder
        nifti_path = line.PathNIFTI

        print(f"Working on Patient = {patient_nr} modality = {modality}")

        # Define which files to resample /resize for modality
        files_to_process = ["PatXXX_3D_body.nii"] + resampling_plan.get(modality, [])

        # Get original pixel spacing
        pixel_size = np.array(line.PixelSpacing.replace("(", "").replace(")", "").replace(",", "").split()).astype(
            np.float32)
        current_dim_x = int(pixel_size[0] * 10000) / 10000
        current_dim_y = int(pixel_size[1] * 10000) / 10000
        print(current_dim_x, current_dim_y)

        # Using only Start - End Slice based on manually performed corrupted slice identification on MR_in
        row = df_patient_info[df_patient_info['StudyID'] == patient_nr]

        if row.empty:
            print(f"Patient {patient_nr} not in Excel")
            continue

        first_slice = int(row.iloc[0]["SliceStart"])
        last_slice = int(row.iloc[0]["SliceEndAfterCut"])
        print("SLICE RANGE: ", first_slice, last_slice)

        # Loop through NIFTI files to process per modality
        for file in files_to_process:

            # Load file
            file_name = file.replace("PatXXX", patient_nr)
            image_path = os.path.join(nifti_path, file_name)
            image = nib.load(image_path)
            image = affine_no_rotation(image)
            initial_affine = image.affine

            # Resample to desired voxel size (x,y,z)
            key = "mask" if "mask" in file_name else modality
            background = modality_background[key]

            image_resampled = processing.resample_to_output(image, voxel_sizes=[desired_voxel_size_x, desired_voxel_size_y, desired_voxel_size_z],
                                                                      order=1, mode='constant', cval=background)

            nii_array_resampled = image_resampled.get_fdata()

            # Resize to desired image dimension (x,y)
            nii_array_resampled = crop_or_pad(nii_array_resampled,
                                              (crop_x_desired, crop_y_desired, nii_array_resampled.shape[2]),
                                              background)

            # Save the resampled and resized modality
            if "mask" in file_name:
                if "metal" in file_name:
                    resampled_path = os.path.join(path_normalization, modality + "_masks_metal", file_name) # path for resampled masks
                else:
                    resampled_path = os.path.join(path_normalization, "masks", file_name) # path for resampled masks
            else:
                resampled_path = os.path.join(path_normalization, modality, file_name) # path for resampled modality

            im = nib.Nifti1Image(nii_array_resampled[:, :, first_slice:last_slice], initial_affine)
            nib.save(im, resampled_path)

        print(f"Patient {patient_nr} resampled and resized!")
    print("Processed all patients!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Resampling NIFTI files")
    parser.add_argument('--path_root_preprocessed', type=str, required=True, help="Root path for processed data output")
    parser.add_argument('--modalities', type=str, required=False, default="CT,MR_in",
                        help="Comma-separated list of modalities to process (e.g., CT,MR_in,MR_opp,MR_W,MR_F)")

    args = parser.parse_args()
    modalities_list = [m.strip() for m in args.modalities.split(',') if m.strip()]

    resample_and_resize(args.path_root_preprocessed, modalities_list)






