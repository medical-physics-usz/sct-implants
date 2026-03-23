import os
from glob import glob
import nibabel as nib
import pandas as pd
import pydicom
from nibabel import processing
import numpy as np
import argparse

from postprocessing_helpers import crop_or_pad, get_slice_number, load_dicom_series

# Target voxel spacing and resolution
original_spacing = (1.6304, 1.6304, 2.0)
target_spacing = (1.5625, 1.5625, 2.0)  # (x, y, z)
target_size = [320, 240]  # matrix size (x, y)

def convert_NIFTI_to_DICOM_with_resampling(path_real_data, path_fake_data, path_excel, dataset, model, model_name):
    """Convert NIFTI predictions to DICOM, resampled to original resolution.
    Parameters:
        path_real_data (str) -- path to original DICOM data
        path_fake_data (str) -- path to synthetic NIFTI data
        path_excel (str) -- path to Excel folder containing patient_info.xlsx
    """

    # Variables
    df_patient_info = pd.read_excel(os.path.join(path_excel, "patient_info.xlsx"), engine="openpyxl")

    print(f"Processing: {dataset} {model} {model_name}")

    # Loop through patients
    for pat_idx, pat_row in df_patient_info.iterrows():

        patient_nr = pat_row.StudyID
        start_slice = pat_row.SliceStart
        end_slice = pat_row.SliceEndAfterCut
        total_slices = pat_row.NumberSlices

        print(f"Processing: {patient_nr} {start_slice} {end_slice} {total_slices}")

        # NIFTI path
        nifti_path_fake = os.path.join(path_fake_data, dataset, model, "test", model_name, "fake_nifti", patient_nr)

        if not os.path.exists(nifti_path_fake): #if patient-nr not part of current split
            continue

        # DICOM path
        dicom_path_real = os.path.join(path_real_data, patient_nr, "CT")

        # Output path
        output_path = os.path.join(path_fake_data, dataset, model, "test", model_name, "fake_dicom", patient_nr)
        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        ###########################################################################

        # 1. Load NIFTI and create volume
        mask_paths = glob(os.path.join(nifti_path_fake, f"{patient_nr}_*.nii"))
        sorted_files = sorted(mask_paths, key=lambda f: get_slice_number(f, patient_nr))

        # Load + stack
        slices = [nib.load(p).get_fdata() for p in sorted_files]
        volume = np.stack(slices, axis=-1)  # shape: (Y, X, Z)

        # Create NIfTI image with reference affine
        ref_affine = np.diag(list(original_spacing) + [1])
        nii_volume = nib.Nifti1Image(volume, ref_affine)

        ###########################################################################

        # 2.0 Resample to desired voxel spacing
        resampled_nii = processing.resample_to_output(
            nii_volume,
            voxel_sizes=target_spacing,
            order=1,
            mode='constant',
            cval=-1024
        )

        # Extract final volume
        resampled_volume = resampled_nii.get_fdata()

        ###########################################################################

        # 3.0 Resize to desired resolution

        nii_array_resized = crop_or_pad(resampled_volume,
                                        (target_size[0], target_size[1], resampled_nii.shape[2]),
                                        -1024)

        print("Original shape:", volume.shape,
              "Resampled shape:", resampled_volume.shape,
              "Resized shape:", nii_array_resized.shape)

        ###########################################################################

        # 4.0 Load original DICOM

        original_dicom_slices = load_dicom_series(dicom_path_real, start_slice, total_slices - end_slice)
        print(len(original_dicom_slices), original_dicom_slices[0].pixel_array.shape)

        # 5.0 Overwrite DICOm files
        uid = pydicom.uid.generate_uid()
        for i in range(nii_array_resized.shape[2]):
            dcm = original_dicom_slices[i]
            slice_data = np.transpose(nii_array_resized[:, :, i], (1, 0)).astype(np.int16)

            # Overwrite pixel data
            dcm.PixelData = slice_data.tobytes()
            dcm.Rows, dcm.Columns = slice_data.shape
            dcm.BitsStored = 16
            dcm.BitsAllocated = 16
            dcm.HighBit = 15
            dcm.PixelRepresentation = 1
            dcm.RescaleSlope = 1
            dcm.RescaleIntercept = 0

            # Optional tag updates
            dcm.SeriesDescription = f"{patient_nr}: synthetic CT generated from {model} {model_name}"
            dcm.SeriesInstanceUID = uid
            dcm.PatientPosition  = "HFS"

            # Save new DICOM
            new_filename = f"{patient_nr}_{i}.dcm"
            dcm.save_as(os.path.join(output_path, new_filename))

        print(f"✅ {nii_array_resized.shape[2]} DICOM slices saved to {output_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert NIFTI files to DICOM with resampling to original resolution")
    parser.add_argument('--path_original_dicom_data', type=str, required=True, help="Path of original DICOM data")
    parser.add_argument('--path_fake_nifti_data', type=str, required=True, help="Path of synthetic NIFTI data")
    parser.add_argument('--path_excel', type=str, required=True, help="Path to excels about patient_info.xlsx")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name: eg. patients_with_hip_implant")
    parser.add_argument('--model', type=str, required=True, help="Model used: eg. pix2pix")
    parser.add_argument('--model_name', type=str, required=True, help="Name used to save model: eg. MR_in_resnet_9blocks_split1")

    args = parser.parse_args()

    # Run Convertion
    convert_NIFTI_to_DICOM_with_resampling(args.path_original_dicom_data, args.path_fake_nifti_data, args.path_excel, args.dataset, args.model, args.model_name)
