import argparse
import os

import numpy as np
import pandas as pd
import glob
import nibabel as nib
from preprocessing_helpers import create_folder, setup_nifti_directories, extract_and_save_body_mask, \
    save_dicom_tags_to_excel, save_nifti_image, get_masked_input, get_body_mask_threshold, extract_and_save_metal_mask
from preprocessing_helpers import convert_dicom_to_nifti, extract_dicom_parameters, save_dicom_modality_info
from preprocessing_helpers import get_patient_ID
from metal_mask_extractor import MetalMaskExtractor

# --- CONSTANTS ---
THRESHOLD_CT_BODY_MASK = -400
THRESHOLD_MR_BODY_MASK = 20

# --- DICOM TAGS ---

dicom_tags = {'Rows', 'Columns', 'PixelSpacing', 'ImagePositionPatient', 'PatientAge', 'PatientBirthDate', 'StudyDate',
              'PatientSex', 'Modality', 'SeriesDescription',
              'BodyPartExamined'}

# Adapted from https://github.com/medical-physics-usz/synthetic_CT_generation with thesis-specific modifications
def preprocess(paths_raw_dicom_data, path_root_preprocessed, modalities_to_process):
    """Preprocess DICOM files and convert to NIfTI.
    Converts raw DICOM to NIfTI, extracts body/metal masks,
    and saves metadata to Excel.

    Parameters:
        paths_raw_dicom_data (list[str]): one or more raw DICOM paths
        path_root_preprocessed (str): output root for processed data
        modalities_to_process (list[str]): list of modalities (e.g., ["CT","MR_in"])
    """

    # Define data and excel paths
    path_data = os.path.join(path_root_preprocessed, "data_exported")
    path_excel = os.path.join(path_root_preprocessed, "Excel")
    excel_metadata = os.path.join(path_excel, "metadata_CT_MR.xlsx")
    excel_patient_info = os.path.join(path_excel, "patient_info.xlsx")

    create_folder(path_excel, "excel")

    number_of_processed_patients = 0

    # Instantiation of MetalMaskExtractor Class to extract metal mask from CT and metal candidate mask from MR
    metal_mask_extractor = MetalMaskExtractor()

    # Read patient info excel
    df_patient_info = pd.read_excel(excel_patient_info, engine="openpyxl")  # to only preprocess patients of interest

    # Extract patient paths from all raw DICOM data directories
    patient_paths = []
    for path in paths_raw_dicom_data:
        matched_paths = glob.glob(f'{path}/*/')
        for matched_path in matched_paths:
            patient_paths.extend([matched_path])

    # Loop through all patients and apply pipeline: DICOM to NIFTI transformation, mask extraction, DICOM params to excel
    for patient_path in patient_paths:

        patient_nr = get_patient_ID(patient_path)
        number_of_processed_patients += 1

        #if patient_nr not in ["1PA173", "1PA180"]:
        #    continue

        # Skip patients that are not of interest
        if not patient_nr in df_patient_info["StudyID"].values:
            continue

        # Loop through all modalities (CT, MR_in, ...)
        for modality in modalities_to_process:
            # Create NIFTI folders
            dicom_path, nifti_path = setup_nifti_directories(modality, patient_path, patient_nr, path_data)

            # Get DICOM parameters
            sample_image, z_coords, slice_thickness, shape, pix_spacing, im_position = extract_dicom_parameters(
                dicom_path)

            # Convert original DICOM to NIFTI and save it
            nifti_filename = patient_nr + "_3D_input.nii"
            saved_nifti_file_path = convert_dicom_to_nifti(dicom_path, nifti_path, nifti_filename)

            # Extract masks (body, metal, metal candidate)
            nii_image = nib.load(saved_nifti_file_path)

            # Extract body mask
            body_mask_threshold = THRESHOLD_CT_BODY_MASK if modality == "CT" else THRESHOLD_MR_BODY_MASK
            body_mask = extract_and_save_body_mask(nii_image, nifti_path, body_mask_threshold, patient_nr, modality)

            # Extract metal mask
            if modality in ["CT", "MR_in"]:
                extract_and_save_metal_mask(metal_mask_extractor, nii_image, nifti_path, patient_nr, body_mask, modality)

            # Save DICOM modality info
            df_dicom_modality_info = save_dicom_modality_info(
                shapeZ=shape[2],
                slice_thickness=slice_thickness,
                modality=modality,
                modality_dicom_path=dicom_path,
                path_nifti=nifti_path,
                patient_nr=patient_nr,
            )

            print(f"👍 NIfTI files successfully created for patient: {patient_nr}, modality: {modality}")

            # Modality info and DICOM tags to excel
            save_dicom_tags_to_excel(sample_image, dicom_tags, {""}, df_dicom_modality_info, excel_metadata)

        # Calculate intersection of CT and MR body mask
        patient_path = os.path.join(path_data, patient_nr)

        body_mask_ct = nib.load(os.path.join(patient_path, "CT_nifti", f"mask_{patient_nr}_3D_body.nii")).get_fdata()
        body_mask_mr = nib.load(os.path.join(patient_path, "MR_in_nifti", f"mask_{patient_nr}_3D_body.nii")).get_fdata()

        body_mask_intersection = ((body_mask_ct > 0) & (body_mask_mr > 0)).astype(np.int16)

        for modality in modalities_to_process:

            # Path Nifti
            modality_path = os.path.join(patient_path, modality + "_nifti")

            # Load modality (NIFTI)
            modality_image = nib.load(os.path.join(modality_path, f"{patient_nr}_3D_input.nii"))
            modality_array = modality_image.get_fdata()
            modality_affine = modality_image.affine

            # Get masked input
            modality_masked = get_masked_input(modality_array, body_mask_intersection, modality)

            # Save masked modality
            path_masked_input_file = os.path.join(modality_path, patient_nr + '_3D_body.nii')
            save_nifti_image(modality_masked, modality_affine, path_masked_input_file)

            # Save the intersected body mask
            path_mask_file = os.path.join(modality_path, f'mask_{patient_nr}_3D_body.nii') #_intersection.nii'
            save_nifti_image(body_mask_intersection, modality_affine, path_mask_file)

    print(f"Total patients preprocessed: {number_of_processed_patients}")
    print("Saving final DICOM metadata excel")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Preprocess DICOM files and convert to NIfTI")
    parser.add_argument('--path_raw_dicom_data', type=str, nargs="+", required=True, help="One or more paths to raw DICOM data directories")
    parser.add_argument('--path_root_preprocessed', type=str, required=True, help="Root path for processed data output")
    parser.add_argument('--modalities', type=str, required=False, default="CT,MR_in,MR_opp",
                        help="Comma-separated list of modalities to process (e.g., CT,MR_in,MR_opp)")

    args = parser.parse_args()
    modalities_list = [m.strip() for m in args.modalities.split(',') if m.strip()]

    preprocess(args.path_raw_dicom_data, args.path_root_preprocessed, modalities_list)














