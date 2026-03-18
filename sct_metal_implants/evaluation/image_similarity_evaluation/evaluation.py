import argparse
from glob import glob
import nibabel as nib
import os
import numpy as np

from image_similarity_metrics import get_surrounding_area, mean_squared_error, mean_absolute_error, peak_signal_to_noise_ratio, \
    structural_similarity_index
from evaluation_utils import build_summary_tables, save_results_to_excel

# Material/Tissue thresholds CT
CT_THRESHOLDS = {
    "air": -400,
    "bones": 250,
    "max": 3000,
    "min": -1024,
    "metal": 2000
}

# Adapted from https://github.com/medical-physics-usz/synthetic_CT_generation with thesis-specific modifications
def evaluate_image_similarity_metrics(path_real_data_preprocessed, path_fake_data, path_output):
    """Evaluate similarity between synthetic CTs and reference CTs.

    Computes per-slice metrics (MAE, MSE, PSNR, SSIM) on masked regions,
    including inside/outside metal masks and surrounding areas.
    Aggregates results into summary tables and saves them to Excel.

    Parameters:
        path_real_data_preprocessed (str): Path to preprocessed real CT data.
        path_fake_data (str): Path to synthetic CT NIfTI data.
        path_output (str): Directory where Excel results are saved.
    """

    # Paths true data
    path_normalization_before = os.path.join(path_real_data_preprocessed, "normalization", "before")
    path_masks =  os.path.join(path_normalization_before, "masks")
    path_metal_mask = os.path.join(path_normalization_before, "CT_masks_metal")
    path_real_ct = os.path.join(path_normalization_before, "CT")

    # Path fake data
    path_fake_ct = os.path.join(path_fake_data, "fake_nifti")

    # Path to store excel-file with evaluated results
    path_excel = os.path.join(path_output, "image_similarity_results.xlsx")

    # Get patients in test set
    patients = [patient_nr for patient_nr in os.listdir(path_fake_ct) if os.path.isdir(os.path.join(path_fake_ct, patient_nr))]

    # Per evaluation variables
    surrounding_masks = {}
    result_test = []
    result_metal = []

    # Variables for Metal detection metrics
    total_metal_slices = 0
    total_drawn_in_region = 0
    total_drawn_outside_region = 0
    total_slices = 0

    # Loop over patients in test set
    for patient_nr in patients:

        fake_ct_paths = sorted(glob(os.path.join(path_fake_ct, patient_nr, '*.nii')))
        nr_slices = (len(fake_ct_paths))
        total_slices += nr_slices

        # Loop over slices of current patient
        for slice in range(nr_slices):
            # Loading fake CT
            fake_ct = nib.load(os.path.join(path_fake_ct, patient_nr, patient_nr + "_" + str(slice) + '.nii'))
            fake_ct_numpy = fake_ct.get_fdata()

            # Loading Body Mask
            mask_path = glob(os.path.join(path_masks, "mask_" + patient_nr + '*.nii'))
            mask_image = nib.load(mask_path[0])
            mask_nii_array = mask_image.get_fdata()
            slice_mask = mask_nii_array[:, :, int(slice)]

            # Loading Metal Mask
            metal_mask_path = glob(os.path.join(path_metal_mask, "mask_" + patient_nr + '*.nii'))
            metal_mask_image = nib.load(metal_mask_path[0])
            metal_mask_nii_array = metal_mask_image.get_fdata()
            metal_slice_mask = metal_mask_nii_array[:, :, int(slice)]

            # Loading (real) CT
            real_slice_path = glob(os.path.join(path_real_ct, patient_nr + '*.nii'))
            real_ct_image = nib.load(real_slice_path[0])
            real_ct_nii_array = real_ct_image.get_fdata()
            real_ct_nii_array[real_ct_nii_array < CT_THRESHOLDS["min"]] = CT_THRESHOLDS["min"]
            real_ct_nii_array[real_ct_nii_array > CT_THRESHOLDS["max"]] = CT_THRESHOLDS["max"]
            real_ct_numpy = real_ct_nii_array[:, :, int(slice)].astype(np.int16)

            # Calculate Surrounding mask 3D, since calculated in 3D only needed once per patient -> save to dictionary
            if not patient_nr in surrounding_masks:
                spacing = metal_mask_image.header.get_zooms()[::-1]
                surrounding_mask = get_surrounding_area(metal_mask_nii_array, spacing, radius_mm=20)
                surrounding_masks[patient_nr] = surrounding_mask
            else:
                surrounding_mask = surrounding_masks[patient_nr]

            # Surrounding mask of current slice 2D
            surrounding_metal_slice_mask = surrounding_mask[:, :, int(slice)]

            # Threshold implant fake
            metal_mask_fake_ct = (fake_ct_numpy > CT_THRESHOLDS["metal"])

            # Calculating Evaluation Metrics: Image Similarity metrics
            mae = mean_absolute_error(real_ct_numpy, fake_ct_numpy, slice_mask)
            mse = mean_squared_error(real_ct_numpy, fake_ct_numpy, slice_mask)
            psnr = peak_signal_to_noise_ratio(real_ct_numpy, fake_ct_numpy, slice_mask)
            ssim = structural_similarity_index(real_ct_numpy, fake_ct_numpy)

            # MAE inside / outside metal mask
            mae_implant = mean_absolute_error(real_ct_numpy, fake_ct_numpy, metal_slice_mask)
            mae_nonmetal = mean_absolute_error(real_ct_numpy, fake_ct_numpy, slice_mask * (1 - metal_slice_mask))

            # MAE inside / outside surrounding region
            mae_implant_sur = mean_absolute_error(real_ct_numpy, fake_ct_numpy, surrounding_metal_slice_mask)
            mae_nonmetal_sur = mean_absolute_error(real_ct_numpy, fake_ct_numpy,
                                                   slice_mask * (1 - surrounding_metal_slice_mask))

            ### Number of drawn metal pixels outside of surrounding metal mask
            # Pixels in fake metal mask that are outside the surrounding mask (False Positives)
            outside_mask = metal_mask_fake_ct.astype(bool) & ~surrounding_metal_slice_mask.astype(bool)
            num_outside_pixels = np.sum(outside_mask)
            has_outside_pixels = num_outside_pixels > 0
            if has_outside_pixels:
                total_drawn_outside_region += 1

            # Inside Pixels (True Positives)
            if np.sum(metal_slice_mask) > 0:
                total_metal_slices += 1
                # Overlap of metal mask (sCT) and surrounding mask (real CT)
                inside_mask = metal_mask_fake_ct.astype(bool) & surrounding_metal_slice_mask.astype(bool)
                has_metal_inside = np.sum(inside_mask) > 0
                if has_metal_inside:
                    total_drawn_in_region += 1

                result_metal.append([patient_nr, mae, has_metal_inside])

            result_test.append(
                [patient_nr, mae, mse, psnr, ssim, mae_implant, mae_nonmetal, mae_implant_sur,
                 mae_nonmetal_sur, num_outside_pixels, has_outside_pixels])

    tables = build_summary_tables(result_test, result_metal)

    save_results_to_excel(
        path_excel,
        tables["combined_mean"],
        tables["combined_std"],
        tables["mean_per_patient_normal"],
        tables["std_per_patient_normal"],
        tables["mean_per_patient_metal"],
        tables["std_per_patient_metal"],
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluation of sCTs compared to reference CTs")
    parser.add_argument('--path_root_preprocessed', type=str, required=True, help="Root path for processed data output")
    parser.add_argument('--path_root_inference', type=str, required=True, help="Root path for inference data output")
    parser.add_argument('--path_output', type=str, required=True, help="Directory where result-containing Excel is saved")

    args = parser.parse_args()

    evaluate_image_similarity_metrics(args.path_root_preprocessed, args.path_root_inference, args.path_output)

