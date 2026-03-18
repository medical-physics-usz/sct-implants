import argparse
import os
import numpy as np
import nibabel as nib
from preprocessing_helpers import create_folder
import glob


def normalize_mr(volume, clip_percentiles=(0, 99), new_min=0, new_max=1):
    """Normalize MR volume with percentile clipping.
    Parameters:
        volume (ndarray): input MR volume
        clip_percentiles (tuple): lower/upper percentiles for clipping
        new_min (float): minimum of normalized range
        new_max (float): maximum of normalized range
    Returns:
        ndarray: normalized MR volume
    """
    # Clip out outliers
    lower = np.percentile(volume, clip_percentiles[0])
    upper = np.percentile(volume, clip_percentiles[1])
    volume = np.clip(volume, lower, upper)

    # Min-max normalize to [new_min, new_max]
    if upper == lower:
        return np.full_like(volume, new_min)

    norm_volume = (volume - lower) / (upper - lower)
    return norm_volume * (new_max - new_min) + new_min

def normalize_ct(volume, min_, max_):
    """Normalize CT volume with fixed clipping.
    Parameters:
        volume (ndarray): input CT volume
        min_ (float): minimum HU value
        max_ (float): maximum HU value
    Returns:
        ndarray: normalized CT volume
    """
    # Clip CT values at min and max
    volume = np.clip(volume, min_, max_)
    norm_volume = (volume - min_) / (max_ - min_)

    return norm_volume

def normalize_all(path_root_preprocessed, modalities_to_process, max_ct_intensity=3000):
    """Normalize all NIfTI files for given modalities.
    Parameters:
        path_root_preprocessed (str): root path for processed data
        modalities_to_process (list[str]): list of modalities to normalize
        max_ct_intensity (int): CT clipping maximum
    Returns:
        None
    """
    # Paths to input folder (before) and output folder (after)
    path_normalization_before = os.path.join(path_root_preprocessed, "normalization", "before")
    path_normalization_after = os.path.join(path_root_preprocessed, "normalization", "after")

    # Loop through modalities
    for modality in modalities_to_process:
        # Modality paths
        path_modality_before = os.path.join(path_normalization_before, modality)
        path_modality_after = os.path.join(path_normalization_after, modality)
        create_folder(path_modality_after, "normalization", force=True)

        # Get all NIFTI paths
        image_paths = glob.glob(os.path.join(path_modality_before, "*.nii"))

        # Loop through all NIFTI files and normalize
        for image_path in image_paths:
            image = nib.load(image_path)
            patient = str(image_path.split('/')[-1])
            image_array = np.array(image.get_fdata())
            print(f"Mean before= {np.mean(image_array)} patient {patient}.")

            if modality == "CT":
                normalized_image_array = normalize_ct(image_array, min_=-1024, max_=max_ct_intensity)
            else:
                normalized_image_array = normalize_mr(image_array)

            print("mean after= {}.".format(np.mean(normalized_image_array)))
            norm_nifti = nib.Nifti1Image(normalized_image_array, image.affine)
            path_norm_nifti = os.path.join(path_modality_after, patient)
            nib.save(norm_nifti, path_norm_nifti)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Normalizing Images")
    parser.add_argument('--path_root_preprocessed', type=str, required=True, help="Root path for processed data output")
    parser.add_argument('--modalities', type=str, required=False, default="CT,MR_in,MR_opp",
                        help="Comma-separated list of modalities to process (e.g., CT,MR_in,MR_opp)")
    parser.add_argument('--max_ct_intensity', type=int, required=False, default=3000,
                        help="Max Intensity for CT")

    args = parser.parse_args()
    modalities_list = [m.strip() for m in args.modalities.split(',') if m.strip()]
    max_ct_intensity = args.max_ct_intensity

    normalize_all(args.path_root_preprocessed, modalities_list, max_ct_intensity)