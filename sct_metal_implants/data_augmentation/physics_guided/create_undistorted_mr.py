from sct_metal_implants.data_augmentation.donor_receiver.dicom_loader import DICOMSeries

import nibabel as nib
import numpy as np

def create_undistorted_mr(augmented_patient_nr, dicom_path, output_path):

    # Path to metal mask
    metal_mask_path = f"mask_{augmented_patient_nr}.nii" # metal-mask-path

    # Path to MR of receiver patient (without implant)
    receiver_patient_nr = augmented_patient_nr.split("_")[0]
    dicom_path_patient = dicom_path + receiver_patient_nr

    # Load MR without implant
    mr = DICOMSeries(dicom_path_patient, "MR_in")
    print(mr.get_volume().shape)

    # Load metal mask
    metal_mask_img = nib.load(metal_mask_path)
    metal_mask = metal_mask_img.get_fdata().transpose(2, 1, 0)
    print(metal_mask.shape)

    # Mask mr
    undistorted_mr = mr.volume.copy()
    undistorted_mr[metal_mask.astype(bool)] = 0

    # Save undistorted MR
    hdr = metal_mask_img.header
    hdr['cal_min'] = 0
    hdr['cal_max'] = 500

    # Load MR volume (assumed shape: Z, Y, X or similar)
    mr_volume = mr.volume.astype(np.float32)

    # Apply mask (set metal voxels to 0 or NaN)
    undistorted_mr = mr_volume.copy()
    undistorted_mr[metal_mask.astype(bool)] = 0

    # Use affine from the metal mask or create identity if none exists
    affine = metal_mask_img.affine

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(undistorted_mr.transpose(2, 1, 0), affine, hdr)

    # Save undistorted MR
    output_path_patient = output_path + f"undistorted_MR_in_{augmented_patient_nr}.nii"
    nib.save(nifti_img, output_path_patient)
    print(f"Saved undistorted MR of: {augmented_patient_nr}")
