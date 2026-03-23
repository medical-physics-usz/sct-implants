import os
import numpy as np
import nibabel as nib
from pydicom.uid import generate_uid

from sct_metal_implants.data_augmentation.donor_receiver.dicom_loader import DICOMSeries
from sct_metal_implants.data_augmentation.physics_guided.GRE_Dixon3D_ForwardModel_IntravoxelSinc import \
    GRE_Dixon3D_ForwardModel_IntraVoxelSinc

# -----------------------------
# DICOM-derived acquisition params
# -----------------------------
# TE_ms = 2.39 # for opp - less voids
TE_ms = 4.77 #for in phase - more voids
BW_per_pixel = 1116.0

# DICOM: InPlanePhaseEncodingDirection='COL' -> readout is rows -> axis=0 in numpy (y)
readout_axis = 0          # 0 = rows/y, 1 = cols/x
gradient_polarity = "+"   # flip to "-" if distortion direction is reversed

# Spacing
VOXEL_SPACING_MM = (1.5625, 1.5625, 2.0)  # numpy order (y,x,z)

# B0 voxel size (your high-res off-res map)
B0_SPACING_MM = (0.5, 0.5, 0.5)


def create_distorted_mr(augmented_patient_nr, output_path_undistorted, output_path_permuted_B0, path_receiver, output_path):

    undistorted_path_patient = os.path.join(output_path_undistorted, f"undistorted_MR_in_{augmented_patient_nr}.nii")
    permuted_B0_path_patient = os.path.join(output_path_permuted_B0, f"B0_{augmented_patient_nr}.nii")

    # Load NIfTIs
    und_img = nib.load(undistorted_path_patient)
    b0_img = nib.load(permuted_B0_path_patient)

    und = und_img.get_fdata(dtype=np.float32)
    b0 = b0_img.get_fdata(dtype=np.float32)

    # Initialize model
    model = GRE_Dixon3D_ForwardModel_IntraVoxelSinc(
        TE_ms=TE_ms,
        BW_per_pixel=BW_per_pixel,
        gradient_polarity=gradient_polarity,
        readout_axis=readout_axis,  # COL -> readout rows
        voxel_spacing_mm=VOXEL_SPACING_MM,
        b0_spacing_mm=B0_SPACING_MM,
        void_floor=0,
        dephasing_power=1,
        jacobian_strength_inplane=0
    )

    # Make prediction
    pred, inter = model.forward(und, b0, verbose=True)

    # Load original MR-inphase
    receiver_patient_nr = augmented_patient_nr.split("_")[0]
    dicom_path_receiver_patient = path_receiver + receiver_patient_nr
    mr = DICOMSeries(dicom_path_receiver_patient, "MR_in")

    # Save DICOM
    output_path_patient = os.path.join(output_path, augmented_patient_nr)
    os.makedirs(output_path_patient, exist_ok=True)

    description = f"Augmented: {augmented_patient_nr}"
    save_dicom(pred.transpose(2, 1, 0), mr.get_slices(), "MR_in", description, output_path_patient)


# Helper to save final DICOM
def save_dicom(volume, original_slices, modality, description, output_dir):

    series_description = f"Augmented {modality} series: {description}"
    output_dir_modality = os.path.join(output_dir, modality)
    os.makedirs(output_dir_modality, exist_ok=True)

    new_series_uid = generate_uid()

    for i, slice_data in enumerate(volume):
        ref_dcm = original_slices[i]
        new_dcm = ref_dcm.copy()

        # Metadata
        new_dcm.SeriesInstanceUID = new_series_uid
        new_dcm.SeriesDescription = series_description
        new_dcm.InstanceNumber = i + 1

        # Convert HU to raw pixel values
        slice_data = np.clip(slice_data, -1024, 3071)  # clip to safe HU range
        slice_data = (slice_data + 1024).astype(np.uint16)

        new_dcm.PixelData = slice_data.tobytes()
        new_dcm.Rows, new_dcm.Columns = slice_data.shape

        # Correct pixel format metadata
        new_dcm.RescaleSlope = 1
        new_dcm.RescaleIntercept = -1024
        new_dcm.PixelRepresentation = 0  # unsigned
        new_dcm.BitsAllocated = 16
        new_dcm.BitsStored = 16
        new_dcm.HighBit = 15

        filename = os.path.join(output_dir_modality, f"slice_{i:03d}.dcm")
        new_dcm.save_as(filename)

    print(f"✅ DICOM series saved to: {output_dir_modality}")