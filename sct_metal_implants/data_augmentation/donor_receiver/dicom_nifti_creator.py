import os
import numpy as np
import nibabel as nib
from pydicom.uid import generate_uid


class DicomNiftiCreator:
    """Create and save NIfTI or DICOM series."""

    # Used for DICOM series description
    mr_void_creation_map = {
        "standard": "std",
        "overwrite_min_intensity": "min",
        "overwrite_gradient_based": "gradient",
        "sample_from_distribution": "sample"
    }

    def __init__(self, output_root_dir, donor_patient_nr, receiver_patient_nr, structure_name, sides_to_transform):
        """Initialize creator and output directory.
        Parameters:
            output_root_dir (str) -- root output directory
            donor_patient_nr (str) -- donor patient ID
            receiver_patient_nr (str) -- receiver patient ID
            structure_name (str) -- structure being transformed
            sides_to_transform (list) -- sides (e.g., ['L','R'])
        """
        self.output_dir = os.path.join(output_root_dir, f"{receiver_patient_nr}_aug_by_{donor_patient_nr}")
        self.description = f"{structure_name} {' '.join(sides_to_transform)} to {receiver_patient_nr} from {donor_patient_nr}"
        os.makedirs(self.output_dir, exist_ok=True)

    def get_affine(self, spacing, origin, orientation):
        """Build affine matrix from spacing, origin, orientation."""
        dz, dy, dx = spacing
        row_cos = np.array(orientation[:3])
        col_cos = np.array(orientation[3:])
        slice_cos = np.cross(row_cos, col_cos)

        direction = np.stack([row_cos, col_cos, slice_cos], axis=1)  # shape (3,3)

        affine = np.eye(4)
        affine[:3, :3] = direction * [dx, dy, dz]  # note voxel size order
        affine[:3, 3] = origin
        return affine

    def save_nifti(self, volume, affine, output_path):
        """Save NIfTI with affine."""
        nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
        nib.save(nifti_img, output_path)
        print(f"Saved NIfTI to: {output_path}")

    def save_nifti_for_imagej(self, volume, spacing, output_path):
        """Save NIfTI in ImageJ-compatible format."""
        # Transpose from (Z, Y, X) → (X, Y, Z)
        volume_reordered = np.transpose(volume, (2, 1, 0))  # X, Y, Z

        # Build basic affine (spacing only, no orientation)
        affine = np.eye(4)
        affine[0, 0] = spacing[2]  # dx
        affine[1, 1] = spacing[1]  # dy
        affine[2, 2] = spacing[0]  # dz

        img = nib.Nifti1Image(volume_reordered.astype(np.float32), affine)
        nib.save(img, output_path)
        print(f"✅ Saved NIfTI for ImageJ to: {output_path}")

    def create_file_name(self, modality, mr_void_creation_method):
        """Create output filename."""
        if modality == "CT":
            return "CT.nii"
        return f"{modality}_{self.mr_void_creation_map[mr_void_creation_method]}.nii"

    def save_dicom(self, volume, original_slices, modality):
        """Save volume as a new DICOM series."""
        series_description = f"Augmented {modality} series: {self.description}"
        output_dir_modality = os.path.join(self.output_dir, modality)
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




