import os

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from ct_handler import CTHandler
from mr_handler import MRHandler
from dicom_loader import DICOMSeries, RTStruct

class PatientHandler:
    """Manage CT, MR, and masks for a patient."""

    def __init__(self, root_dir, patient_nr, structure_keyword):
        """Initialize patient handler.
        Parameters:
            root_dir (str) -- DICOM dataset root directory
            patient_nr (str) -- patient ID
            structure_keyword (str) -- keyword for structure selection (eg., FemurHead)
        """

        self.patient_nr = patient_nr
        self.structure_keyword = structure_keyword
        self.patient_dir = os.path.join(root_dir, patient_nr)

        # Load DICOM files
        self.ct = DICOMSeries(self.patient_dir, "CT")
        self.mr = DICOMSeries(self.patient_dir, "MR_in")
        self.rt_struct = RTStruct(self.patient_dir)

        # Modality Handlers
        self.ct_handler = CTHandler()
        self.mr_handler = MRHandler()

        # Masks
        self.structure_masks = self.rt_struct.get_structure_masks(structure_keyword, self.ct) #dict
        self.ct_body_mask = self.calculate_ct_body_mask() #array

        # Donor Masks
        self.implant_masks = {}
        self.implants = {}
        self.mr_void_masks = {}
        self.mr_voids = {}

    def calculate_ct_body_mask(self):
        """Calculate body mask on CT."""
        return self.ct_handler.get_body_mask(self.ct.get_volume())

    def pad_on_top(self, mask, target_shape):
        """Pad mask in Z to match target shape.
        Parameters:
            mask (ndarray) -- input mask
            target_shape (tuple) -- desired shape
        Returns:
            ndarray -- padded mask
        """
        pad_z = target_shape[0] - mask.shape[0]
        if pad_z < 0:
            raise ValueError("Mask is larger than target in z-dimension!")
        padding = ((0, pad_z), (0, 0), (0, 0))  # pad only in Z
        return np.pad(mask, padding, mode='constant', constant_values=0)

    def pad_on_top(self, mask, target_shape):
        tz = target_shape[0]
        mz = mask.shape[0]

        if mz < tz:
            pad_z = tz - mz
            padding = ((0, pad_z), (0, 0), (0, 0))
            return np.pad(mask, padding, mode="constant", constant_values=0)

        if mz > tz:
            # crop from "top" (end of z) to match
            return mask[:tz, :, :]

        return mask

    def set_implant(self, implant_mask, structure_to_transform):
        if implant_mask.shape != self.ct.volume.shape:
            implant_mask = self.pad_on_top(implant_mask, self.ct.volume.shape)
        self.implant_masks[structure_to_transform] = implant_mask

    def set_mr_void(self, mr_void, structure_to_transform):
        if mr_void.shape != self.mr.volume.shape:
            mr_void = self.pad_on_top(mr_void, self.mr.volume.shape)
        self.mr_voids[structure_to_transform] = mr_void

    def set_transformed_structure(self, transformed_structure, structure_to_transform):
        if transformed_structure.shape != self.ct.volume.shape:
            transformed_structure = self.pad_on_top(transformed_structure, self.ct.volume.shape)
        self.structure_masks[f"{structure_to_transform}_transformed"] = transformed_structure

    def calculate_implant_and_void(self, structure_to_transform, void_creation_method="standard"):
        """Calculate implant and MR void.
        Parameters:
            structure_to_transform (str) -- structure name
            void_creation_method (str) -- void creation method
        """

        if self.patient_nr.startswith("Pat0") or self.patient_nr.startswith("1PA"): # All implant patients start with Pat0XX
            # Get implant mask
            implant_mask = self.ct_handler.get_implant_mask(self.ct.get_volume(),
                                                            self.ct_body_mask,
                                                            self.structure_masks[structure_to_transform])

            self.implant_masks[structure_to_transform] = implant_mask
            self.implants[structure_to_transform] = self.ct.get_volume() * implant_mask

            # Get patient-specific void threshold, if available
            patient_void_threshold, patient_void_radius = self.get_patient_void_threshold_and_radius(self.patient_nr)

            # Extract void mask around implant
            self.mr_void_masks[structure_to_transform] = self.mr_handler.extract_mr_void(self.mr.get_volume(),
                                                                                         self.implant_masks[structure_to_transform],
                                                                                         self.mr.get_spacing(),
                                                                                         self.ct_body_mask,
                                                                                         threshold=patient_void_threshold,
                                                                                         radius_mm=patient_void_radius)

            # Calculate void intensities that should be transferred to receiver
            self.mr_voids[structure_to_transform] = self.mr_handler.calculate_mr_void(self.mr.get_volume(),
                                                             self.mr_void_masks[structure_to_transform],
                                                             self.implant_masks[structure_to_transform],
                                                             method=void_creation_method)
        else:
            raise ValueError("Invalid patient type.")


    def overwrite_volumes(self, structure_to_transform):
        """Overwrite CT with implant and MR with blended void region.
        Parameters:
            structure_to_transform (str) -- structure name
        """
        self.ct.volume[self.implant_masks[structure_to_transform].astype(bool)] = 3000
        self.blend_with_smooth_border(structure_to_transform, 10.0)

    def blend_with_smooth_border(self, structure_to_transform, fade_distance=10.0):
        """Blend MR with smooth void border.
        Parameters:
            structure_to_transform (str) -- structure name
            fade_distance (float) -- fade distance in mm
        """
        # Boolean mask of the void
        mask_void = self.mr_voids[structure_to_transform].astype(bool)

        # Distance inside the void to the nearest non-void pixel
        distance = distance_transform_edt(mask_void, sampling=self.mr.get_spacing())

        # Only fade in the void region
        blend_factor = np.clip(distance / fade_distance, 0, 1)

        # Create blend mask: 0 = use MR, 1 = use void
        blend_mask = np.where(mask_void, blend_factor, 0)

        # Blend the MR image with the void values
        self.mr.volume = (1 - blend_mask) * self.mr.volume + blend_mask * self.mr_voids[structure_to_transform]

    def get_patient_void_threshold_and_radius(self, patient_nr):
        """Get patient-specific void threshold and radius."""
        patient_look_up = {
            "Pat002": {"threshold": 0.1, "radius_mm": 25.0},
            "Pat003": {"threshold": 0.1, "radius_mm": 40.0},
            "Pat004": {"threshold": 0.1, "radius_mm": 25.0},
            "Pat005": {"threshold": 0.1, "radius_mm": 25.0},
            "Pat006": {"threshold": 0.075, "radius_mm": 20.0},
            "Pat008": {"threshold": 0.1, "radius_mm": 20.0},
            "Pat011": {"threshold": 0.1, "radius_mm": 40.0},
            "Pat012": {"threshold": 0.1, "radius_mm": 25.0},
            "Pat022": {"threshold": 0.2, "radius_mm": 25.0},
        }

        patient_values = patient_look_up.get(patient_nr, {"threshold": 0.1, "radius_mm": 25.0})
        return patient_values["threshold"], patient_values["radius_mm"]
