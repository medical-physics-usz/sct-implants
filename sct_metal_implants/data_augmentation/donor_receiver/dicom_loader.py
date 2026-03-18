import os
import numpy as np
import pydicom
from skimage.draw import polygon

class DICOMSeries:
    """Handle a DICOM series and build a 3D volume."""

    def __init__(self, patient_dir, modality):
        """Initialize DICOMSeries.
        Parameters:
            patient_dir (str) -- patient directory
            modality (str) -- modality subfolder name (eg., CT)
        """
        path = os.path.join(patient_dir, modality)
        self.volume, self.spacing, self.origin, self.slices = self.load_dicom_series(path)

    def load_dicom_series(self, dir_path, skip_first=0, skip_last=0):
        """Load a DICOM series and return volume with metadata.
        Parameters:
            dir_path (str) -- path to DICOM folder
            skip_first (int) -- number of slices to skip at beginning
            skip_last (int) -- number of slices to skip at end
        Returns:
            tuple -- (volume, spacing, origin, slices)
        """
        files = sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".dcm")
        ])

        slices = [pydicom.dcmread(f) for f in files]
        slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

        # Extract spacing and origin
        z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
        pixel_spacing = slices[0].PixelSpacing  # [dy, dx]
        spacing = [z_spacing, pixel_spacing[0], pixel_spacing[1]]  # (Z, Y, X)
        origin = list(map(float, slices[0].ImagePositionPatient))  # (X, Y, Z)
        orientation = list(map(float, slices[0].ImageOrientationPatient))  # 6 values

        slices = slices[skip_first:-skip_last] if skip_last > 0 else slices[skip_first:]

        volume = np.stack([
            getattr(ds, 'RescaleSlope', 1) * ds.pixel_array.astype(np.float32) +
            getattr(ds, 'RescaleIntercept', 0) for ds in slices
        ])

        return volume, spacing, origin, slices

    def get_volume(self):
        return self.volume

    def get_spacing(self):
        return self.spacing

    def get_origin(self):
        return self.origin

    def get_slices(self):
        return self.slices

class RTStruct:
    """Handle DICOM RT Structure Set."""

    def __init__(self, patient_directory, folder_name="RTst"):
        """Initialize RTStruct.
        Parameters:
            patient_directory (str) -- patient directory
            folder_name (str) -- name of folder containing RTSTRUCT file (.dcm)
        """
        path = os.path.join(patient_directory, folder_name)
        self.file = self.load_file(path)

    def load_file(self, path):
        """Load RT Structure file.
        Parameters:
            path (str) -- path to RTst folder
        Returns:
            Dataset -- pydicom dataset
        """
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".dcm")
        ]

        ds = pydicom.dcmread(files[0])
        return ds

    def get_structure_masks(self, structure_keyword, ct):
        """Get masks for structures matching a keyword.
        Parameters:
            structure_keyword (str) -- keyword to search in ROI names
            ct (DICOMSeries) -- reference CT series
        Returns:
            dict -- structure name -> mask volume
        """
        contours = self.get_structure_contours(structure_keyword)
        masks = self.contours_to_mask(contours, ct.get_volume().shape, ct.get_spacing(), ct.get_origin())
        return masks

    def get_structure_contours(self, structure_keyword):
        """Extract contour points for ROIs matching keyword.
        Parameters:
            structure_keyword (str) -- keyword to search
        Returns:
            dict -- structure name -> list of contours
        """
        roi_map = {
            item.ROIName: item.ROINumber
            for item in self.file.StructureSetROISequence
            if structure_keyword.lower() in item.ROIName.lower()
        }

        contour_data = {}
        for roi_name, roi_number in roi_map.items():
            for roi in self.file.ROIContourSequence:
                if roi.ReferencedROINumber == roi_number:
                    contour_data[roi_name] = [
                        np.array(contour.ContourData).reshape(-1, 3)
                        for contour in roi.ContourSequence
                    ]
        return contour_data

    def contours_to_mask(self, contours, volume_shape, spacing, origin):
        """Convert contour points to binary masks.
        Parameters:
            contours (dict) -- structure name -> list of contours
            volume_shape (tuple) -- (Z, Y, X)
            spacing (list) -- voxel spacing (Z, Y, X)
            origin (list) -- volume origin (X, Y, Z)
        Returns:
            dict -- structure name -> binary mask volume
        """
        structure_masks = {}
        for structure, contour_list in contours.items():
            mask = np.zeros(volume_shape, dtype=np.uint8)
            for pts in contour_list:
                z_idx = int(round((pts[0, 2] - origin[2]) / spacing[0]))
                x_voxels = (pts[:, 0] - origin[0]) / spacing[2]
                y_voxels = (pts[:, 1] - origin[1]) / spacing[1]
                rr, cc = polygon(y_voxels, x_voxels, shape=volume_shape[1:])
                if 0 <= z_idx < volume_shape[0]:
                    mask[z_idx, rr.astype(int), cc.astype(int)] = 1
            structure_masks[structure] = mask
        return structure_masks
