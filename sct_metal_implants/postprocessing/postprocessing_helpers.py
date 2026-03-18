import math
import os
import re
import nibabel as nib
import pydicom
import numpy as np

# --- Resize ---

# Function based on https://github.com/medical-physics-usz/synthetic_CT_generation
def crop_or_pad(img, new_size_tuple,background):
    for axis in range(2):
        if new_size_tuple[axis] != img.shape[axis]:
            if new_size_tuple[axis] > img.shape[axis]:
                img = pad(img, new_size_tuple[axis], axis, background)
            else:
                img = crop(img, new_size_tuple[axis], axis)
    return img

# Function based on https://github.com/medical-physics-usz/synthetic_CT_generation
def pad(img, size, axis, background):

    old_size = img.shape[axis]
    pad_size = float(size - old_size) / 2
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (math.floor(pad_size), math.ceil(pad_size))
    return np.pad(img, pads, 'constant', constant_values=(background,background ))

# Function based on https://github.com/medical-physics-usz/synthetic_CT_generation
def crop(img, size, axis):
    y_min = 0
    y_max = img.shape[0]
    x_min = 0
    x_max = img.shape[1]
    if axis == 0:
        y_min = int(float(y_max - size) / 2)
        y_max = y_min + size
    else:
        x_min = int(float(x_max - size) / 2)
        x_max = x_min + size

    return img[y_min: y_max, x_min: x_max, :]

# Sort using regex to extract slice number
def get_slice_number(filepath, patient_nr):
    """Extract slice number from NIfTI filename."""
    filename = os.path.basename(filepath)
    match = re.match(rf"{patient_nr}_(\d+)\.nii", filename)
    return int(match.group(1)) if match else -1  # invalid = -1


def load_dicom_series(dir_path, skip_first=0, skip_last=0):
    """Load a DICOM series and return slices.
    Parameters:
        dir_path (str) -- directory path
        skip_first (int) -- number of slices to skip at beginning
        skip_last (int) -- number of slices to skip at end
    Returns:
        list -- list of DICOM slices
    """
    files = sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".dcm")
    ])

    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    # Extract spacing and origin
    slices = slices[skip_first:-skip_last] if skip_last > 0 else slices[skip_first:]

    return slices