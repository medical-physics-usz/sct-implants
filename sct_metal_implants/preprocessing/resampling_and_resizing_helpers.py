import math
import os
import re
import shutil
import string

import numpy as np
import pandas as pd
import pydicom
from pydicom import dcmread
import dicom2nifti
import nibabel as nib
import cv2
from scipy.ndimage import morphology, measurements, filters, \
    binary_opening, binary_closing, binary_erosion, binary_dilation, binary_fill_holes

"""Utilities adapted from medical-physics-usz/synthetic_CT_generation.

Source: https://github.com/medical-physics-usz/synthetic_CT_generation
Modifications: Adjusted and extended for use in the current project 
"""

def affine_no_rotation(image):
    initial_affine=image.affine
    initial_affine[0, 0] = abs(initial_affine[0, 0])
    initial_affine[1, 1] = abs(initial_affine[1, 1])
    image = nib.Nifti1Image(np.array(image.get_fdata()), initial_affine)
    return image

# --- Resize ---

def crop_or_pad(img, new_size_tuple,background):
    for axis in range(2):
        if new_size_tuple[axis] != img.shape[axis]:
            if new_size_tuple[axis] > img.shape[axis]:
                img = pad(img, new_size_tuple[axis], axis, background)
            else:
                img = crop(img, new_size_tuple[axis], axis)
    return img

def pad(img, size, axis, background):

    old_size = img.shape[axis]
    pad_size = float(size - old_size) / 2
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (math.floor(pad_size), math.ceil(pad_size))
    return np.pad(img, pads, 'constant', constant_values=(background,background ))


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

# Maintain full content, no cropping

def resize_and_pad(img, target_size=(256, 256), background=0):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)

    # Resize while preserving aspect ratio
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Pad to target size
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = np.pad(
        resized,
        ((top, bottom), (left, right), (0, 0)) if img.ndim == 3 else ((top, bottom), (left, right)),
        mode='constant',
        constant_values=background
    )

    return padded

