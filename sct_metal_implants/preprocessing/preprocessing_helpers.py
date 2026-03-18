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

# --- DIRECTORIES ---

def create_folder(path, description="", force=False):
    if os.path.exists(path):
        if force:
            try:
                print(f"⚠️  {description or 'Folder'} already exists at '{path}'. Removing and recreating.")
                shutil.rmtree(path)
                os.makedirs(path)
            except OSError as e:
                print(f"❌ Error removing folder '{path}': {e.strerror}")
        else:
            print(f"✅ {description or 'Folder'} already exists at '{path}'. Skipping creation.")
    else:
        print(f"📁 Creating {description or 'folder'} at '{path}'.")
        os.makedirs(path)

# NIFTI DIRECTORIES

def setup_nifti_directories(modality, path, patient_number, new_path):
    dicom_path = os.path.join(path, modality)
    nifti_path = os.path.join(new_path, patient_number, modality + "_nifti")

    if os.path.exists(dicom_path):
        if os.path.exists(nifti_path):
            print(f"Path {nifti_path} already exists - Remove! ⚠️")
            shutil.rmtree(nifti_path)

        os.makedirs(nifti_path)

    return dicom_path, nifti_path

# --- DICOM ---

def extract_dicom_parameters(modality_dicom_path):
    """
    Processes DICOM files in a specified directory, sorts them by z-coordinates, modifies metadata, and extracts image parameters.

    Args:
    - modality_dicom_path (str): Path to the directory containing DICOM files for the modality.
    - modality_category (str): Prefix/category to identify DICOM files within the directory.

    Returns:
    - sample_image (pydicom.Dataset): A sample DICOM image used to extract shape, pixel spacing, and image position.
    - z_coords (list): List of z-coordinates for the sorted slices.
    - slice_thickness (float): Calculated slice thickness between DICOM slices.
    - shape (tuple): Shape of the DICOM images (x, y, z).
    - pix_spacing (list): Pixel spacing in x and y dimensions.
    - im_position (list): Image position of the DICOM slices.
    """
    # Get all DICOM files that match the modality category
    dcm_files = [file for file in os.listdir(modality_dicom_path)]
    dcm_paths = [os.path.join(modality_dicom_path, file) for file in dcm_files if
                 os.path.isfile(os.path.join(modality_dicom_path, file))]

    # Sort DICOM files by z-coordinates
    sorted_dcm_paths, z_coords, error_sorting = sort_slices(dcm_paths)

    # Calculate slice thickness
    slice_thickness = [np.round(z_coords[j] - z_coords[i], 1) for i, j in
                       zip(np.arange(len(z_coords) - 1), np.arange(1, len(z_coords)))]
    slice_thickness = slice_thickness[0]  # Assume constant thickness across slices

    # Load a sample image to get shape and pixel spacing information
    sample_image = dcmread(sorted_dcm_paths[2])
    shape = (sample_image.Columns, sample_image.Rows, len(sorted_dcm_paths))
    pix_spacing = sample_image.PixelSpacing
    im_position = sample_image.ImagePositionPatient

    return sample_image, z_coords, slice_thickness, shape, pix_spacing, im_position

def sort_slices(filepaths):
    error = []
    positions = []
    try:
        filepaths.sort(key=lambda x: float(dcmread(x).ImagePositionPatient[2]),
                       reverse=False)
        datasets = [dcmread(x, force=True) for x in filepaths]
        positions = [round(float(ds.ImagePositionPatient[2]), 2) for ds in datasets]
        positions.sort(reverse=False)
        # positions = [round(float(ds.ImagePositionPatient[2]),3) for ds in datasets]
    except AttributeError:
        try:
            filepaths.sort(key=lambda x: float(dcmread(x).SliceLocation), reverse=True)
        except AttributeError:
            try:
                sample_image = dcmread(filepaths[0])
                if sample_image.PatientPosition == 'HFS':
                    filepaths.sort(key=lambda x: dcmread(x, force=True).ImageIndex,
                                   reverse=True)
                if sample_image.PatientPosition == 'FFS':
                    filepaths.sort(key=lambda x: dcmread(x, force=True).ImageIndex)
            except AttributeError:
                error = 'Ordering of slices not possible due to lack of attributes'
    return filepaths, positions, error


def convert_dicom_to_nifti(modality_dicom_path, path_nifti, desired_filename):
    """
    Convert DICOM files to NIfTI format, save with a custom filename, and return the path of the saved file.

    Parameters:
    - modality_dicom_path (str): Path to the directory containing DICOM files.
    - path_nifti (str): Directory where the NIfTI file will be saved.
    - desired_filename (str): Desired name for the output NIfTI file, including the extension (e.g., 'output_file.nii').

    Returns:
    - str: The full path to the saved NIfTI file.

    Raises:
    - FileNotFoundError: If the DICOM path does not exist.
    - Exception: For errors during the conversion or renaming process.
    """
    if not os.path.exists(modality_dicom_path):
        raise FileNotFoundError(f"🔴 DICOM path {modality_dicom_path} does not exist.")

    # Convert DICOM to NIfTI
    if os.listdir(path_nifti):
        print("3D NIfTI already created.")
        nifti_file_path = os.path.join(path_nifti, desired_filename)
    else:
        try:
            dicom2nifti.convert_dir.convert_directory(
                modality_dicom_path,
                path_nifti,
                compression=False,  # No compression
                reorient=False  # No reorientation
            )

            # Define the default filename from the DICOM directory
            default_filename = os.listdir(path_nifti)[0]
            default_filepath = os.path.join(path_nifti, default_filename)

            # Define the new filename and path
            nifti_file_path = os.path.join(path_nifti, desired_filename)

            # Rename the file
            os.rename(default_filepath, nifti_file_path)
        except Exception as e:
            print(f"🔴  Error during DICOM to NIfTI conversion: {e}")
            raise

    return nifti_file_path

def save_dicom_modality_info(patient_nr, modality, modality_dicom_path, path_nifti, shapeZ, slice_thickness, **kwargs):
    """
    Create a DataFrame with DICOM modality information for a given patient and modality.

    Args:
        shapeZ(str): number of slices
        slice_thickness (float): Thickness of the slices.
        patient_folder (str): Patient folder name.
        treatment_day (str): Treatment day.
        treatment_site (str): Treatment site.
        modality (str): Modality name (e.g., "CT", "MR").
        modality_dicom_path (str): Path to the DICOM files.
        path_nifti (str): Path to save the NIFTI files.
        **kwargs: Additional keyword arguments for more specific information.

    Returns:
        pd.DataFrame: DataFrame with the collected information.
    """
    sl_t = slice_thickness if slice_thickness else 0
    #### check whether pathDICOM and pathNIFTI is what we need
    df_data = {
        "PatientNr": patient_nr,
        'ModalityFolder': modality,
        'PathDICOM': modality_dicom_path,
        'PathNIFTI': path_nifti,
        "shapeZ": shapeZ,
        'SliceThickness': sl_t,
    }

    df_paths = pd.DataFrame([df_data])
    return df_paths

# --- BODY MASK ---

def extract_and_save_body_mask(nii_image, path_nifti, threshold_for_body_mask, patient_folder, modality):
    """
    Process CT modality to create and save body mask and masked input.

    Args:
        nii_image (nib.Nifti1Image): NIFTI image object.
        path_nifti (str): Directory path to save NIFTI files.
        threshold_for_body_mask (float): Threshold for CT body mask.
        patient_folder (str): Folder name for the patient.
        modality (str): Modality name ("CT", "CT_reg", or "MR").

    Returns:
        np.ndarray: Masked input array.
    """
    # Get NIFTI array (data) and affine matrix from the NIFTI image object
    nii_array = nii_image.get_fdata()
    affine = nii_image.affine

    # Generate body mask
    mask_threshold = get_body_mask_threshold(nii_array, threshold_for_body_mask)

    # Save the body mask as a NIFTI file
    path_mask_file = os.path.join(path_nifti, f'mask_{patient_folder}_3D_body.nii')
    save_nifti_image(mask_threshold, affine, path_mask_file)

    # Create masked input and save it
    masked_input = get_masked_input(nii_array, mask_threshold, modality)
    path_masked_input_file = os.path.join(path_nifti, patient_folder + '_3D_body.nii')
    save_nifti_image(masked_input, affine, path_masked_input_file)

    return mask_threshold

def get_body_mask_threshold(nii_array, threshold):
    mask = np.zeros(nii_array.shape)
    mask[nii_array > threshold] = 1
    mask[nii_array <= threshold] = 0
    mask = binary_erosion(mask, iterations=2).astype(np.uint8)
    mask = get_mask_biggest_contour(mask)
    #mask = binary_dilation(mask, iterations=5).astype(np.int16) # before intersected mask, dilation was used
    return mask.astype(np.int16)

def get_mask_biggest_contour(mask):
    for i in range(mask.shape[2]):
        inmask = np.expand_dims(mask[:, :, i].astype(np.uint8), axis=2)
        ret, bin_img = cv2.threshold(inmask, 0.5, 1, cv2.THRESH_BINARY)
        (cnts, _) = cv2.findContours(np.expand_dims(bin_img, axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # return None, if no contours detected
        if len(cnts) != 0:

            # sort contours by area descending
            cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
            area1 = cv2.contourArea(cnts_sorted[0])

            keep = [cnts_sorted[0]]

            # decide whether to keep 2nd contour too
            if len(cnts_sorted) > 1:
                area2 = cv2.contourArea(cnts_sorted[1])
                if area1 > 0 and (area2 >= (1.0 - 0.5) * area1):
                    keep.append(cnts_sorted[1])

            # draw kept contours into a fresh mask
            out = np.zeros_like(bin_img, dtype=np.uint8)
            cv2.drawContours(out, keep, contourIdx=-1, color=1, thickness=-1)

            mask[:, :, i] = out

    return mask

def save_nifti_image(data, affine, file_path):
    """
    Save a NIFTI image to the specified path.

    Args:
        data (np.ndarray): Data to save.
        affine (np.ndarray): Affine matrix of the NIFTI image.
        file_path (str): Path where the NIFTI image will be saved.
    """
    if data.dtype == bool:
        data = data.astype(np.uint8)
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, file_path)

def get_masked_input(nii_array, mask, modality):
    """
    Get the masked input array with the appropriate background based on the modality.

    Args:
        nii_array (np.ndarray): NIFTI array of the image.
        mask (np.ndarray): Binary mask array.
        modality (str): Modality name ("CT", "CT_reg", or "MR").

    Returns:
        np.ndarray: Masked input array with proper background values.
    """
    masked_input = nii_array * mask

    # Set the background value based on the modality
    if modality == "CT":
        masked_input[mask == 0] = -1024  # Background for CT and CT_reg
    else:
        masked_input[mask == 0] = 0  # Background for MR

    return masked_input.astype(np.int16)

# --- METAL MASK ---

def extract_and_save_metal_mask(metal_mask_extractor, nii_image, nifti_path, patient_nr, body_mask, modality):
    # Get NIFTI array (data) and affine matrix from the NIFTI image object
    nii_array = nii_image.get_fdata()
    affine = nii_image.affine

    # Extract Metal Mask
    if modality == "CT":
        body_mask_metal = body_mask
        metal_mask = metal_mask_extractor.get_metal_mask_CT(nii_array, body_mask_metal, min_region=30)  ###
    # Extract Metal Candidate Mask
    elif modality == "MR_in":
        body_mask_metal = metal_mask_extractor.extract_body_mask(nii_array)
        metal_mask = metal_mask_extractor.extract_void(nii_array, body_mask_metal)

    metal_plot_path = os.path.join(nifti_path, f'{patient_nr}_metal_mask.html')
    metal_mask_extractor.plot_regions(metal_mask, body_mask_metal, metal_plot_path)

    # Save the body mask as a NIFTI file
    path_mask_file = os.path.join(nifti_path, f'mask_{patient_nr}_3D_metal.nii')
    save_nifti_image(metal_mask, affine, path_mask_file)

# --- PATIENT INFO ---

def get_patient_ID(path):
    return path.split("/")[-2]

# --- EXCEL ---

def save_dicom_tags_to_excel(sample_image, dicom_tags, ignore_tags, df_dicom_modality_info, path_excel):
    """
    Extract DICOM tags and save them to an Excel file.

    Args:
        sample_image: The DICOM image used to extract tags.
        dicom_tags: Tags to search for in the DICOM image.
        ignore_tags: Tags to ignore during extraction.
        df_dicom_modality_info: DataFrame containing modality info to concatenate.
        path_excel: Path to the Excel file for saving the results.
    """
    # Extract DICOM tags
    dicom_dict_tags = search_dicom_tags(sample_image, dicom_tags, ignore_tags)
    df_dicom = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in dicom_dict_tags.items()})

    # Concatenate modality info with DICOM tags
    df_patient = pd.concat([df_dicom_modality_info.reset_index(drop=True),
                            df_dicom.reset_index(drop=True)], axis=1, join='outer', sort=True)

    # Check if Excel file already exists
    if os.path.exists(path_excel):
        #df_all = pd.read_excel(path_excel)
        df_all = pd.read_excel(path_excel, engine='openpyxl')

        df_all = df_all[df_all.columns.drop(list(df_all.filter(regex='Unnamed.*')))]
        df_all = pd.concat([df_all.reset_index(drop=True), df_patient.reset_index(drop=True)], axis=0,
                           sort=True)
        df_all.to_excel(path_excel, index=False)  # Save updated DataFrame
    else:
        df_patient.to_excel(path_excel, index=False)  # Create new Excel file

def search_dicom_tags(ds, tags, ignore_tags):
    dict_dicom = {}
    for name in tags:
        if ((name in ds) and (name not in ignore_tags)):
            dict_dicom[name] = assign_type(ds[name].value, ignore_tags)
        else:
            dict_dicom[name] = 'NaN'
    delete_keys = []
    dict_dicom2 = dict_dicom.copy()
    for key, value in dict_dicom.items():
        if type(value) == dict:
            out_dict = dict_dicom[key]
            dict_dicom2.update(out_dict)
            delete_keys.append(key)
    for key in delete_keys:
        del dict_dicom2[key]
    return dict_dicom2

def assign_type(s, ignore_keys):
    if type(s) == list or type(s) == pydicom.multival.MultiValue:
        try:
            for x in s:
                if type(x) == pydicom.valuerep.DSfloat:
                    list_values = [float(x) for x in s if x not in ignore_keys]
                    return str(tuple(list_values))
                else:
                    list_values = [x for x in s if x not in ignore_keys]
                    return str(tuple(list_values))
        except ValueError:
            try:
                return [float(x) for x in s if x not in ignore_keys]
            except ValueError:
                return [format_string(x) for x in s if ((len(x) > 0) and (x not in ignore_keys))]
    elif type(s) == pydicom.sequence.Sequence:
        return get_seq_data(s, ignore_keys)
    else:
        s = str(s)
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return format_string(s)

def format_string(in_string):
    formatted = re.sub(r'[^\x00-\x7f]', r'', str(in_string))  # Remove non-ascii characters
    formatted = ''.join(filter(lambda x: x in string.printable, formatted))
    if len(formatted) == 1 and formatted == '?':
        formatted = None
    return formatted

def get_seq_data(sequence, ignore_keys):
    seq_data = {}
    for seq in sequence:
        for s_key in seq.dir():
            if s_key in ignore_keys:
                continue
            s_val = getattr(seq, s_key, '')
            if type(s_val) == pydicom.sequence.Sequence:
                _seq = get_seq_data(s_val, ignore_keys)
                seq_data[s_key] = _seq
                continue
            if type(s_val) == str:
                s_val = format_string(s_val)
            else:
                s_val = assign_type(s_val, ignore_keys)
            if s_val:
                seq_data[s_key] = s_val
    return seq_data