import os
from collections import defaultdict

import pandas as pd
from data.base_dataset import BaseDataset, get_params, get_transform
import nibabel as nib
import numpy as np
import torch
from PIL import Image

# [thesis] AlignedImplantDataset class created based on AlignedDataset
class AlignedImplantDataset(BaseDataset):
    """Flexible multi-input dataset (e.g., MR modalities, masks) with CT output."""

    def __init__(self, opt):
        """Initialize the dataset class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.context = opt.pseudo3D_window

        # Load allowed patients to include from Excel
        self.allowed_patients = None
        if hasattr(opt, 'patient_list_excel') and opt.patient_list_excel and os.path.exists(opt.patient_list_excel):
            df = pd.read_excel(opt.patient_list_excel, engine='openpyxl')
            if 'StudyID' not in df.columns:
                raise ValueError("Excel file must contain a column named 'patient_id'")
            self.allowed_patients = set(df['StudyID'].astype(str).str.strip())

            # patients used for validation/testing
            test_patients = set(df[df["Split"] == opt.current_split]["StudyID"].astype(str))
            # patients used for training
            train_patients = set(df[df["Split"] != opt.current_split]["StudyID"].astype(str))

            print(test_patients, train_patients)

            if opt.phase == "train":
                self.allowed_patients = train_patients
            else:
                self.allowed_patients = test_patients

            print(f"[INFO] Filtering dataset to {len(self.allowed_patients)} patient(s) from Excel.")

        # Define input channels
        self.input_channels = opt.input_modalities

        # Build dict of input channel -> folder path
        self.channel_dirs = {
            ch: os.path.join(opt.dataroot, ch)
            for ch in self.input_channels
        }

        # Define output channels
        self.output_channels = opt.output_modalities

        # Map output channel -> folder path
        self.output_dirs = {
            ch: os.path.join(opt.dataroot, ch)
            for ch in self.output_channels
        }

        file_dict = defaultdict(list)

        # Path to CT target images
        self.ct_dir = os.path.join(opt.dataroot, 'CT')

        # Group filenames by patient ID
        for file in os.listdir(self.ct_dir):
            if not file.endswith('.nii'):
                continue
            patient_id, slice_idx = file.replace('.nii', '').split('-')

            if self.allowed_patients and patient_id not in self.allowed_patients:
                continue  # Skip unwanted patients

            file_dict[patient_id].append((int(slice_idx)))

        # Collect valid file-names
        valid_filenames = []

        for patient_id in file_dict:
            slice_indices = sorted(file_dict[patient_id])
            min_idx = min(slice_indices)
            max_idx = max(slice_indices)

            for idx in slice_indices:
                if (idx - self.context >= min_idx) and (idx + self.context <= max_idx):
                    # This slice has enough neighbors
                    filename = f"{patient_id}-{idx}.nii"
                    valid_filenames.append(filename)

        self.filenames = sorted(valid_filenames)


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains:
            A (tensor)              -- concatenated input images (eg. MR reconstructions)
            B (tensor)              -- its corresponding image in the target domain (CT)
            A_paths (str)           -- image paths
            B_paths (str)           -- image paths
            CT_metal_mask (tensor)  -- CT metal mask
            MR_metal_mask (tensor)  -- MR metal mask
        """
        filename = self.filenames[index]
        patient_id, slice_str = filename.replace('.nii', '').split('-')
        slice_idx = int(slice_str)

        # Load CT target
        ct_path = os.path.join(self.ct_dir, filename)
        ct_img_np = self.load_nifti_slice(ct_path)
        ct_img_pil = Image.fromarray((ct_img_np * 255).astype(np.uint8))

        # Transform setup
        transform_params = get_params(self.opt, ct_img_pil.size)
        transform_for_inputs = get_transform(self.opt, transform_params, grayscale=True)
        transform_for_target = get_transform(self.opt, transform_params, grayscale=True)

        # Load all input channels with pseudo-3D context
        input_imgs = []
        input_paths = []

        for ch in self.input_channels:
            slices = []
            for offset in range(-self.context, self.context + 1):
                neighbor_idx = slice_idx + offset
                neighbor_filename = f"{patient_id}-{neighbor_idx}.nii"
                neighbor_path = os.path.join(self.channel_dirs[ch], neighbor_filename)

                slice_np = self.load_nifti_slice(neighbor_path)
                slice_pil = Image.fromarray((slice_np * 255).astype(np.uint8))
                slices.append(transform_for_inputs(slice_pil))
                input_paths.append(neighbor_path)

            input_channel_tensor = torch.cat(slices, dim=0)  # [3, H, W]
            input_imgs.append(input_channel_tensor)

        # Combine channels → shape: [num_input_channels, H, W]
        A = torch.cat(input_imgs, dim=0)

        # Load all output channels (to be used as Pix2Pix target)
        output_imgs = []

        for ch in self.output_channels:
            output_path = os.path.join(self.output_dirs[ch], filename)
            output_np = self.load_nifti_slice(output_path)
            output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
            output_tensor = transform_for_target(output_pil)
            output_imgs.append(output_tensor)

        # Concatenate outputs into a single multi-channel tensor
        B = torch.cat(output_imgs, dim=0)  # Shape: [C_out, H, W]

        # Load CT Metal Mask
        ct_metal_mask = self.load_metal_mask(filename, 'CT_masks_metal', transform_for_target)

        # Load MR Metal Mask
        mr_metal_mask = self.load_metal_mask(filename, 'MR_in_masks_metal', transform_for_inputs)

        return {
            'A': A,
            'B': B,
            'CT_metal_mask': ct_metal_mask,
            'MR_metal_mask': mr_metal_mask,
            'A_paths': input_paths,
            'B_paths': ct_path
        }

    def load_nifti_slice(self, path):
        """Loads a 2D slice from a NIfTI file (already normalized to [0, 1]).
        Parameters:
            - path (str): Path to NIFTI file
        Returns:
            - img_data (np.ndarray): loaded NIFTI slice
        """
        img_data = np.squeeze(nib.load(path).get_fdata(caching="unchanged"))
        return img_data.astype(np.float32)

    def load_metal_mask(self, filename, folder_name, transform):
        """Load a metal mask slice.
        Parameters:
            - path (str): Path to NIFTI file
            - folder_name (str): Folder name
            - transform (callable): Transformation function to be applied
        Returns:
            - img_pil (tensor): loaded and transformed mask slice
        """

        metal_mask_path = os.path.join(self.opt.dataroot, folder_name, filename)
        img_np = self.load_nifti_slice(metal_mask_path)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        return transform(img_pil)
