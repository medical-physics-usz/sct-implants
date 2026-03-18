import os
from data.base_dataset import BaseDataset, get_params, get_transform
import nibabel as nib
import numpy as np
import torch
from PIL import Image

class UnalignedDatasetImplant(BaseDataset):
    """Flexible multi-input dataset (e.g., MR modalities, masks) with CT output."""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.opt = opt

        # Define input channels
        if self.opt.input_nc == 2:
            self.input_channels = ['MR_in', 'MR_opp']
        else:
            self.input_channels = ['MR_in']

        # Path to CT target images
        self.ct_dir = os.path.join(opt.dataroot, opt.phase, 'CT')

        # Build dict of input channel -> folder path
        self.channel_dirs = {
            ch: os.path.join(opt.dataroot, opt.phase, ch)
            for ch in self.input_channels
        }

        # Use filenames from MR as input list
        self.filenames_A = sorted([
            f for f in os.listdir(self.channel_dirs[self.input_channels[0]]) if f.endswith('.nii')
        ])

        # Use filenames from CT as target list
        self.filenames_B = sorted([
            f for f in os.listdir(self.ct_dir) if f.endswith('.nii')
        ])

    def __len__(self):
        return max(len(self.filenames_A), len(self.filenames_B))

    def __getitem__(self, index):

        # Get A (MR) sample
        filename_A = self.filenames_A[index % len(self.filenames_A)]

        # Get B (CT) sample - random unless serial_batches = True
        if self.opt.serial_batches:
            index_B = index % len(self.filenames_B)
        else:
            index_B = np.random.randint(len(self.filenames_B))
        filename_B = self.filenames_B[index_B]


        # === Load all input channels dynamically ===
        input_imgs = []
        input_paths = []

        for ch in self.input_channels:
            ch_path = os.path.join(self.channel_dirs[ch], filename_A)
            ch_img_np = self.load_nifti_slice(ch_path)
            ch_img_pil = Image.fromarray((ch_img_np * 255).astype(np.uint8))
            input_imgs.append(ch_img_pil)
            input_paths.append(ch_path)

        # === Load CT target ===
        ct_path = os.path.join(self.ct_dir, filename_B)
        ct_img_np = self.load_nifti_slice(ct_path)
        ct_img_pil = Image.fromarray((ct_img_np * 255).astype(np.uint8))

        # === Apply same transform to all ===
        transform_params = get_params(self.opt, ct_img_pil.size)
        transform_for_inputs = get_transform(self.opt, transform_params, grayscale=True)
        transform_for_target = get_transform(self.opt, transform_params, grayscale=True)

        # Apply to input channels
        A = torch.cat([transform_for_inputs(img) for img in input_imgs], dim=0)

        # Apply to CT target
        B = transform_for_target(ct_img_pil)

        return {
            'A': A,
            'B': B,
            'A_paths': input_paths,
            'B_paths': ct_path
        }

    def load_nifti_slice(self, path):
        """Loads a 2D slice from a NIfTI file (already normalized to [0, 1])."""
        img_data = np.squeeze(nib.load(path).get_fdata(caching="unchanged"))
        return img_data.astype(np.float32)
