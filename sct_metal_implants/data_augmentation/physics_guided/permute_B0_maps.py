import os
import nibabel as nib
import numpy as np

def create_permuted_B0_map(augmented_patient_nr, original_path, output_path, flip_x, flip_y, flip_z, swap_x_y):

    # Path to off-freq map (original)
    orignal_B0_path = os.path.join(original_path, f"B0_{augmented_patient_nr}.nii") # orignal-B0-path

    # Load B0 map
    img = nib.load(orignal_B0_path)
    data = img.get_fdata()

    if flip_x:
        data = np.flip(data, axis=0)
    if flip_y:
        data = np.flip(data, axis=1)
    if flip_z:
        data = np.flip(data, axis=2)

    if swap_x_y:
        data = data.transpose(1,0,2)

    output_path_patient = f"{output_path}/B0_{augmented_patient_nr}.nii"
    nib.save(nib.Nifti1Image(data.astype(np.float32),img.affine), output_path_patient)
    print(f"Saved permuted B0: {augmented_patient_nr}")
