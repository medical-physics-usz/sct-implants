import argparse
import os
import pandas as pd

from create_undistorted_mr import create_undistorted_mr
from sct_metal_implants.data_augmentation.physics_guided.create_distorted_mr import create_distorted_mr
from sct_metal_implants.data_augmentation.physics_guided.permute_B0_maps import create_permuted_B0_map


def augment_from_config(path_donor, path_receiver, output_path, path_config):
    """
    Run augmentations for all patients defined in the Excel config.
    """

    # Make sure output folders exist
    os.makedirs(output_path, exist_ok=True)
    output_path_undistorted = os.path.join(output_path, "undistorted_MR_in")
    os.makedirs(output_path_undistorted, exist_ok=True)
    output_path_permuted_B0 = os.path.join(output_path, "permuted_B0_maps")
    os.makedirs(output_path_permuted_B0, exist_ok=True)
    output_path_distorted = os.path.join(output_path, "distorted_MR_in")
    os.makedirs(output_path_distorted, exist_ok=True)

    # Read excel with augmentation configurations
    excel_configs = os.path.join(path_config, "data_augmentation_configurations.xlsx")
    df_configs = pd.read_excel(excel_configs, engine="openpyxl")
    cols = ["DonorPatientLeft", "DonorPatientRight", "ReceiverPatient"]
    df_configs[cols] = df_configs[cols].astype("object")
    df_configs = df_configs.where(pd.notna(df_configs), None) #make sure empty cells "None" and not np.nan (float)
    df_configs = df_configs.sort_values("ReceiverPatient")

    for index, line in df_configs.iterrows():
        donor_left_patient_nr = line.DonorPatientLeft
        donor_right_patient_nr = line.DonorPatientRight
        receiver_patient_nr = line.ReceiverPatient

        print(donor_left_patient_nr, donor_right_patient_nr, receiver_patient_nr)

        # Use a combined donor id in outputs, e.g. "Pat006_L_Pat123_R"
        donors_used = []
        if donor_left_patient_nr:
            donors_used.append(f"{donor_left_patient_nr}_L")
        if donor_right_patient_nr:
            donors_used.append(f"{donor_right_patient_nr}_R")
        donor_id_str = "_".join(donors_used) if donors_used else "NA"
        if donor_left_patient_nr == donor_right_patient_nr:
            donor_id_str = f"{donor_left_patient_nr}_L_R"

        augmented_patient_nr = f"{receiver_patient_nr}_aug_by_{donor_id_str}"
        print(augmented_patient_nr)

        # Create undistorted MR
        create_undistorted_mr(augmented_patient_nr, path_donor, output_path_undistorted)

        # Permute off-frequency map
        create_permuted_B0_map(augmented_patient_nr, output_path_permuted_B0, flip_x=False, flip_y=False, flip_z=False, swap_x_y=True)

        # Augment MR -> distorted MR
        create_distorted_mr(augmented_patient_nr, output_path_undistorted, output_path_permuted_B0, path_receiver, output_path_distorted)

    print("Augmented all patients in config file!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run physics-guided augmentation pipeline with configurable paths.")

    parser.add_argument("--donor_path", required=True,
                        help="Path to raw DICOM files of patients with implants (donor).")
    parser.add_argument("--receiver_path", required=True,
                        help="Path to raw DICOM files of patients without implants (receiver).")
    parser.add_argument("--output_path", required=True,
                        help="Path where augmented patient data will be saved.")
    parser.add_argument("--config_path", required=True,
                        help="Folder containing data_augmentation_configurations.xlsx")

    args = parser.parse_args()
    augment_from_config(args.donor_path, args.receiver_path, args.output_path, args.config_path)