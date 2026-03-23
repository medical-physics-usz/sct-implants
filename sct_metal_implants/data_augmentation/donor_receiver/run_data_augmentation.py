import argparse
import os

import pandas as pd

from transformator import Transformator
from dicom_nifti_creator import DicomNiftiCreator
from visualizer import Visualizer
from patient_handler import PatientHandler


def augment_from_single_or_multiple_donors(donor_left_patient_nr, donor_right_patient_nr, receiver_patient_nr,
                                           path_donor, path_receiver, output_path, plot_path):
    """Augment a single receiver patient by inserting implants/voids from left/right donor(s).

    Parameters:
        donor_left_patient_nr (str|None) -- left donor patient ID
        donor_right_patient_nr (str|None) -- right donor patient ID
        receiver_patient_nr (str) -- receiver patient ID
        path_donor (str) -- donor data path
        path_receiver (str) -- receiver data path
        output_path (str) -- output directory
        plot_path (str) -- plot directory
    """

    # Structure and Void creation definition
    structure_name = "FemurHead"
    void_creation_method = "standard"

    # Load receiver patient
    receiver_patient = PatientHandler(path_receiver, receiver_patient_nr, structure_name)

    # Load donor patient(s)
    donor_patients = {}

    if donor_left_patient_nr:
        donor_patients["L"] = PatientHandler(path_donor, donor_left_patient_nr, structure_name)
    if donor_right_patient_nr:
        donor_patients["R"] = PatientHandler(path_donor, donor_right_patient_nr, structure_name)

    sides_to_transform = donor_patients.keys()

    # For each requested side, take implant/void from that side's donor and insert into the receiver
    for side in sides_to_transform:
        structure_to_transform = f"{structure_name}_{side}"
        donor_patient = donor_patients[side]

        # Calculate implant + MR void on the donor side
        donor_patient.calculate_implant_and_void(structure_to_transform, void_creation_method=void_creation_method)

        # Compute transform from donor->receiver based on the structure mask
        transformator = Transformator(donor_patient.structure_masks[structure_to_transform],
                                      receiver_patient.structure_masks[structure_to_transform])

        # Transform implant + void and set them on the receiver
        transformed_implant = transformator.transform(donor_patient.implant_masks[structure_to_transform])
        transformed_void = transformator.transform(donor_patient.mr_voids[structure_to_transform])
        transformed_implant_HU = transformator.transform(donor_patient.implants[structure_to_transform])

        receiver_patient.set_implant(transformed_implant, structure_to_transform)
        receiver_patient.set_mr_void(transformed_void, structure_to_transform)

        # Cut-off femur bone in receiver patient to imitate THR surgery
        receiver_femur_mask = receiver_patient.structure_masks[structure_to_transform]
        receiver_axis = transformator.compute_pca_alignment(receiver_femur_mask)
        print(receiver_axis.shape)

        receiver_patient.ct_handler.get_femur_head_mask(receiver_femur_mask)
        kept_mask, removed_mask = receiver_patient.ct_handler.cut_femur_by_fraction(receiver_femur_mask, receiver_axis)
        receiver_patient.ct.volume[kept_mask.astype(bool)] = 7

        # Apply overwrite for this side
        receiver_patient.overwrite_volumes(structure_to_transform)

        implant_mask = receiver_patient.implant_masks[structure_to_transform].astype(bool)
        print(donor_patient.ct.volume.shape, receiver_patient.ct.volume.shape)

        if donor_patient.ct.volume.shape != receiver_patient.ct.volume.shape:
            transformed_implant_HU = receiver_patient.pad_on_top(transformed_implant_HU, receiver_patient.ct.volume.shape)

        receiver_patient.ct.volume[implant_mask] = transformed_implant_HU[implant_mask]

    # Plot (receiver + each donor)
    visualizer = Visualizer()

    # Use a combined donor id in outputs, e.g. "Pat006_L_Pat123_R"
    donors_used = []
    if "L" in donor_patients:
        donors_used.append(f"{donor_left_patient_nr}_L")
    if "R" in donor_patients:
        donors_used.append(f"{donor_right_patient_nr}_R")
    donor_id_str = "_".join(donors_used) if donors_used else "NA"
    if donor_left_patient_nr == donor_right_patient_nr:
        donor_id_str = f"{donor_left_patient_nr}_L_R"

    # Receiver plot
    html_path_receiver = os.path.join(plot_path, "receiver", f"{receiver_patient_nr}_aug_by_{donor_id_str}.html")
    visualizer.plot_patient(receiver_patient, html_path_receiver)

    # Donor plots
    html_path_donor_L = os.path.join(plot_path, "donor", f"{donor_left_patient_nr}_L.html")
    html_path_donor_R = os.path.join(plot_path, "donor", f"{donor_right_patient_nr}_R.html")
    if "L" in donor_patients:
        visualizer.plot_patient(donor_patients["L"], html_path_donor_L)
    if "R" in donor_patients:
        visualizer.plot_patient(donor_patients["R"], html_path_donor_R)


    # Save to DICOM
    file_creator = DicomNiftiCreator(
        output_path,
        donor_id_str,
        receiver_patient_nr,
        structure_name,
        sides_to_transform
    )

    # Create DICOM and save
    file_creator.save_dicom(
        receiver_patient.ct.get_volume(),
        receiver_patient.ct.get_slices(),
        modality="CT"
    )

    file_creator.save_dicom(
        receiver_patient.mr.get_volume(),
        receiver_patient.mr.get_slices(),
        modality="MR_in"
    )

def augment_from_config(path_donor, path_receiver, output_path, plot_path, path_config):
    """
    Run augmentations for all patients defined in the Excel config.
    """

    # Make sure output/plot folders exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(os.path.join(plot_path, "donor"), exist_ok=True)
    os.makedirs(os.path.join(plot_path, "receiver"), exist_ok=True)

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

        augment_from_single_or_multiple_donors(donor_left_patient_nr, donor_right_patient_nr, receiver_patient_nr,
                                               path_donor, path_receiver, output_path, plot_path)

    print("Augmented all patients in config file!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run augmentation pipeline with configurable paths.")

    parser.add_argument("--donor_path", required=True,
                        help="Path to raw DICOM files of patients with implants (donor).")
    parser.add_argument("--receiver_path", required=True,
                        help="Path to raw DICOM files of patients without implants (receiver).")
    parser.add_argument("--output_path", required=True,
                        help="Path where augmented patient data will be saved.")
    parser.add_argument("--plot_path", required=True,
                        help="Path where plots will be saved.")
    parser.add_argument("--config_path", required=True,
                        help="Folder containing data_augmentation_configurations.xlsx")

    args = parser.parse_args()
    augment_from_config(args.donor_path, args.receiver_path, args.output_path, args.plot_path, args.config_path)
