from transformator import Transformator
from dicom_nifti_creator import DicomNiftiCreator
from visualizer import Visualizer
from patient_handler import  PatientHandler


def augment_single_donor(path_donor, path_receiver, output_path, plot_path, donor_patient_nr, receiver_patient_nr, sides_to_transform):

    # Structure of Interest
    structure_name = "FemurHead"

    # Other variables
    void_creation_method = "standard"

    # Load Data
    donor_patient = PatientHandler(path_donor, donor_patient_nr, structure_name)
    receiver_patient = PatientHandler(path_receiver, receiver_patient_nr, structure_name)

    # For all sides (L, R)
    for side in sides_to_transform:
        structure_to_transform = f"{structure_name}_{side}"

        # Calculate Donor Implant and Void
        donor_patient.calculate_implant_and_void(structure_to_transform, void_creation_method=void_creation_method)

        # Transform Implant and Void from Donor to Recipient
        transformator = Transformator(donor_patient.structure_masks[structure_to_transform],
                                      receiver_patient.structure_masks[structure_to_transform])

        transformed_implant = transformator.transform(donor_patient.implant_masks[structure_to_transform])
        transformed_void = transformator.transform(donor_patient.mr_voids[structure_to_transform])

        # Set Transformed Implant and Void in receiver patient
        receiver_patient.set_implant(transformed_implant, structure_to_transform)
        receiver_patient.set_mr_void(transformed_void, structure_to_transform)

        # Overwrite Receiver Voxels with Transformed Implant / Void
        receiver_patient.overwrite_volumes(structure_to_transform)

    # Plot femur and implant in donor and receiver (augmented)
    html_path_donor = plot_path + f"/donor/Pat{donor_patient_nr}.html"
    html_path_receiver = plot_path + f"/receiver/Pat{receiver_patient_nr}_aug_by_Pat{donor_patient_nr}.html"

    visualizer = Visualizer()
    visualizer.plot_patient(receiver_patient, html_path_receiver)
    visualizer.plot_patient(donor_patient, html_path_donor)

    # Save to NIFTI and DICOM
    file_creator = DicomNiftiCreator(output_path,
                                     donor_patient_nr,
                                     receiver_patient_nr,
                                     structure_name,
                                     sides_to_transform)

    # Create DICOM
    file_creator.save_dicom(receiver_patient.ct.get_volume(),
                            receiver_patient.ct.get_slices(),
                            modality="CT")

    file_creator.save_dicom(receiver_patient.mr.get_volume(),
                            receiver_patient.mr.get_slices(),
                            modality="MR_in")

