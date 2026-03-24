# import the libraries
import argparse

import matplotlib.pyplot as plt
import os
import numpy as np
import pydicom
import pymedphys
import gc
import pandas as pd

# Constants declaration
PRESCRIBED_DOSE = 40  # Gy
DOSE_CUTOFF_PERCENTAGES = [0.9, 0.5]

GAMMA_OPTIONS = {'interp_fraction': 10,  # Should be 10 or more
                 'max_gamma': 1.5,
                 'random_subset': None,  # Can be used to get quick pass rates
                 'local_gamma': True,  # Change to false for global gamma
                 'ram_available': 2*2**29  # 1/2 GB = 2**29
}

DOSE_DIFFERENCE_CRITERION = 1 # in [%] (measured dose can differ from planned dose by up to 1 %)
DISTANCE_TO_AGREEMENT = 1 # in [mm] (measured dose point can be up to 1 mm away spatially from planned point)


def perform_gamma_analysis(RTDose_path, output_path, MR_path, path_excel):

    # Load excel with patient data
    df_patient_info = pd.read_excel(os.path.join(path_excel, "patient_info.xlsx"), engine="openpyxl")
    file_name = "RTDose_1_physicalDose.dcm"

    # Create results directory
    os.makedirs(output_path, exist_ok=True)

    # Store results
    results = []

    # Loop through patients
    for pat_idx, pat_row in df_patient_info.iterrows():

        patient_nr = pat_row.StudyID
        split = pat_row.Split
        start_slice = pat_row.SliceStart

        if not os.path.isdir(os.path.join(RTDose_path, "rCT", patient_nr)):
            continue

        print(f"Processing {patient_nr}")

        # Path to RTDose files of rCT and sCT
        path_rCT = os.path.join(RTDose_path, "rCT", patient_nr, file_name)
        path_sCT = os.path.join(RTDose_path, "sCT", patient_nr, file_name)

        # Load Reference and Evaluation axes and doses
        axes_reference, dose_reference, axes_evaluation, dose_evaluation, z_slice_dmax = load_reference_and_evaluation(path_sCT, path_rCT)

        # Load corresponding MR slice
        MR_slice = load_MR_slice(MR_path, z_slice_dmax, patient_nr, start_slice)

        # Get dmax and calculate dose cutoff for gamma analysis at 90% and 50%
        dmax = np.max(dose_reference)

        # Prepare one output row for this patient
        row = {"patient_nr": patient_nr, "split": split}
        metric_cols = []

        for percentage in DOSE_CUTOFF_PERCENTAGES:
            label = f"D{int(percentage * 100)}"  # 0.9 -> D90, 0.5 -> D50
            print(f"{label} dose cutoff")

            dose_cutoff_threshold = calculate_dose_cutoff(percentage, dmax, PRESCRIBED_DOSE)

            # Perform gamma analysis 1% 1mm
            gamma = calculate_gamma(1, 1, dose_cutoff_threshold, axes_reference, dose_reference, axes_evaluation, dose_evaluation)
            passing_rate, failing_rate = calculate_passing_rate(gamma)
            print(passing_rate, failing_rate)

            # Store passing-rate
            metric_col = f"{label}_{DOSE_DIFFERENCE_CRITERION}{DISTANCE_TO_AGREEMENT}"
            metric_cols.append(metric_col)
            row[metric_col] = passing_rate

            # Create Gamma Plot for patient and Dose Cutoff
            create_plot(dose_reference, dose_evaluation, gamma, percentage, patient_nr, output_path, MR_slice,
                        threshold=DOSE_DIFFERENCE_CRITERION, distance=DISTANCE_TO_AGREEMENT)

        results.append(row)

    save_results_to_excel(results, output_path, metric_cols)

def save_results_to_excel(results, output_path, metric_cols):

    os.makedirs(output_path, exist_ok=True)

    # Build dataframe
    if isinstance(results[0], dict):
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame(results, columns=["patient_nr", "split"] + metric_cols)

    # -------- Save detailed results --------
    detailed_path = os.path.join(output_path, "gamma_results.xlsx")
    df.to_excel(detailed_path, index=False)
    print(f"Saved results to: {detailed_path}")

    # -------- Summary per split --------
    grouped = df.groupby("split")[metric_cols]

    means = grouped.mean()
    stds = grouped.std(ddof=1)

    summary = pd.DataFrame(index=means.index)

    for col in metric_cols:
        summary[col] = (
            means[col].round(3).astype(str)
            + " ± "
            + stds[col].round(3).astype(str)
        )

    # -------- Overall row (across all patients) --------
    overall_row = {}
    for col in metric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=1)
        overall_row[col] = f"{mean_val:.3f} ± {std_val:.3f}"

    summary.loc["ALL"] = overall_row

    summary = summary.reset_index()

    summary_path = os.path.join(output_path, "gamma_results_summary_by_split.xlsx")
    summary.to_excel(summary_path, index=False)

    print(f"Saved split summary to: {summary_path}")

def calculate_dose_cutoff(percentage, dmax, prescribed_dose):
    return prescribed_dose * percentage / dmax * 100

def get_gamma_options(dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff):

    gamma_options = GAMMA_OPTIONS.copy()
    gamma_options["dose_percent_threshold"] = dose_percent_threshold
    gamma_options["distance_mm_threshold"] = distance_mm_threshold
    gamma_options["lower_percent_dose_cutoff"] = lower_percent_dose_cutoff

    return gamma_options

def calculate_gamma(dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff,
                    axes_reference, dose_reference, axes_evaluation, dose_evaluation):

    gamma_options = get_gamma_options(dose_percent_threshold, distance_mm_threshold, lower_percent_dose_cutoff)

    gamma = pymedphys.gamma(
        axes_reference, dose_reference,
        axes_evaluation, dose_evaluation,
        **gamma_options)

    return gamma

def calculate_passing_rate(gamma):

    gamma_nonan = gamma[np.logical_not(np.isnan(gamma))]

    N_pass = len(gamma_nonan[np.where(gamma_nonan < 1)])
    N_fail = len(gamma_nonan[np.where(gamma_nonan >= 1)])
    N_tot = len(gamma_nonan)

    passing_rate = round(100.*N_pass/N_tot, 2)
    failing_rate = round(100.*N_fail/N_tot, 2)

    return passing_rate, failing_rate

def load_reference_and_evaluation(path_sCT, path_rCT, n_slices_to_analyze=40):
    reference = pydicom.dcmread(path_sCT, force=True)
    evaluation = pydicom.dcmread(path_rCT, force=True)

    axes_reference, dose_reference = pymedphys.dicom.zyx_and_dose_from_dataset(reference)
    axes_evaluation, dose_evaluation = pymedphys.dicom.zyx_and_dose_from_dataset(evaluation)

    index_dmax = np.unravel_index(np.argmax(dose_reference, axis=None), dose_reference.shape)
    z_slice_dmax = index_dmax[0]
    z_slice_start = max(0, z_slice_dmax - int(0.5 * n_slices_to_analyze))
    z_slice_stopp = min(z_slice_dmax + int(0.5 * n_slices_to_analyze), dose_reference.shape[0])

    axes_reference = (axes_reference[0][z_slice_start:z_slice_stopp], axes_reference[1], axes_reference[2])
    axes_evaluation = (axes_evaluation[0][z_slice_start:z_slice_stopp], axes_evaluation[1], axes_evaluation[2])

    dose_reference = dose_reference[z_slice_start:z_slice_stopp, :, :].copy()
    dose_evaluation = dose_evaluation[z_slice_start:z_slice_stopp, :, :].copy()

    gc.collect()  # optional

    return axes_reference, dose_reference, axes_evaluation, dose_evaluation, z_slice_dmax


def load_MR_slice(path, z_slice, patient_nr, start_slice):

    mr_dir = os.path.join(path, patient_nr, "MR_in")

    files = []
    for f in os.listdir(mr_dir):
        fp = os.path.join(mr_dir, f)
        if os.path.isdir(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            inst = getattr(ds, "InstanceNumber", None)
            if inst is None:
                continue
            files.append((int(inst), fp))
        except Exception:
            continue

    if not files:
        raise RuntimeError(f"No readable DICOMs with InstanceNumber found in {mr_dir}")

    files.sort(key=lambda x: x[0])

    slice_idx = z_slice + start_slice
    _, dcm_path = files[slice_idx]

    ds = pydicom.dcmread(dcm_path, force=True)
    return ds.pixel_array.astype(np.float32)

def create_plot(
    dose_reference,
    dose_evaluation,
    gamma22,
    threshold_percentage,
    patient_nr,
    output_path,
    MR_slice,
    dose_alpha=0.75, # transparency of the dose overlay on MR
    mr_vmin=-100,
    mr_vmax=600,
    dose_cmap="coolwarm",
    mr_cmap="gray",
    threshold=1,
    distance=1
):
    fig = plt.figure(figsize=(10, 9))
    columns = 2
    rows = 2

    limit_plot = [0, 240, 0, 320]
    slice_plot = np.unravel_index(np.argmax(dose_reference), dose_reference.shape)[0]

    # Cropped views for plotting
    mr_crop = MR_slice[limit_plot[0]:limit_plot[1], limit_plot[2]:limit_plot[3]]
    dose_ref_crop = dose_reference[slice_plot][limit_plot[0]:limit_plot[1], limit_plot[2]:limit_plot[3]]
    dose_eval_crop = dose_evaluation[slice_plot][limit_plot[0]:limit_plot[1], limit_plot[2]:limit_plot[3]]

    # Use same dose scale for both overlays
    dose_vmin = 0.0
    dose_vmax = float(np.max(dose_reference[slice_plot]))

    # --- 1) MR + Dose on sCT (blended)
    ax2 = fig.add_subplot(rows, columns, 1)
    ax2.imshow(mr_crop, cmap=mr_cmap, vmin=mr_vmin, vmax=mr_vmax)
    im_dose_ref = ax2.imshow(
        dose_ref_crop,
        cmap=dose_cmap,
        vmin=dose_vmin,
        vmax=dose_vmax,
        alpha=dose_alpha
    )
    plt.colorbar(im_dose_ref, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis("off")
    ax2.set_title("MR + Dose on sCT [Gy]", fontsize=18)

    # --- 2) MR + Dose on CT (blended)
    ax3 = fig.add_subplot(rows, columns, 2)
    ax3.imshow(mr_crop, cmap=mr_cmap, vmin=mr_vmin, vmax=mr_vmax)
    im_dose_eval = ax3.imshow(
        dose_eval_crop,
        cmap=dose_cmap,
        vmin=dose_vmin,
        vmax=dose_vmax,
        alpha=dose_alpha
    )
    plt.colorbar(im_dose_eval, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis("off")
    ax3.set_title("MR + Dose on CT [Gy]", fontsize=18)

    # --- 3) Relative dose difference [%] (unchanged)
    ax4 = fig.add_subplot(rows, columns, 3)
    im_rDiff = 100.0 * (dose_evaluation[slice_plot] - dose_reference[slice_plot]) / np.amax(dose_reference[slice_plot])
    im_rDiff_crop = im_rDiff[limit_plot[0]:limit_plot[1], limit_plot[2]:limit_plot[3]]
    im_diff = ax4.imshow(im_rDiff_crop, cmap="bwr", vmin=-2.5, vmax=2.5)
    plt.colorbar(im_diff, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis("off")
    ax4.set_title("Relative dose difference [%]", fontsize=18)

    # --- 4) Gamma X%/Xmm
    ax6 = fig.add_subplot(rows, columns, 4)
    g22_crop = gamma22[slice_plot][limit_plot[0]:limit_plot[1], limit_plot[2]:limit_plot[3]]
    im_g22 = ax6.imshow(g22_crop, cmap="bwr")
    plt.colorbar(im_g22, ax=ax6, fraction=0.046, pad=0.04)
    ax6.axis("off")

    ax2.set_title("MR + Dose on sCT [Gy]", fontsize=18, pad=18)
    ax3.set_title("MR + Dose on rCT [Gy]", fontsize=18, pad=18)
    ax4.set_title("Relative dose difference [%]", fontsize=18, pad=18)
    ax6.set_title(f"Local gamma map {threshold}% / {distance}mm", fontsize=18, pad=18)

    plt.tight_layout(pad=0)

    plt.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.90,
        bottom=0.05,
        wspace=0.10,
        hspace=0.25
    )

    plt.savefig(f"{output_path}/{patient_nr}_Gamma_{str(int(100*threshold_percentage))}_with_MR_{threshold}_{distance}.png", dpi=200)
    plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform Gamma Analysis and RTDose Files.")
    parser.add_argument('--path_RTDose_files', type=str, required=True, help="Path to folder containing RTDose files calculating in dosimetric evaluation.")
    parser.add_argument('--path_output', type=str, required=True, help="Path to save output of Gamma Analysis.")
    parser.add_argument('--path_original_dicom_data', type=str, required=True, help="Path of original DICOM data (for MR image).")
    parser.add_argument('--path_excel', type=str, required=True, help="Path to folder containing excel patient_info.xlsx")

    args = parser.parse_args()

    # Run Gamma Analysis
    perform_gamma_analysis(args.path_RTDose_files, args.path_output, args.path_original_dicom_data, args.path_excel)