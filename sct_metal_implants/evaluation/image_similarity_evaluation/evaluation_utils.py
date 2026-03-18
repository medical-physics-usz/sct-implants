import pandas as pd


def build_summary_tables(result_test, result_metal):
    """
    From raw per-slice rows (result_test, result_metal), build:
      - combined_mean / combined_std (one row each)
      - mean/std per patient for normal + metal datasets
    """
    # === NORMAL DATASET ===
    columns_normal = [
        'Patient', 'MAE', 'MSE', 'PSNR', 'SSIM',
        'MAE_metal', 'MAE_outside_metal',
        'MAE_metal_region', 'MAE_outside_metal_region',
        'Num_outside_pixels', 'FPR'
    ]
    df_all_normal = pd.DataFrame(result_test, columns=columns_normal)

    mean_normal = df_all_normal.drop(columns=['Patient']).mean()
    std_normal  = df_all_normal.drop(columns=['Patient']).std()

    # === METAL DATASET ===
    columns_metal = ['Patient', 'MAE_IMPLANT_SLICES', 'TPR']
    df_all_metal = pd.DataFrame(result_metal, columns=columns_metal)

    mean_metal = df_all_metal.drop(columns=['Patient']).mean()
    std_metal  = df_all_metal.drop(columns=['Patient']).std()

    # === COMBINE MEAN/STDS ===
    combined_mean = pd.concat([mean_normal, mean_metal], axis=0).to_frame().T
    combined_std  = pd.concat([std_normal, std_metal], axis=0).to_frame().T

    # === PER-PATIENT RESULTS ===
    mean_per_patient_normal = df_all_normal.groupby('Patient').mean().reset_index()
    std_per_patient_normal  = df_all_normal.groupby('Patient').std().reset_index()

    mean_per_patient_metal = df_all_metal.groupby('Patient').mean().reset_index()
    std_per_patient_metal  = df_all_metal.groupby('Patient').std().reset_index()

    return {
        "combined_mean": combined_mean,
        "combined_std": combined_std,
        "mean_per_patient_normal": mean_per_patient_normal,
        "std_per_patient_normal": std_per_patient_normal,
        "mean_per_patient_metal": mean_per_patient_metal,
        "std_per_patient_metal": std_per_patient_metal,
        "df_all_normal": df_all_normal,
        "df_all_metal": df_all_metal,
    }


def save_results_to_excel(
    output_path,
    combined_mean,
    combined_std,
    mean_per_patient_normal,
    std_per_patient_normal,
    mean_per_patient_metal,
    std_per_patient_metal
):
    """
    Save evaluation results into a structured Excel file with multiple sheets.

    Parameters
    ----------
    output_path : str
        Path to the Excel file to create.
    combined_mean, combined_std : pd.DataFrame
        Overall mean and std results for the model.
    mean_per_patient_normal, std_per_patient_normal : pd.DataFrame
        Per-patient metrics (mean/std) for the normal evaluation.
    mean_per_patient_metal, std_per_patient_metal : pd.DataFrame
        Per-patient metrics (mean/std) for the metal evaluation.
    """

    with pd.ExcelWriter(output_path) as writer:
        combined_mean.to_excel(writer, sheet_name="Overall Mean", index=False)
        combined_std.to_excel(writer, sheet_name="Overall Std", index=False)
        mean_per_patient_normal.to_excel(writer, sheet_name="Per-Patient Normal Mean", index=False)
        std_per_patient_normal.to_excel(writer, sheet_name="Per-Patient Normal Std", index=False)
        mean_per_patient_metal.to_excel(writer, sheet_name="Per-Patient Metal Mean", index=False)
        std_per_patient_metal.to_excel(writer, sheet_name="Per-Patient Metal Std", index=False)
