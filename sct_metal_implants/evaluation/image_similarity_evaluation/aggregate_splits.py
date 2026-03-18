import argparse
import os
import pandas as pd
import numpy as np

def aggregate_mean_and_std_image_similarities(path_excel, results_path, columns_to_keep):
    """
    Aggregates mean ± std per split and adds an overall row
    computed from the means across splits.
    """

    df_experiments_info = pd.read_excel(
        os.path.join(path_excel, "experiments.xlsx"),
        engine="openpyxl"
    )

    for _, exp_row in df_experiments_info.iterrows():

        dataset = exp_row.Dataset
        model = exp_row.Model
        model_name = exp_row.Name

        split_means = []      # to compute overall mean/std later
        formatted_rows = []  # rows with "mean ± std"

        for i in range(1, 6):
            split_nr = f"split{i}"
            model_name_split = f"{model_name}_{split_nr}"

            results_excel_path = os.path.join(
                results_path,
                dataset,
                model,
                "test",
                model_name_split,
                "image_similarity_results.xlsx"
            )

            if not os.path.isfile(results_excel_path):
                print(results_excel_path + " does not exist")
                continue

            results_xls = pd.ExcelFile(results_excel_path)
            df_mean = pd.read_excel(results_xls, "Overall Mean")
            df_std = pd.read_excel(results_xls, "Overall Std")

            # keep only requested columns
            mean_vals = df_mean[columns_to_keep].iloc[0]
            std_vals = df_std[columns_to_keep].iloc[0]

            split_means.append(mean_vals.values)

            # format: mean ± std
            formatted = pd.Series(
                {
                    col: format_mean_std(mean_vals[col], std_vals[col], col)
                    for col in mean_vals.index
                },
                name=split_nr
            )

            formatted.name = split_nr
            formatted_rows.append(formatted)

        # ----- overall row (computed from split means) -----
        split_means = np.vstack(split_means)
        overall_mean = split_means.mean(axis=0)
        overall_std = split_means.std(axis=0, ddof=1)

        overall_formatted = [
            format_mean_std(m, s, col)
            for m, s, col in zip(overall_mean, overall_std, columns_to_keep)
        ]

        overall_row = pd.Series(
            overall_formatted,
            index=columns_to_keep,
            name="Overall (mean ± std across splits)"
        )

        # ----- final dataframe -----
        df_out = pd.DataFrame(formatted_rows)
        df_out = pd.concat([df_out, overall_row.to_frame().T])

        # ----- save -----
        out_path = os.path.join(
            results_path,
            dataset,
            model,
            "test",
            f"{model_name}_aggregated_image_similarity.xlsx"
        )

        df_out.to_excel(out_path, engine="openpyxl")

        print(f"Saved: {out_path}")

def format_mean_std(m, s, col):
    if col in {"FPR", "TPR"}:
        m = m * 100
        s = s * 100
        return f"{m:.2f}"
    return f"{m:.2f} ± {s:.2f}"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Aggregate image similartiy from splits")
    parser.add_argument('--path_excel', type=str, required=True, help="Path to excel file of experiments to process")
    parser.add_argument('--path_results', type=str, required=True, help="Path to results of sCTs")

    args = parser.parse_args()

    columns = [
        "MAE", "MAE_metal", "MAE_outside_metal", "MAE_metal_region", "MAE_outside_metal_region", "TPR", "FPR"
    ]

    # Run Convertion
    aggregate_mean_and_std_image_similarities(args.path_excel, args.path_results, columns)
