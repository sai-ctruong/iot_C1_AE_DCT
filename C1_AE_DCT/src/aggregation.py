"""Subject-Level Aggregation and Paired Comparison module for C1_AE_DCT (TASK 21).

Steps:
1. Aggregate per subject x method x db x channel over valid windows (mean, median, p90, invalid_rate).
2. Aggregate across 15 subjects with EQUAL weighting (avoiding window-count bias).
3. Compute paired comparison (delta_s = AE_metric - DCT_metric).
4. Export summary_by_subject.csv, paired_comparison.csv, and overall_summary.csv.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from src.results_schema import validate_results_schema


def aggregate_by_subject(
    results_df_or_list: Union[Any, List[Dict[str, Any]]]
) -> Any:
    """
    Step 1: Compute subject-level statistics over valid windows.

    Group keys: (subject, method, db, channel)
    Metrics computed per group:
    - mean, median, p90 for PRD, PRDN, RMSE over valid windows
    - invalid_rate for PRD and PRDN
    """
    validate_results_schema(results_df_or_list)

    if pd is not None and isinstance(results_df_or_list, pd.DataFrame):
        df = results_df_or_list.copy()
    else:
        records = results_df_or_list if isinstance(results_df_or_list, list) else results_df_or_list.to_dict("records")
        df = pd.DataFrame(records) if pd is not None else records

    if pd is not None and isinstance(df, pd.DataFrame):
        # Ensure budget column exists
        if "budget" not in df.columns:
            df["budget"] = df["db"]

        group_cols = ["subject", "method", "db", "budget", "channel"]
        summary_rows = []

        for keys, group in df.groupby(group_cols):
            subj, method, db, budget, ch = keys
            n_total = len(group)

            # Valid PRD subset
            if "valid_prd" in group.columns:
                valid_prd_mask = group["valid_prd"] & (~group["PRD"].isna())
            else:
                valid_prd_mask = group["metric_valid"] & (~group["PRD"].isna())

            prd_vals = group.loc[valid_prd_mask, "PRD"].values
            n_valid_prd = len(prd_vals)
            invalid_rate_prd = (n_total - n_valid_prd) / n_total if n_total > 0 else 0.0

            # Valid PRDN subset
            if "valid_prdn" in group.columns:
                valid_prdn_mask = group["valid_prdn"] & (~group["PRDN"].isna())
            else:
                valid_prdn_mask = group["metric_valid"] & (~group["PRDN"].isna())

            prdn_vals = group.loc[valid_prdn_mask, "PRDN"].values
            n_valid_prdn = len(prdn_vals)
            invalid_rate_prdn = (n_total - n_valid_prdn) / n_total if n_total > 0 else 0.0

            # Valid RMSE subset
            rmse_vals = group.loc[~group["RMSE"].isna(), "RMSE"].values

            row = {
                "subject": subj,
                "method": method,
                "db": db,
                "budget": budget,
                "channel": ch,
                "n_total_windows": n_total,
                "n_valid_prd": n_valid_prd,
                "n_valid_prdn": n_valid_prdn,
                # PRD metrics
                "prd_mean": float(np.mean(prd_vals)) if n_valid_prd > 0 else np.nan,
                "prd_median": float(np.median(prd_vals)) if n_valid_prd > 0 else np.nan,
                "prd_p90": float(np.percentile(prd_vals, 90)) if n_valid_prd > 0 else np.nan,
                "invalid_rate_prd": float(invalid_rate_prd),
                # PRDN metrics
                "prdn_mean": float(np.mean(prdn_vals)) if n_valid_prdn > 0 else np.nan,
                "prdn_median": float(np.median(prdn_vals)) if n_valid_prdn > 0 else np.nan,
                "prdn_p90": float(np.percentile(prdn_vals, 90)) if n_valid_prdn > 0 else np.nan,
                "invalid_rate_prdn": float(invalid_rate_prdn),
                # RMSE metrics
                "rmse_mean": float(np.mean(rmse_vals)) if len(rmse_vals) > 0 else np.nan,
                "rmse_median": float(np.median(rmse_vals)) if len(rmse_vals) > 0 else np.nan,
                "rmse_p90": float(np.percentile(rmse_vals, 90)) if len(rmse_vals) > 0 else np.nan,
            }
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    else:
        # Fallback pure python dict group processing
        groups = {}
        for row in df:
            db_val = int(row["db"])
            budget_val = int(row.get("budget", db_val))
            k = (str(row["subject"]), str(row["method"]), db_val, budget_val, str(row["channel"]))
            if k not in groups:
                groups[k] = []
            groups[k].append(row)

        summary_rows = []
        for (subj, method, db, budget, ch), group_rows in groups.items():
            n_total = len(group_rows)
            prd_vals = []
            prdn_vals = []
            rmse_vals = []

            for r in group_rows:
                is_v_prd = str(r.get("valid_prd", r.get("metric_valid", "True"))).lower() in ["true", "1"]
                is_v_prdn = str(r.get("valid_prdn", r.get("metric_valid", "True"))).lower() in ["true", "1"]

                val_prd = float(r["PRD"]) if r["PRD"] is not None else float("nan")
                val_prdn = float(r["PRDN"]) if r["PRDN"] is not None else float("nan")
                val_rmse = float(r["RMSE"]) if r["RMSE"] is not None else float("nan")

                if is_v_prd and not np.isnan(val_prd):
                    prd_vals.append(val_prd)
                if is_v_prdn and not np.isnan(val_prdn):
                    prdn_vals.append(val_prdn)
                if not np.isnan(val_rmse):
                    rmse_vals.append(val_rmse)

            invalid_rate_prd = (n_total - len(prd_vals)) / n_total if n_total > 0 else 0.0
            invalid_rate_prdn = (n_total - len(prdn_vals)) / n_total if n_total > 0 else 0.0

            s_row = {
                "subject": subj,
                "method": method,
                "db": db,
                "budget": budget,
                "channel": ch,
                "n_total_windows": n_total,
                "n_valid_prd": len(prd_vals),
                "n_valid_prdn": len(prdn_vals),
                "prd_mean": float(np.mean(prd_vals)) if prd_vals else float("nan"),
                "prd_median": float(np.median(prd_vals)) if prd_vals else float("nan"),
                "prd_p90": float(np.percentile(prd_vals, 90)) if prd_vals else float("nan"),
                "invalid_rate_prd": float(invalid_rate_prd),
                "prdn_mean": float(np.mean(prdn_vals)) if prdn_vals else float("nan"),
                "prdn_median": float(np.median(prdn_vals)) if prdn_vals else float("nan"),
                "prdn_p90": float(np.percentile(prdn_vals, 90)) if prdn_vals else float("nan"),
                "invalid_rate_prdn": float(invalid_rate_prdn),
                "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
                "rmse_median": float(np.median(rmse_vals)) if rmse_vals else float("nan"),
                "rmse_p90": float(np.percentile(rmse_vals, 90)) if rmse_vals else float("nan"),
            }
            summary_rows.append(s_row)

        return summary_rows


def compute_overall_summary(
    summary_by_subject: Any
) -> Any:
    """
    Step 2: Aggregate subject-level stats across all subjects with EQUAL weighting (1/15 per subject).

    Group keys: (method, db, channel)
    Computes average of subject-level mean, median, p90, and invalid_rate.
    """
    if pd is not None and isinstance(summary_by_subject, pd.DataFrame):
        df = summary_by_subject.copy()
        group_cols = ["method", "db", "budget", "channel"]
        overall_rows = []

        for keys, group in df.groupby(group_cols):
            method, db, budget, ch = keys
            n_subjects = len(group)

            row = {
                "method": method,
                "db": db,
                "budget": budget,
                "channel": ch,
                "n_subjects": n_subjects,

                # Equal-weighted averages across subjects
                "prd_mean": float(group["prd_mean"].mean(skipna=True)),
                "prd_median": float(group["prd_median"].mean(skipna=True)),
                "prd_p90": float(group["prd_p90"].mean(skipna=True)),
                "invalid_rate_prd": float(group["invalid_rate_prd"].mean(skipna=True)),

                "prdn_mean": float(group["prdn_mean"].mean(skipna=True)),
                "prdn_median": float(group["prdn_median"].mean(skipna=True)),
                "prdn_p90": float(group["prdn_p90"].mean(skipna=True)),
                "invalid_rate_prdn": float(group["invalid_rate_prdn"].mean(skipna=True)),

                "rmse_mean": float(group["rmse_mean"].mean(skipna=True)),
                "rmse_median": float(group["rmse_median"].mean(skipna=True)),
                "rmse_p90": float(group["rmse_p90"].mean(skipna=True)),
            }
            overall_rows.append(row)

        return pd.DataFrame(overall_rows)

    else:
        # Fallback list processing
        records = summary_by_subject if isinstance(summary_by_subject, list) else summary_by_subject.to_dict("records")
        groups = {}
        for r in records:
            k = (r["method"], r["db"], r.get("budget", r["db"]), r["channel"])
            if k not in groups:
                groups[k] = []
            groups[k].append(r)

        overall_rows = []
        for (method, db, budget, ch), group_rows in groups.items():
            n_subjs = len(group_rows)
            row = {
                "method": method,
                "db": db,
                "budget": budget,
                "channel": ch,
                "n_subjects": n_subjs,

                "prd_mean": float(np.nanmean([r["prd_mean"] for r in group_rows])),
                "prd_median": float(np.nanmean([r["prd_median"] for r in group_rows])),
                "prd_p90": float(np.nanmean([r["prd_p90"] for r in group_rows])),
                "invalid_rate_prd": float(np.nanmean([r["invalid_rate_prd"] for r in group_rows])),

                "prdn_mean": float(np.nanmean([r["prdn_mean"] for r in group_rows])),
                "prdn_median": float(np.nanmean([r["prdn_median"] for r in group_rows])),
                "prdn_p90": float(np.nanmean([r["prdn_p90"] for r in group_rows])),
                "invalid_rate_prdn": float(np.nanmean([r["invalid_rate_prdn"] for r in group_rows])),

                "rmse_mean": float(np.nanmean([r["rmse_mean"] for r in group_rows])),
                "rmse_median": float(np.nanmean([r["rmse_median"] for r in group_rows])),
                "rmse_p90": float(np.nanmean([r["rmse_p90"] for r in group_rows])),
            }
            overall_rows.append(row)

        return overall_rows


def compute_paired_comparison(
    summary_by_subject: Any
) -> Any:
    """
    Compute paired comparison AE vs DCT per subject x db x channel.

    Formulas:
    - delta_s = AE_metric_s - DCT_metric_s
    - delta_s < 0 indicates AE achieves lower error than DCT.
    """
    if pd is not None and isinstance(summary_by_subject, pd.DataFrame):
        df = summary_by_subject.copy()
        ae_df = df[df["method"] == "AE"].copy()
        dct_df = df[df["method"] == "DCT"].copy()

        join_keys = ["subject", "db", "budget", "channel"]
        merged = pd.merge(
            ae_df,
            dct_df,
            on=join_keys,
            suffixes=("_ae", "_dct")
        )

        # Compute delta_s = AE - DCT
        merged["delta_prd_mean"] = merged["prd_mean_ae"] - merged["prd_mean_dct"]
        merged["delta_prd_median"] = merged["prd_median_ae"] - merged["prd_median_dct"]
        merged["delta_prdn_mean"] = merged["prdn_mean_ae"] - merged["prdn_mean_dct"]
        merged["delta_prdn_median"] = merged["prdn_median_ae"] - merged["prdn_median_dct"]
        merged["delta_rmse_mean"] = merged["rmse_mean_ae"] - merged["rmse_mean_dct"]
        merged["delta_rmse_median"] = merged["rmse_median_ae"] - merged["rmse_median_dct"]

        # Boolean flags: delta < 0 => AE outperforms DCT (lower distortion)
        merged["ae_wins_prd"] = merged["delta_prd_mean"] < 0
        merged["ae_wins_prdn"] = merged["delta_prdn_mean"] < 0
        merged["ae_wins_rmse"] = merged["delta_rmse_mean"] < 0

        cols = [
            "subject", "db", "budget", "channel",
            "prd_mean_ae", "prd_mean_dct", "delta_prd_mean", "ae_wins_prd",
            "prdn_mean_ae", "prdn_mean_dct", "delta_prdn_mean", "ae_wins_prdn",
            "rmse_mean_ae", "rmse_mean_dct", "delta_rmse_mean", "ae_wins_rmse",
            "delta_prd_median", "delta_prdn_median", "delta_rmse_median"
        ]
        return merged[cols]

    else:
        records = summary_by_subject if isinstance(summary_by_subject, list) else summary_by_subject.to_dict("records")
        ae_map = {(r["subject"], r["db"], r.get("budget", r["db"]), r["channel"]): r for r in records if r["method"] == "AE"}
        dct_map = {(r["subject"], r["db"], r.get("budget", r["db"]), r["channel"]): r for r in records if r["method"] == "DCT"}

        paired_rows = []
        for key, r_ae in ae_map.items():
            if key in dct_map:
                r_dct = dct_map[key]
                subj, db, budget, ch = key

                d_prd = r_ae["prd_mean"] - r_dct["prd_mean"]
                d_prdn = r_ae["prdn_mean"] - r_dct["prdn_mean"]
                d_rmse = r_ae["rmse_mean"] - r_dct["rmse_mean"]

                p_row = {
                    "subject": subj,
                    "db": db,
                    "budget": budget,
                    "channel": ch,
                    "prd_mean_ae": r_ae["prd_mean"],
                    "prd_mean_dct": r_dct["prd_mean"],
                    "delta_prd_mean": d_prd,
                    "ae_wins_prd": bool(d_prd < 0),
                    "prdn_mean_ae": r_ae["prdn_mean"],
                    "prdn_mean_dct": r_dct["prdn_mean"],
                    "delta_prdn_mean": d_prdn,
                    "ae_wins_prdn": bool(d_prdn < 0),
                    "rmse_mean_ae": r_ae["rmse_mean"],
                    "rmse_mean_dct": r_dct["rmse_mean"],
                    "delta_rmse_mean": d_rmse,
                    "ae_wins_rmse": bool(d_rmse < 0),
                    "delta_prd_median": r_ae["prd_median"] - r_dct["prd_median"],
                    "delta_prdn_median": r_ae["prdn_median"] - r_dct["prdn_median"],
                    "delta_rmse_median": r_ae["rmse_median"] - r_dct["rmse_median"],
                }
                paired_rows.append(p_row)

        return paired_rows


import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np


def save_records_to_csv(records: Union[List[Dict[str, Any]], Any], filepath: Path) -> None:
    """Save records to CSV using pandas if available, or native csv module fallback."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None and isinstance(records, pd.DataFrame):
        records.to_csv(filepath, index=False)
    else:
        rec_list = records if isinstance(records, list) else (records.to_dict("records") if hasattr(records, "to_dict") else [])
        if not rec_list:
            return
        fieldnames = list(rec_list[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rec_list)


def export_aggregation_reports(
    results_input: Union[Any, List[Dict[str, Any]], str, Path],
    output_dir: Union[str, Path] = "results"
) -> Dict[str, Path]:
    """
    Full TASK 21 Aggregation pipeline:
    1. Reads/receives detailed window-level results.
    2. Computes summary_by_subject.
    3. Computes paired_comparison.
    4. Computes overall_summary.
    5. Saves CSV files to output_dir.

    Returns:
    --------
    output_paths : Dict[str, Path]
        Paths to summary_by_subject.csv, paired_comparison.csv, and overall_summary.csv.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load results if input is file path
    if isinstance(results_input, (str, Path)):
        file_p = Path(results_input)
        if pd is not None:
            df_results = pd.read_csv(file_p)
        else:
            with open(file_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                df_results = list(reader)
    else:
        df_results = results_input

    # Step 1: Subject aggregation
    df_subject = aggregate_by_subject(df_results)

    # Step 2: Overall aggregation across subjects
    df_overall = compute_overall_summary(df_subject)

    # Step 3: Paired comparison AE vs DCT
    df_paired = compute_paired_comparison(df_subject)

    # File output paths
    path_subject = out_path / "summary_by_subject.csv"
    path_paired = out_path / "paired_comparison.csv"
    path_overall = out_path / "overall_summary.csv"

    save_records_to_csv(df_subject, path_subject)
    save_records_to_csv(df_paired, path_paired)
    save_records_to_csv(df_overall, path_overall)

    print(f"[TASK 21] Aggregation reports exported successfully to {out_path}:")
    print(f"  - {path_subject.name}")
    print(f"  - {path_paired.name}")
    print(f"  - {path_overall.name}")

    return {
        "summary_by_subject": path_subject,
        "paired_comparison": path_paired,
        "overall_summary": path_overall,
    }

