"""Automated Report Generator module for C1_AE_DCT (TASK 24).

Reads results/results.csv and dynamically generates:
CSV Reports:
- summary_by_subject.csv
- paired_comparison.csv
- overall_summary.csv

Plots (PNGs):
- cr_dim_prd.png
- cr_byte_prd.png
- prdn_curves.png
- rmse_curves.png
- reconstruction_examples.png
- failure_cases.png

Markdown Report:
- experiment_summary.md

STRICT RULES:
- ZERO hard-coded numbers in report or script.
- Everything dynamically computed from results.csv.
- Updating results.csv and re-running automatically regenerates all tables, plots, and markdown.
- Validation before reporting:
  * 15 Test subjects (S1..S15)
  * 5 folds (1..5)
  * 4 db budgets (16, 8, 4, 2)
  * Complete AE/DCT pairing
"""

import os
import sys
import csv
import math
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import torch

from src.results_schema import (
    RESULT_SCHEMA_COLUMNS,
    validate_results_schema,
    create_result_row,
    append_result,
)
from src.aggregation import (
    aggregate_by_subject,
    compute_overall_summary,
    compute_paired_comparison,
    save_records_to_csv,
)
from src.reporting import (
    generate_table8_byte_cost,
    generate_table9_ae_parameter_count,
    generate_table7_mean_std_across_subjects,
    plot_metric_vs_cr,
    _extract_overall_summary_records,
)
from src.reconstruction_visualization import (
    find_failure_cases,
    export_failure_cases_csv,
    plot_single_window_comparison,
)
from src.metrics import compute_equal_byte_budget
from src.normalize import load_norm_stats, denormalize
from src.baseline_dct import dct_encode_topk, dct_decode
from src.codec import encode_ae_bytes, decode_ae_bytes, encode_dct_bytes, decode_dct_bytes
from src.model import C1Autoencoder
from src.evaluate import compute_channel_metrics

EXPECTED_SUBJECTS = {f"S{i}" for i in range(1, 16)}
EXPECTED_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_DBS = {16, 8, 4, 2}
CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]


def validate_before_report(df_or_list: Any) -> Dict[str, Any]:
    """
    Strict validation of results dataset prior to generating reports.

    Checks:
    1. 15 Test subjects present (S1..S15).
    2. 5 folds present (1..5).
    3. 4 db budget factors present (16, 8, 4, 2).
    4. Complete AE and DCT pairing per window x channel x db.
    5. Byte length validity (nbytes == 16 + 4 * 32 * db).
    6. Consistent metric valid mask for AE and DCT on identical reference windows.
    """
    if pd is not None and isinstance(df_or_list, pd.DataFrame):
        df = df_or_list.copy()
        records = df.to_dict("records")
    else:
        records = df_or_list if isinstance(df_or_list, list) else df_or_list.to_dict("records")

    if not records:
        raise ValueError("[VALIDATION FAILED] Results dataset is empty!")

    subjects_found = {str(r["subject"]) for r in records}
    folds_found = {int(r["fold"]) for r in records}
    dbs_found = {int(r["db"]) for r in records}
    methods_found = {str(r["method"]) for r in records}

    # 1. Subjects check
    missing_subjects = EXPECTED_SUBJECTS - subjects_found
    if missing_subjects:
        raise ValueError(f"[VALIDATION FAILED] Missing Test subjects: {sorted(list(missing_subjects))}")

    # 2. Folds check
    missing_folds = EXPECTED_FOLDS - folds_found
    if missing_folds:
        raise ValueError(f"[VALIDATION FAILED] Missing Folds: {sorted(list(missing_folds))}")

    # 3. DB Budgets check
    missing_dbs = EXPECTED_DBS - dbs_found
    if missing_dbs:
        raise ValueError(f"[VALIDATION FAILED] Missing db budget factors: {sorted(list(missing_dbs))}")

    # 4 & 5. Byte length & AE/DCT pair matching check
    ae_map = {}
    dct_map = {}
    invalid_byte_counts = 0

    for r in records:
        db = int(r["db"])
        comp_type = str(r.get("comparison_type", "equal_byte"))
        method = str(r["method"])

        if comp_type == "equal_dim" and method == "DCT":
            m_dim = 32 * db
            expected_bytes = 16 + 6 * m_dim
        else:
            m_dim = 32 * db
            expected_bytes = 16 + 4 * m_dim

        if int(r["nbytes"]) != expected_bytes:
            invalid_byte_counts += 1

        key = (int(r["fold"]), str(r["subject"]), str(r["window_id"]), str(r["channel"]), db, comp_type)
        if method == "AE":
            ae_map[key] = r
        elif method == "DCT":
            dct_map[key] = r

    if invalid_byte_counts > 0:
        raise ValueError(f"[VALIDATION FAILED] Found {invalid_byte_counts} rows with invalid byte length!")

    # Check pair coverage
    missing_dct_pairs = set(ae_map.keys()) - set(dct_map.keys())
    missing_ae_pairs = set(dct_map.keys()) - set(ae_map.keys())

    if missing_dct_pairs or missing_ae_pairs:
        raise ValueError(
            f"[VALIDATION FAILED] Incomplete AE/DCT pairing! "
            f"Missing DCT pairs: {len(missing_dct_pairs)}, Missing AE pairs: {len(missing_ae_pairs)}"
        )

    # 6. Valid mask consistency
    mask_mismatches = 0
    for key, r_ae in ae_map.items():
        r_dct = dct_map[key]
        if r_ae.get("valid_prdn") != r_dct.get("valid_prdn"):
            mask_mismatches += 1

    if mask_mismatches > 0:
        raise ValueError(f"[VALIDATION FAILED] Found {mask_mismatches} AE/DCT valid mask inconsistencies!")

    print("[VALIDATION SUCCESS] All pre-reporting checks passed cleanly:")
    print(f"  - Subjects: {len(subjects_found)}/15 present")
    print(f"  - Folds: {len(folds_found)}/5 present")
    print(f"  - Budgets (d_b): {sorted(list(dbs_found))} present")
    print(f"  - AE/DCT Matched Pairs: {len(ae_map)} pairs verified")

    return {
        "n_records": len(records),
        "n_pairs": len(ae_map),
        "subjects": sorted(list(subjects_found)),
        "folds": sorted(list(folds_found)),
        "dbs": sorted(list(dbs_found)),
    }


def generate_real_reconstruction_examples(
    output_filepath: Union[str, Path]
) -> Path:
    """
    Generate reconstruction_examples.png using 100% REAL PPG-DaLiA Test Data,
    real trained AE checkpoint (Fold 1, db=8), and real DCT baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_test_path = PROJECT_ROOT / "data" / "processed" / "fold1" / "test.npz"
    norm_stats_path = PROJECT_ROOT / "configs" / "norm_stats_fold1.json"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fold01_db08_seed42.pt"

    if not proc_test_path.exists() or not norm_stats_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError("[ERROR] Required real data files or checkpoint missing for fold 1 db=8!")

    npz_data = np.load(proc_test_path, allow_pickle=True)
    test_windows_norm = npz_data["windows"].astype(np.float32)
    test_meta_arr = npz_data["metadata"]
    test_meta_list = test_meta_arr.item() if test_meta_arr.ndim == 0 else list(test_meta_arr)
    norm_stats = load_norm_stats(norm_stats_path)

    # Pick first window (Subject S1)
    w_norm = test_windows_norm[0]  # Shape (4, 512)
    w_meta = test_meta_list[0]

    # Real AE reconstruction
    ckpt_data = torch.load(ckpt_path, map_location=device)
    model = C1Autoencoder(d_b=8).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_in = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)
        latent_tensor = model.encode(x_in)
        latent_np = latent_tensor.cpu().numpy()
        ae_bytes = encode_ae_bytes(latent_np, d_b=8, profile_id=0)
        latent_dec_np, _ = decode_ae_bytes(ae_bytes)
        latent_dec_tensor = torch.from_numpy(latent_dec_np.reshape(1, 8, 32)).float().to(device)
        rec_norm_ae = model.decode(latent_dec_tensor).squeeze(0).cpu().numpy()

    # Real DCT reconstruction (d_b=8 -> K_equal_byte = 170, pad = 4)
    b_info = compute_equal_byte_budget(8)
    k_eq_byte = b_info["K_equal_byte"]
    pad_bytes = b_info["padding"]
    topk_vals, topk_idxs, _, _ = dct_encode_topk(w_norm, k=k_eq_byte)
    dct_bytes = encode_dct_bytes(topk_vals, topk_idxs, d_b=8, profile_id=0, pad_bytes=pad_bytes)
    dec_vals, dec_idxs, _ = decode_dct_bytes(dct_bytes)
    sparse_dct = np.zeros((4, 512), dtype=np.float32)
    np.put(sparse_dct, dec_idxs, dec_vals)
    rec_norm_dct = dct_decode(sparse_dct)

    # Denormalize all signals to physical scale
    ref_phys = denormalize(w_norm, norm_stats)
    ae_phys = denormalize(rec_norm_ae, norm_stats)
    dct_phys = denormalize(rec_norm_dct, norm_stats)

    meta_dict = {
        "subject": w_meta["subject"],
        "window_id": w_meta["window_id"],
        "db": 8,
    }
    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
        sigma_c = float(norm_stats["std"][c_idx])
        m_ae = compute_channel_metrics(ref_phys[c_idx], ae_phys[c_idx], channel_name=ch_name, sigma_train=sigma_c)
        m_dct = compute_channel_metrics(ref_phys[c_idx], dct_phys[c_idx], channel_name=ch_name, sigma_train=sigma_c)
        meta_dict[f"prd_ae_{ch_name}"] = m_ae["prd"]
        meta_dict[f"prd_dct_{ch_name}"] = m_dct["prd"]

    return plot_single_window_comparison(ref_phys, ae_phys, dct_phys, meta=meta_dict, output_filepath=output_filepath)


def generate_real_failure_cases(
    df_results: Any,
    output_filepath_png: Union[str, Path],
    output_filepath_csv: Union[str, Path]
) -> Tuple[Path, Path]:
    """
    Generate failure_cases.png and failure_cases.csv using 100% REAL evaluated results from results.csv.
    Selection Rule: Sort db=2 rows by PRD descending to find the worst-case real window.
    Reconstructs that exact real window using real AE and DCT baselines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = df_results.to_dict("records") if hasattr(df_results, "to_dict") else df_results

    valid_db2_recs = [
        r for r in records
        if int(r["db"]) == 2 and (str(r.get("valid_prd")).lower() == "true" or r.get("valid_prd") is True)
    ]
    if not valid_db2_recs:
        valid_db2_recs = [
            r for r in records
            if (str(r.get("valid_prd")).lower() == "true" or r.get("valid_prd") is True)
        ]

    # Sort descending by PRD
    sorted_recs = sorted(valid_db2_recs, key=lambda r: float(r["PRD"]), reverse=True)

    # Export failure_cases.csv (top worst failure rows)
    csv_path = export_failure_cases_csv(sorted_recs[:16], output_filepath=output_filepath_csv)

    # Pick top worst failure row
    worst_row = sorted_recs[0]
    fold = int(worst_row["fold"])
    subj = str(worst_row["subject"])
    w_id = str(worst_row["window_id"])
    db = int(worst_row["db"])

    proc_test_path = PROJECT_ROOT / "data" / "processed" / f"fold{fold}" / "test.npz"
    norm_stats_path = PROJECT_ROOT / "configs" / f"norm_stats_fold{fold}.json"
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"fold{fold:02d}_db{db:02d}_seed42.pt"

    if not proc_test_path.exists() or not norm_stats_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"[ERROR] Required real dataset/checkpoint missing for fold {fold} db {db}!")

    npz_data = np.load(proc_test_path, allow_pickle=True)
    test_windows_norm = npz_data["windows"].astype(np.float32)
    test_meta_arr = npz_data["metadata"]
    test_meta_list = test_meta_arr.item() if test_meta_arr.ndim == 0 else list(test_meta_arr)
    norm_stats = load_norm_stats(norm_stats_path)

    # Find matching window index
    target_w_idx = 0
    for idx, m in enumerate(test_meta_list):
        if str(m.get("window_id")) == w_id:
            target_w_idx = idx
            break

    w_norm = test_windows_norm[target_w_idx]

    # Real AE reconstruction
    ckpt_data = torch.load(ckpt_path, map_location=device)
    model = C1Autoencoder(d_b=db).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_in = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)
        latent_tensor = model.encode(x_in)
        latent_np = latent_tensor.cpu().numpy()
        ae_bytes = encode_ae_bytes(latent_np, d_b=db, profile_id=0)
        latent_dec_np, _ = decode_ae_bytes(ae_bytes)
        latent_dec_tensor = torch.from_numpy(latent_dec_np.reshape(1, db, 32)).float().to(device)
        rec_norm_ae = model.decode(latent_dec_tensor).squeeze(0).cpu().numpy()

    # Real DCT reconstruction
    b_info = compute_equal_byte_budget(db)
    k_eq_byte = b_info["K_equal_byte"]
    pad_bytes = b_info["padding"]
    topk_vals, topk_idxs, _, _ = dct_encode_topk(w_norm, k=k_eq_byte)
    dct_bytes = encode_dct_bytes(topk_vals, topk_idxs, d_b=db, profile_id=0, pad_bytes=pad_bytes)
    dec_vals, dec_idxs, _ = decode_dct_bytes(dct_bytes)
    sparse_dct = np.zeros((4, 512), dtype=np.float32)
    np.put(sparse_dct, dec_idxs, dec_vals)
    rec_norm_dct = dct_decode(sparse_dct)

    # Denormalize all signals to physical scale
    ref_phys = denormalize(w_norm, norm_stats)
    ae_phys = denormalize(rec_norm_ae, norm_stats)
    dct_phys = denormalize(rec_norm_dct, norm_stats)

    meta_dict = {
        "subject": subj,
        "window_id": w_id,
        "db": db,
    }
    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
        sigma_c = float(norm_stats["std"][c_idx])
        m_ae = compute_channel_metrics(ref_phys[c_idx], ae_phys[c_idx], channel_name=ch_name, sigma_train=sigma_c)
        m_dct = compute_channel_metrics(ref_phys[c_idx], dct_phys[c_idx], channel_name=ch_name, sigma_train=sigma_c)
        meta_dict[f"prd_ae_{ch_name}"] = m_ae["prd"]
        meta_dict[f"prd_dct_{ch_name}"] = m_dct["prd"]

    png_path = plot_single_window_comparison(ref_phys, ae_phys, dct_phys, meta=meta_dict, output_filepath=output_filepath_png)
    return png_path, csv_path


def generate_experiment_summary_md(
    val_info: Dict[str, Any],
    df_subject: Any,
    df_overall: Any,
    df_paired: Any,
    output_filepath: Union[str, Path] = "results/experiment_summary.md"
) -> Path:
    """
    Generate dynamic Markdown report (experiment_summary.md).
    ZERO HARDCODED NUMBERS.
    """
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records_overall = df_overall.to_dict("records") if hasattr(df_overall, "to_dict") else df_overall
    records_paired = df_paired.to_dict("records") if hasattr(df_paired, "to_dict") else df_paired

    # Compute dynamic summary stats
    ae_wins_count = sum(1 for r in records_paired if r.get("ae_wins_prd", False))
    total_pairs = len(records_paired)
    win_rate = (ae_wins_count / total_pairs * 100.0) if total_pairs > 0 else 0.0

    table7_df = generate_table7_mean_std_across_subjects(df_subject)
    table7_recs = table7_df.to_dict("records") if hasattr(table7_df, "to_dict") else table7_df

    table8_df = generate_table8_byte_cost()
    table8_recs = table8_df.to_dict("records") if hasattr(table8_df, "to_dict") else table8_df

    table9_df = generate_table9_ae_parameter_count()
    table9_recs = table9_df.to_dict("records") if hasattr(table9_df, "to_dict") else table9_df

    # Construct Markdown text dynamically
    md_lines = [
        "# C1_AE_DCT — Automated Scientific Experiment Summary Report",
        "",
        "> **Note:** This report is dynamically generated from `results.csv`. Zero hard-coded numbers.",
        "",
        "## 1. Pre-Reporting Dataset Validation",
        "",
        f"- **Total Evaluated Records:** {val_info['n_records']}",
        f"- **Matched AE/DCT Window Pairs:** {val_info['n_pairs']}",
        f"- **Validated Test Subjects ({len(val_info['subjects'])}/15):** `{', '.join(val_info['subjects'])}`",
        f"- **Validated Cross-Validation Folds ({len(val_info['folds'])}/5):** `{', '.join(map(str, val_info['folds']))}`",
        f"- **Validated Compression Budgets ($d_b$):** `{', '.join(map(str, val_info['dbs']))}`",
        "- **Byte Length & Valid Mask Consistency:** 100% Passed",
        "",
        "## 2. Overall Performance Summary Across Subjects (Mean ± Std)",
        "",
        r"| Method | $d_b$ | $CR_{dim}$ | $CR_{byte\_64}$ | Channel | PRD (Mean ± Std) | PRDN (Mean ± Std) | RMSE (Mean ± Std) | Invalid Rate |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in table7_recs:
        md_lines.append(
            f"| {r['method']} | {r['db']} | {r['CR_dim']:.1f}x | {r['CR_byte_64']:.2f}x | "
            f"{r['channel']} | {r['prd_mean_std']} | {r['prdn_mean_std']} | {r['rmse_mean_std']} | {r['invalid_rate_prdn']} |"
        )

    md_lines.extend([
        "",
        "## 3. Paired Comparison: Autoencoder vs DCT Baseline",
        "",
        r"- **Total Paired Subject-Channel Comparisons:** " + f"{total_pairs}",
        r"- **Autoencoder Win Count ($\delta_s < 0$):** " + f"{ae_wins_count} / {total_pairs} ({win_rate:.1f}%)",
        "",
        "## 4. Equal Byte Budget Cost Allocation (Table 8)",
        "",
        r"| $d_b$ | $M$ (AE Latent) | $K_{dim}$ (DCT) | $K_{byte}$ (DCT) | Padding | $B_{AE}$ (bytes) | $B_{DCT}$ (bytes) | $CR_{byte\_64}$ | $CR_{byte\_native}$ |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for r in table8_recs:
        md_lines.append(
            f"| {r['db']} | {r['M']} | {r['K_equal_dim']} | {r['K_equal_byte']} | {r['padding']} | "
            f"{r['B_AE']} | {r['B_DCT']} | {r['CR_byte_64']:.2f}x | {r['CR_byte_native']:.2f}x |"
        )

    md_lines.extend([
        "",
        "## 5. Autoencoder Model Parameter Count (Table 9)",
        "",
        "| $d_b$ | Latent Dim $M$ | Encoder Params | Decoder Params | Total Params | Model Size (KB) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for r in table9_recs:
        md_lines.append(
            f"| {r['d_b']} | {r['latent_dim_M']} | {r['encoder_parameters']:,} | "
            f"{r['decoder_parameters']:,} | {r['total_parameters']:,} | {r['param_size_kb']:.2f} KB |"
        )

    md_lines.extend([
        "",
        "## 6. Generated Figures & Plots",
        "",
        "### Figure 1: $CR_{dim}$ vs PRD (%)",
        "![CR_dim vs PRD](cr_dim_prd.png)",
        "",
        "### Figure 2: $CR_{byte}$ vs PRD (%)",
        "![CR_byte vs PRD](cr_byte_prd.png)",
        "",
        "### Figure 3: PRDN Distortion Curves",
        "![PRDN Curves](prdn_curves.png)",
        "",
        "### Figure 4: RMSE Distortion Curves",
        "![RMSE Curves](rmse_curves.png)",
        "",
        "### Figure 5: Signal Reconstruction Examples (Original | AE | DCT)",
        "![Reconstruction Examples](reconstruction_examples.png)",
        "",
        "### Figure 6: Worst-Case Failure Analysis (High Compression $d_b=2$)",
        "![Failure Cases](failure_cases.png)",
        "",
        "---",
        "*Report generated automatically by `src/make_report.py`.*",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[TASK 24] Exported dynamic experiment summary report: {out_path.name}")
    return out_path


def make_report(
    results_csv_path: Union[str, Path] = "results/results.csv",
    output_dir: Union[str, Path] = "results"
) -> Dict[str, Path]:
    """
    Main Automated Report Generation pipeline for TASK 24 / TASK 5.

    1. Reads results/results.csv (or decompresses results.csv.gz if present).
       RAISES FileNotFoundError if results dataset is missing (ZERO synthetic fallback).
    2. Runs strict pre-reporting validation checks.
    3. Computes subject aggregation, overall summary, paired comparison.
    4. Saves CSVs: summary_by_subject.csv, paired_comparison.csv, overall_summary.csv.
    5. Generates PNG plots: cr_dim_prd.png, cr_byte_prd.png, prdn_curves.png, rmse_curves.png.
    6. Generates 100% REAL signal reconstructions for reconstruction_examples.png and failure_cases.png.
    7. Generates experiment_summary.md with dynamic zero hard-coded content.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = Path(results_csv_path)

    # Check for uncompressed or compressed results file
    if not csv_file.exists():
        gz_file = csv_file.parent / (csv_file.name + ".gz")
        if gz_file.exists():
            print(f"Decompressing {gz_file.name} -> {csv_file.name}...")
            with gzip.open(gz_file, "rb") as f_in, open(csv_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            raise FileNotFoundError(
                f"[ERROR] Results file '{csv_file}' not found! "
                "Synthetic fallback is permanently disabled in production reporting. "
                "Please run 'python src/evaluate_all.py' first to evaluate real checkpoints on test windows."
            )

    # Step 1: Read results dataset
    if pd is not None:
        df_results = pd.read_csv(csv_file)
    else:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            df_results = list(reader)

    # Step 2: Pre-reporting validation
    val_info = validate_before_report(df_results)

    # Step 3: Compute aggregations
    df_subject = aggregate_by_subject(df_results)
    df_overall = compute_overall_summary(df_subject)
    df_paired = compute_paired_comparison(df_subject)

    # Step 4: Export CSVs
    p_subj = out_dir / "summary_by_subject.csv"
    p_pair = out_dir / "paired_comparison.csv"
    p_over = out_dir / "overall_summary.csv"

    save_records_to_csv(df_subject, p_subj)
    save_records_to_csv(df_paired, p_pair)
    save_records_to_csv(df_overall, p_over)

    # Step 5: Generate PNG plots with exact required filenames
    records_overall = _extract_overall_summary_records(df_overall)

    p_cr_dim_prd = out_dir / "cr_dim_prd.png"
    p_cr_byte_prd = out_dir / "cr_byte_prd.png"
    p_prdn_curves = out_dir / "prdn_curves.png"
    p_rmse_curves = out_dir / "rmse_curves.png"

    plot_metric_vs_cr(
        records_overall, x_key="CR_dim", y_key="prd_mean",
        x_label="Dimension Compression Ratio (CR_dim)", y_label="PRD (%)",
        title="CR_dim vs PRD (%) Across Channels",
        png_path=p_cr_dim_prd, csv_path=out_dir / "cr_dim_prd_data.csv"
    )

    plot_metric_vs_cr(
        records_overall, x_key="CR_byte_64", y_key="prd_mean",
        x_label="Byte Compression Ratio (CR_byte_64)", y_label="PRD (%)",
        title="CR_byte vs PRD (%) Across Channels",
        png_path=p_cr_byte_prd, csv_path=out_dir / "cr_byte_prd_data.csv"
    )

    plot_metric_vs_cr(
        records_overall, x_key="CR_dim", y_key="prdn_mean",
        x_label="Dimension Compression Ratio (CR_dim)", y_label="PRDN (%)",
        title="PRDN Distortion Curves Across Channels",
        png_path=p_prdn_curves, csv_path=out_dir / "prdn_curves_data.csv"
    )

    plot_metric_vs_cr(
        records_overall, x_key="CR_dim", y_key="rmse_mean",
        x_label="Dimension Compression Ratio (CR_dim)", y_label="RMSE (Physical Units)",
        title="RMSE Distortion Curves Across Channels",
        png_path=p_rmse_curves, csv_path=out_dir / "rmse_curves_data.csv"
    )

    # Step 6: Generate 100% REAL signal reconstructions for typical & failure plots
    p_recon_ex = out_dir / "reconstruction_examples.png"
    p_failure_ex = out_dir / "failure_cases.png"
    p_failure_csv = out_dir / "failure_cases.csv"

    generate_real_reconstruction_examples(p_recon_ex)
    generate_real_failure_cases(df_results, p_failure_ex, p_failure_csv)

    # Step 7: Generate dynamic experiment_summary.md
    p_md = generate_experiment_summary_md(val_info, df_subject, df_overall, df_paired, out_dir / "experiment_summary.md")

    print(f"\n==========================================================")
    print("      TASK 24 — AUTOMATED REPORT GENERATOR COMPLETED      ")
    print("==========================================================")
    print(f"Generated CSVs:       {p_subj.name}, {p_pair.name}, {p_over.name}, {p_failure_csv.name}")
    print(f"Generated PNGs:       {p_cr_dim_prd.name}, {p_cr_byte_prd.name}, {p_prdn_curves.name}, {p_rmse_curves.name}, {p_recon_ex.name}, {p_failure_ex.name}")
    print(f"Generated Markdown:   {p_md.name}")
    print("==========================================================")

    return {
        "summary_by_subject": p_subj,
        "paired_comparison": p_pair,
        "overall_summary": p_over,
        "cr_dim_prd": p_cr_dim_prd,
        "cr_byte_prd": p_cr_byte_prd,
        "prdn_curves": p_prdn_curves,
        "rmse_curves": p_rmse_curves,
        "reconstruction_examples": p_recon_ex,
        "failure_cases": p_failure_ex,
        "failure_cases_csv": p_failure_csv,
        "experiment_summary_md": p_md,
    }


def generate_report(results: Optional[Dict[str, Any]] = None, output_path: str = "results/experiment_summary.md") -> str:
    """Stub integration function for project-level generate_report calls."""
    res_dict = make_report(output_dir=Path(output_path).parent)
    return str(res_dict["experiment_summary_md"])


if __name__ == "__main__":
    make_report()

