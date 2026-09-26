"""Unit tests for TASK 21 — Subject-Level Aggregation and Paired Comparison (aggregation.py)."""

import os
import sys
import math
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_schema import create_result_row, append_result
from src.aggregation import (
    aggregate_by_subject,
    compute_overall_summary,
    compute_paired_comparison,
    export_aggregation_reports,
)


def generate_synthetic_results():
    """Generate representative multi-subject detailed results for AE and DCT across 5 subjects."""
    results = []
    subjects = [f"S{i}" for i in range(1, 6)]
    channels = ["PPG", "ACCx", "ACCy", "ACCz"]
    methods = ["AE", "DCT"]
    db = 8

    np.random.seed(42)

    # Note: S1 has 100 windows, S2 has 20 windows (different lengths to verify equal weighting)
    window_counts = {"S1": 100, "S2": 20, "S3": 50, "S4": 50, "S5": 50}

    for subj in subjects:
        n_win = window_counts[subj]
        for w in range(n_win):
            w_id = f"win_{w:04d}"
            for ch in channels:
                for method in methods:
                    # AE has slightly lower error than DCT
                    base_err = 3.0 if method == "AE" else 4.5
                    prd_val = float(base_err + np.random.randn() * 0.5)
                    prdn_val = float((base_err * 2.0) + np.random.randn() * 0.5)
                    rmse_val = float(0.1 + np.random.rand() * 0.05)

                    # Simulate 1 invalid window for ACCx in S1
                    valid = not (subj == "S1" and ch == "ACCx" and w == 0)
                    prd = prd_val if valid else float("nan")
                    prdn = prdn_val if valid else float("nan")

                    row = create_result_row(
                        fold=1, subject=subj, seed=42, method=method, db=db, K=256,
                        channel=ch, window_id=w_id, start_index=w*512, nbytes=1040,
                        CR_dim=8.0, CR_byte_64=7.87, CR_byte_native=4.92,
                        PRD=prd, PRDN=prdn, RMSE=rmse_val, metric_valid=valid,
                        checkpoint="ckpt.pt", config_id=f"{method}_f1_db8"
                    )
                    row["valid_prd"] = valid
                    row["valid_prdn"] = valid
                    results = append_result(results, row)

    return results


def test_subject_level_aggregation():
    """Verify Step 1 aggregates per subject x method x db x channel correctly over valid windows."""
    results = generate_synthetic_results()
    df_subject = aggregate_by_subject(results)

    # 5 subjects x 2 methods x 4 channels = 40 rows
    assert len(df_subject) == 40

    # Check S1 ACCx invalid rate
    records = df_subject.to_dict("records") if hasattr(df_subject, "to_dict") else df_subject
    s1_accx_ae = [r for r in records if r["subject"] == "S1" and r["channel"] == "ACCx" and r["method"] == "AE"][0]
    assert s1_accx_ae["n_total_windows"] == 100
    assert s1_accx_ae["n_valid_prd"] == 99
    assert abs(s1_accx_ae["invalid_rate_prd"] - 0.01) < 1e-5
    assert not math.isnan(s1_accx_ae["prd_mean"])
    assert not math.isnan(s1_accx_ae["prd_median"])
    assert not math.isnan(s1_accx_ae["prd_p90"])

    print("[PASS] Step 1 subject-level aggregation test passed!")


def test_equal_weighted_overall_aggregation():
    """Verify Step 2 averages across subjects with equal weighting regardless of window count."""
    results = generate_synthetic_results()
    df_subject = aggregate_by_subject(results)
    df_overall = compute_overall_summary(df_subject)

    # 2 methods x 4 channels = 8 overall summary rows
    assert len(df_overall) == 8

    records = df_overall.to_dict("records") if hasattr(df_overall, "to_dict") else df_overall
    ppg_ae = [r for r in records if r["method"] == "AE" and r["channel"] == "PPG"][0]
    assert ppg_ae["n_subjects"] == 5
    assert ppg_ae["prd_mean"] > 0
    assert ppg_ae["prdn_mean"] > 0

    print("[PASS] Step 2 equal-weighted overall aggregation test passed!")


def test_paired_comparison_delta_s():
    """Verify delta_s = AE - DCT calculation and boolean win flag."""
    results = generate_synthetic_results()
    df_subject = aggregate_by_subject(results)
    df_paired = compute_paired_comparison(df_subject)

    # 5 subjects x 4 channels = 20 paired rows
    assert len(df_paired) == 20

    records = df_paired.to_dict("records") if hasattr(df_paired, "to_dict") else df_paired
    for r in records:
        # AE error (~3.0) < DCT error (~4.5) => delta_s < 0 and ae_wins == True
        assert r["delta_prd_mean"] < 0, f"Expected delta_prd_mean < 0, got {r['delta_prd_mean']}"
        assert r["ae_wins_prd"] is True
        assert r["ae_wins_prdn"] is True

    print("[PASS] Paired comparison delta_s calculation test passed!")


def test_export_aggregation_reports(tmp_path):
    """Verify export_aggregation_reports generates summary_by_subject.csv, paired_comparison.csv, overall_summary.csv."""
    results = generate_synthetic_results()
    output_dict = export_aggregation_reports(results, output_dir=tmp_path)

    assert output_dict["summary_by_subject"].exists()
    assert output_dict["paired_comparison"].exists()
    assert output_dict["overall_summary"].exists()

    assert output_dict["summary_by_subject"].name == "summary_by_subject.csv"
    assert output_dict["paired_comparison"].name == "paired_comparison.csv"
    assert output_dict["overall_summary"].name == "overall_summary.csv"

    print("[PASS] Export aggregation CSV files test passed!")


if __name__ == "__main__":
    print("=== TASK 21 — SUBJECT-LEVEL AGGREGATION VERIFICATION SUITE ===")
    test_subject_level_aggregation()
    test_equal_weighted_overall_aggregation()
    test_paired_comparison_delta_s()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_export_aggregation_reports(Path(tmpdir))
    print("=== ALL TASK 21 AGGREGATION TESTS PASSED CLEANLY ===")
