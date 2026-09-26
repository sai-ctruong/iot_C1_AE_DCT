"""Unit tests for TASK 24 — Automated Report Generator (make_report.py)."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.make_report import (
    validate_before_report,
    generate_synthetic_full_results,
    generate_experiment_summary_md,
    make_report,
)


import csv

try:
    import pandas as pd
except ImportError:
    pd = None


def load_csv_data(filepath: Path) -> Any:
    if pd is not None:
        return pd.read_csv(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_validate_before_report_pass(tmp_path):
    """Verify validate_before_report passes on valid 5-fold 15-subject 4-db dataset."""
    csv_file = tmp_path / "valid_results.csv"
    generate_synthetic_full_results(csv_file)

    df_or_records = load_csv_data(csv_file)

    val_info = validate_before_report(df_or_records)
    assert len(val_info["subjects"]) == 15
    assert len(val_info["folds"]) == 5
    assert val_info["dbs"] == [2, 4, 8, 16]

    print("[PASS] Pre-reporting validation pass test passed!")


def test_validate_before_report_failures(tmp_path):
    """Verify validate_before_report raises ValueError for incomplete datasets."""
    csv_file = tmp_path / "incomplete_results.csv"
    generate_synthetic_full_results(csv_file)

    records = load_csv_data(csv_file)
    if hasattr(records, "to_dict"):
        records = records.to_dict("records")

    # 1. Drop subject S15
    missing_subj = [r for r in records if r["subject"] != "S15"]
    with pytest.raises(ValueError, match="Missing Test subjects"):
        validate_before_report(missing_subj)

    # 2. Change fold 5 to fold 1 (missing fold 5 while keeping all subjects)
    missing_fold = [dict(r, fold=1) if int(r["fold"]) == 5 else r for r in records]
    with pytest.raises(ValueError, match="Missing Folds"):
        validate_before_report(missing_fold)

    # 3. Change db=2 to db=4 (missing db=2 while keeping all subjects and folds)
    missing_db = [dict(r, db=4) if int(r["db"]) == 2 else r for r in records]
    with pytest.raises(ValueError, match="Missing db budget factors"):
        validate_before_report(missing_db)

    print("[PASS] Pre-reporting validation failure detection test passed!")




def test_make_report_pipeline_execution(tmp_path):
    """Verify make_report pipeline creates all 3 CSVs, 6 PNGs, and experiment_summary.md without hard-coding."""
    csv_file = tmp_path / "results.csv"
    out_dir = tmp_path / "report_output"

    generate_synthetic_full_results(csv_file)

    outputs = make_report(results_csv_path=csv_file, output_dir=out_dir)

    # Verify CSV files
    assert outputs["summary_by_subject"].exists()
    assert outputs["paired_comparison"].exists()
    assert outputs["overall_summary"].exists()

    assert outputs["summary_by_subject"].name == "summary_by_subject.csv"
    assert outputs["paired_comparison"].name == "paired_comparison.csv"
    assert outputs["overall_summary"].name == "overall_summary.csv"

    # Verify 6 PNG files
    png_keys = [
        "cr_dim_prd", "cr_byte_prd", "prdn_curves",
        "rmse_curves", "reconstruction_examples", "failure_cases"
    ]
    for key in png_keys:
        assert outputs[key].exists(), f"PNG plot {key} missing!"
        assert outputs[key].stat().st_size > 0

    # Verify Markdown file
    md_file = outputs["experiment_summary_md"]
    assert md_file.exists()
    assert md_file.name == "experiment_summary.md"

    md_content = md_file.read_text(encoding="utf-8")
    assert "C1_AE_DCT — Automated Scientific Experiment Summary Report" in md_content
    assert all(s in md_content for s in ["S1", "S2", "S15"])
    assert "cr_dim_prd.png" in md_content

    print("[PASS] Full make_report pipeline execution test passed!")



if __name__ == "__main__":
    print("=== TASK 24 — AUTOMATED REPORT GENERATOR VERIFICATION SUITE ===")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        test_validate_before_report_pass(p)
        test_validate_before_report_failures(p)
        test_make_report_pipeline_execution(p)
    print("=== ALL TASK 24 REPORT GENERATOR TESTS PASSED CLEANLY ===")
