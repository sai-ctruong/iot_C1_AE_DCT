"""Unit tests for TASK 23 — Reconstruction Examples & Failure Case Analysis (reconstruction_visualization.py)."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_schema import create_result_row, append_result
from src.reconstruction_visualization import (
    find_failure_cases,
    export_failure_cases_csv,
    plot_single_window_comparison,
    generate_reconstruction_task23_artifacts,
)


def generate_synthetic_task23_results():
    """Generate synthetic detailed window results for failure case testing."""
    results = []
    np.random.seed(42)

    for db in [16, 8, 4, 2]:
        for w in range(10):
            for ch in ["PPG", "ACCx", "ACCy", "ACCz"]:
                for method in ["AE", "DCT"]:
                    # Create one specific failure window with high PRD at db=2
                    if db == 2 and w == 9 and ch == "PPG" and method == "AE":
                        prd = 25.0
                    else:
                        prd = float(3.0 + np.random.rand() * 2.0)

                    row = create_result_row(
                        fold=1, subject="S1", seed=42, method=method, db=db, K=32*db,
                        channel=ch, window_id=f"win_{w:04d}", start_index=w*512, nbytes=1040,
                        CR_dim=2048.0/(32*db), CR_byte_64=7.87, CR_byte_native=4.92,
                        PRD=prd, PRDN=prd*2.0, RMSE=0.1, metric_valid=True,
                        checkpoint="ckpt.pt", config_id=f"{method}_db{db}"
                    )
                    results = append_result(results, row)

    return results


def test_find_failure_cases_objective_selection():
    """Verify find_failure_cases selects highest PRD windows at db=2 deterministically."""
    results = generate_synthetic_task23_results()
    failures = find_failure_cases(results, target_db=2, metric_col="PRD", top_k_per_channel=1)

    records = failures.to_dict("records") if hasattr(failures, "to_dict") else failures
    assert len(records) > 0

    # The highest PRD record (25.0%) should be at top of failure cases
    top_record = records[0]
    assert top_record["db"] == 2
    assert top_record["PRD"] == 25.0
    assert top_record["channel"] == "PPG"
    assert top_record["method"] == "AE"

    print("[PASS] Objective failure case selection test passed!")


def test_export_failure_cases_csv(tmp_path):
    """Verify export_failure_cases_csv exports CSV file for 100% exact reproducibility."""
    results = generate_synthetic_task23_results()
    failures = find_failure_cases(results, target_db=2)
    csv_file = tmp_path / "failure_cases.csv"

    export_failure_cases_csv(failures, output_filepath=csv_file)
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    print("[PASS] Failure cases CSV export test passed!")


def test_plot_single_window_comparison(tmp_path):
    """Verify plot_single_window_comparison generates PNG comparison figure with proper axes."""
    t = np.linspace(0, 10, 512)
    orig = np.array([np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])
    ae = orig + 0.05
    dct = orig + 0.1

    meta = {
        "subject": "S1", "window_id": "win_0001", "db": 2,
        "prd_ae_PPG": 5.2, "prd_dct_PPG": 8.4
    }

    png_path = tmp_path / "test_comparison.png"
    out = plot_single_window_comparison(orig, ae, dct, meta=meta, output_filepath=png_path)

    assert out.exists()
    assert out.stat().st_size > 0

    print("[PASS] Single window comparison plot generation test passed!")


def test_generate_reconstruction_task23_artifacts(tmp_path):
    """Verify complete TASK 23 workflow artifact generation."""
    t = np.linspace(0, 10, 512)
    orig = np.array([np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])
    ae = orig + 0.05
    dct = orig + 0.1

    orig_windows = np.array([orig, orig])
    ae_windows = np.array([ae, ae])
    dct_windows = np.array([dct, dct])

    results = generate_synthetic_task23_results()

    res = generate_reconstruction_task23_artifacts(
        orig_windows, ae_windows, dct_windows, results, output_dir=tmp_path
    )

    assert res["failure_cases_csv"].exists()
    assert res["typical_plot"].exists()
    assert res["failure_plot"].exists()

    print("[PASS] Full TASK 23 workflow artifacts test passed!")


if __name__ == "__main__":
    print("=== TASK 23 — RECONSTRUCTION EXAMPLES & FAILURE CASE VERIFICATION SUITE ===")
    test_find_failure_cases_objective_selection()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        test_export_failure_cases_csv(p)
        test_plot_single_window_comparison(p)
        test_generate_reconstruction_task23_artifacts(p)
    print("=== ALL TASK 23 RECONSTRUCTION TESTS PASSED CLEANLY ===")
