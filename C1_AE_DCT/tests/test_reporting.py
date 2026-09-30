"""Unit tests for TASK 22 — Plots and Tables Generation (reporting.py)."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import (
    generate_table8_byte_cost,
    generate_table9_ae_parameter_count,
    generate_table6_ae_vs_dct_per_subject,
    generate_table7_mean_std_across_subjects,
    generate_all_plots_and_tables,
)
from src.aggregation import aggregate_by_subject, compute_overall_summary
from src.results_schema import create_result_row, append_result


def test_table8_byte_cost_values():
    """Verify Table 8 Byte Cost table formulas and structure."""
    table8 = generate_table8_byte_cost()
    records = table8.to_dict("records") if hasattr(table8, "to_dict") else table8

    assert len(records) == 4
    dbs = [r["db"] for r in records]
    assert dbs == [16, 8, 4, 2]

    for r in records:
        assert r["B_AE"] == r["B_DCT"], f"Byte budget mismatch in Table 8 for d_b={r['db']}"
        assert r["CR_byte_64"] > 0
        assert r["CR_byte_native"] > 0

    print("[PASS] Table 8 byte cost test passed!")


def test_table9_ae_parameter_count_values():
    """Verify Table 9 Autoencoder parameter count table across d_b in [16, 8, 4, 2]."""
    table9 = generate_table9_ae_parameter_count()
    records = table9.to_dict("records") if hasattr(table9, "to_dict") else table9

    assert len(records) == 4
    # Expected total params: d_b=16 -> 36724, d_b=8 -> 31596, d_b=4 -> 29032, d_b=2 -> 27750
    expected_totals = {16: 36724, 8: 31596, 4: 29032, 2: 27750}

    for r in records:
        db = r["d_b"]
        assert r["total_parameters"] == expected_totals[db], f"Param mismatch for d_b={db}"
        assert r["encoder_parameters"] > 0
        assert r["decoder_parameters"] > 0

    print("[PASS] Table 9 AE parameter count test passed!")


def test_full_plots_and_tables_generation(tmp_path):
    """
    Verify generation of 5 Figures (PNG + CSV source) and 4 Tables (CSV)
    using representative multi-budget summary dataset.
    """
    raw_results = []
    subjects = [f"S{i}" for i in range(1, 16)]
    channels = ["PPG", "ACCx", "ACCy", "ACCz"]
    methods = ["AE", "DCT"]
    dbs = [16, 8, 4, 2]

    np.random.seed(42)

    for db in dbs:
        for subj in subjects:
            for w in range(5):
                for ch in channels:
                    for method in methods:
                        base_err = (18 - db) * 0.5 if method == "AE" else (18 - db) * 0.7
                        prd = float(base_err + np.random.randn() * 0.2)
                        prdn = float(base_err * 2.0 + np.random.randn() * 0.3)
                        rmse = float(base_err * 0.05 + np.random.rand() * 0.01)

                        row = create_result_row(
                            fold=1, subject=subj, seed=42, method=method, db=db, K=32*db,
                            channel=ch, window_id=f"win_{w:04d}", start_index=w*512, nbytes=16 + 4*32*db,
                            CR_dim=2048.0/(32*db), CR_byte_64=8192.0/(16 + 4*32*db), CR_byte_native=5120.0/(16 + 4*32*db),
                            PRD=prd, PRDN=prdn, RMSE=rmse, metric_valid=True,
                            checkpoint="ckpt.pt", config_id=f"{method}_db{db}"
                        )
                        row["valid_prd"] = True
                        row["valid_prdn"] = True
                        raw_results = append_result(raw_results, row)

    df_subject = aggregate_by_subject(raw_results)
    df_overall = compute_overall_summary(df_subject)

    # Generate all plots & tables
    outputs = generate_all_plots_and_tables(df_subject, df_overall, output_dir=tmp_path)

    # Verify 5 Figures PNG + CSV source
    for fig_id in [1, 2, 3, 4, 5]:
        png_key = f"fig{fig_id}_png" if fig_id != 5 else "fig5_png"
        csv_key = f"fig{fig_id}_csv" if fig_id != 5 else "fig5_csv"

        assert outputs[png_key].exists(), f"Figure {fig_id} PNG file missing!"
        assert outputs[csv_key].exists(), f"Figure {fig_id} CSV source file missing!"
        assert outputs[png_key].stat().st_size > 0
        assert outputs[csv_key].stat().st_size > 0

    # Verify 4 Tables CSV
    for tab_id in [6, 7, 8, 9]:
        key = f"table{tab_id}"
        assert outputs[key].exists(), f"Table {tab_id} CSV file missing!"
        assert outputs[key].stat().st_size > 0

    print("[PASS] Full 5 Figures (PNG+CSV) + 4 Tables (CSV) generation test passed!")


if __name__ == "__main__":
    print("=== TASK 22 — PLOTS AND TABLES VERIFICATION SUITE ===")
    test_table8_byte_cost_values()
    test_table9_ae_parameter_count_values()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_full_plots_and_tables_generation(Path(tmpdir))
    print("=== ALL TASK 22 REPORTING TESTS PASSED CLEANLY ===")
