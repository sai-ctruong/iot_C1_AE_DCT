"""Unit tests for TASK 20 — Detailed Result Schema and Joining (results_schema.py)."""

import sys
import math
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results_schema import (
    RESULT_SCHEMA_COLUMNS,
    create_result_row,
    append_result,
    validate_results_schema,
    join_ae_dct_results,
)


def test_schema_mandatory_columns():
    """Verify all 19 mandatory columns required by TASK 20 are defined in RESULT_SCHEMA_COLUMNS."""
    required = [
        "fold", "subject", "seed", "method", "db", "K", "channel", "window_id",
        "start_index", "nbytes", "CR_dim", "CR_byte_64", "CR_byte_native",
        "PRD", "PRDN", "RMSE", "metric_valid", "checkpoint", "config_id"
    ]
    for col in required:
        assert col in RESULT_SCHEMA_COLUMNS, f"Mandatory column {col} missing from schema!"

    print("[PASS] Schema mandatory columns check passed!")


def test_append_result_building():
    """Verify append_result appends valid records cleanly."""
    container = []
    
    row1 = create_result_row(
        fold=1, subject="S1", seed=42, method="AE", db=8, K=256,
        channel="PPG", window_id="win_0001", start_index=0, nbytes=1040,
        CR_dim=8.0, CR_byte_64=7.87, CR_byte_native=4.92,
        PRD=3.5, PRDN=8.2, RMSE=0.15, metric_valid=True,
        checkpoint="checkpoints/ae_f1_db8.pt", config_id="AE_f1_db8"
    )

    container = append_result(container, row1)
    assert len(container) == 1
    assert container[0]["fold"] == 1
    assert container[0]["subject"] == "S1"
    assert container[0]["budget"] == 8

    # Append via kwargs
    container = append_result(
        container,
        fold=1, subject="S1", seed=42, method="DCT", db=8, K=170,
        channel="PPG", window_id="win_0001", start_index=0, nbytes=1040,
        CR_dim=8.0, CR_byte_64=7.87, CR_byte_native=4.92,
        PRD=4.1, PRDN=9.5, RMSE=0.18, metric_valid=True,
        checkpoint="N/A", config_id="DCT_f1_db8"
    )

    assert len(container) == 2
    assert validate_results_schema(container) is True
    print("[PASS] Append result and schema validation passed!")


def test_validate_results_schema_failures():
    """Verify validate_results_schema detects missing columns or invalid values."""
    # Invalid channel name
    bad_channel_row = [create_result_row(
        fold=1, subject="S1", seed=42, method="AE", db=8, K=256,
        channel="INVALID_CH", window_id="win_0001", start_index=0, nbytes=1040,
        CR_dim=8.0, CR_byte_64=7.87, CR_byte_native=4.92,
        PRD=3.5, PRDN=8.2, RMSE=0.15, metric_valid=True,
        checkpoint="N/A", config_id="AE_f1_db8"
    )]

    with pytest.raises(ValueError, match="invalid channel"):
        validate_results_schema(bad_channel_row)

    # Missing mandatory column
    incomplete_row = [{"fold": 1, "subject": "S1"}]
    with pytest.raises(ValueError, match="missing mandatory columns"):
        validate_results_schema(incomplete_row)

    print("[PASS] Schema validation failure detection passed!")


def test_join_ae_dct_on_exact_window_keys():
    """Verify AE and DCT results on the same window join accurately on fold+subject+window_id+channel+budget."""
    ae_results = []
    dct_results = []

    for ch in ["PPG", "ACCx", "ACCy", "ACCz"]:
        # AE record
        ae_results = append_result(
            ae_results,
            fold=1, subject="S1", seed=42, method="AE", db=8, K=256,
            channel=ch, window_id="win_0001", start_index=0, nbytes=1040,
            CR_dim=8.0, CR_byte_64=7.87, CR_byte_native=4.92,
            PRD=2.0, PRDN=5.0, RMSE=0.1, metric_valid=True,
            checkpoint="ae.pt", config_id="AE_f1_db8"
        )
        # DCT record
        dct_results = append_result(
            dct_results,
            fold=1, subject="S1", seed=42, method="DCT", db=8, K=170,
            channel=ch, window_id="win_0001", start_index=0, nbytes=1040,
            CR_dim=8.0, CR_byte_64=7.87, CR_byte_native=4.92,
            PRD=3.0, PRDN=7.0, RMSE=0.15, metric_valid=True,
            checkpoint="N/A", config_id="DCT_f1_db8"
        )

    merged = join_ae_dct_results(ae_results, dct_results)
    assert len(merged) == 4

    for row in merged:
        assert row["fold"] == 1
        assert row["subject"] == "S1"
        assert row["window_id"] == "win_0001"
        assert row["budget"] == 8
        assert row["PRD_ae"] == 2.0
        assert row["PRD_dct"] == 3.0

    print("[PASS] Joining AE and DCT on exact window keys passed!")


if __name__ == "__main__":
    print("=== TASK 20 — RESULT SCHEMA VERIFICATION SUITE ===")
    test_schema_mandatory_columns()
    test_append_result_building()
    test_validate_results_schema_failures()
    test_join_ae_dct_on_exact_window_keys()
    print("=== ALL RESULT SCHEMA TESTS PASSED CLEANLY ===")
