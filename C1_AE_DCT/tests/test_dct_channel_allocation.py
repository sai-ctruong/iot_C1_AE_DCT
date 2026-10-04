"""Unit test suite for TASK 2 — DCT Top-K Channel Allocation Analysis."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_dct import dct_encode_topk, CHANNEL_NAMES

RESULTS_DIR = PROJECT_ROOT / "results"
DETAIL_CSV = RESULTS_DIR / "dct_channel_allocation_detail.csv"
SUMMARY_CSV = RESULTS_DIR / "dct_channel_allocation_summary.csv"


def test_channel_counts_sum_to_k():
    """1. Test that PPG_count + ACCx_count + ACCy_count + ACCz_count == K for every row."""
    assert DETAIL_CSV.exists(), f"Detail CSV missing: {DETAIL_CSV}"
    df = pd.read_csv(DETAIL_CSV)
    assert len(df) > 0, "Detail CSV is empty"

    sum_counts = df["PPG_count"] + df["ACCx_count"] + df["ACCy_count"] + df["ACCz_count"]
    assert np.all(sum_counts == df["K"]), "Channel counts do not sum to K in all detail rows!"
    print("[PASS] 1. Channel counts sum to K verified.")


def test_channel_counts_bounds_nonnegative():
    """2 & 3. Test that all counts >= 0 and all counts <= K."""
    df = pd.read_csv(DETAIL_CSV)
    for ch in ["PPG", "ACCx", "ACCy", "ACCz"]:
        col = f"{ch}_count"
        assert np.all(df[col] >= 0), f"Found negative count in {col}"
        assert np.all(df[col] <= df["K"]), f"Found count > K in {col}"
    print("[PASS] 2 & 3. Channel counts bounds [0, K] verified.")


def test_all_four_channels_appear():
    """4. Test that all four channels appear in the summary report."""
    assert SUMMARY_CSV.exists(), f"Summary CSV missing: {SUMMARY_CSV}"
    df = pd.read_csv(SUMMARY_CSV)
    channels_found = set(df["channel"].unique())
    expected_channels = {"PPG", "ACCx", "ACCy", "ACCz"}
    assert channels_found == expected_channels, f"Expected channels {expected_channels}, found {channels_found}"
    print("[PASS] 4. All four channels appear verified.")


def test_test_subjects_15():
    """5. Test that exactly 15 subjects are analyzed."""
    df_detail = pd.read_csv(DETAIL_CSV)
    subjs = set(df_detail["subject"].unique())
    assert len(subjs) == 15, f"Expected 15 subjects, found {len(subjs)}: {subjs}"

    df_summary = pd.read_csv(SUMMARY_CSV)
    assert np.all(df_summary["n_subjects"] == 15), "Summary n_subjects != 15 in some rows"
    print("[PASS] 5. Test subjects = 15 verified.")


def test_no_synthetic_data():
    """6. Test that no synthetic dataset tags exist in allocation analysis."""
    df_detail = pd.read_csv(DETAIL_CSV)
    folds_found = set(df_detail["fold"].unique())
    assert folds_found == {1, 2, 3, 4, 5}, f"Expected folds 1..5, got {folds_found}"
    assert not df_detail.isnull().values.any(), "Found NaN values in allocation detail CSV"
    print("[PASS] 6. No synthetic data verified.")
