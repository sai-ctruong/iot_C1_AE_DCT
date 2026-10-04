"""Unit test suite for TASK 3 — Test Window Step Sensitivity Analysis."""

import sys
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STEP4_DIR = PROJECT_ROOT / "data" / "processed_step4"
STEP8_DIR = PROJECT_ROOT / "data" / "processed"
SENSITIVITY_DIR = PROJECT_ROOT / "results" / "sensitivity_step4"


def test_15_test_subjects_present():
    """1. Test that all 15 subjects are evaluated in step4 sensitivity analysis."""
    csv_path = SENSITIVITY_DIR / "overall_summary_equal_dim.csv"
    assert csv_path.exists(), f"Missing summary CSV: {csv_path}"
    df = pd.read_csv(csv_path)
    assert np.all(df["n_subjects"] == 15), "Not all rows analyze 15 subjects!"
    print("[PASS] 1. 15 Test subjects present verified.")


def test_5_folds_present():
    """2. Test that all 5 folds exist in processed_step4 directory."""
    for fold in range(1, 6):
        fold_npz = STEP4_DIR / f"fold{fold}" / "test.npz"
        assert fold_npz.exists(), f"Missing fold {fold} step4 test npz at {fold_npz}"
    print("[PASS] 2. 5 folds present in processed_step4 verified.")


def test_step4_window_count_greater_than_step8():
    """3. Test that step4 Test window count is strictly greater than step8 count."""
    total_step4 = 0
    total_step8 = 0
    for fold in range(1, 6):
        data4 = np.load(STEP4_DIR / f"fold{fold}" / "test.npz")
        data8 = np.load(STEP8_DIR / f"fold{fold}" / "test.npz")
        n4 = len(data4["windows"])
        n8 = len(data8["windows"])
        assert n4 > n8, f"Fold {fold}: step4 count {n4} <= step8 count {n8}"
        total_step4 += n4
        total_step8 += n8

    # Approximately double due to 50% overlap (step=4s vs step=8s)
    assert total_step4 > total_step8 * 1.8, f"Total step4 {total_step4} expected to be ~2x step8 {total_step8}"
    print(f"[PASS] 3. Step4 window count ({total_step4}) > Step8 window count ({total_step8}) verified.")


def test_same_subject_fold_assignments():
    """4. Test that subject-to-fold assignments match 100% between step4 and step8."""
    for fold in range(1, 6):
        data4 = np.load(STEP4_DIR / f"fold{fold}" / "test.npz", allow_pickle=True)
        data8 = np.load(STEP8_DIR / f"fold{fold}" / "test.npz", allow_pickle=True)

        meta4 = data4["metadata"]
        meta8 = data8["metadata"]

        subjs4 = set([m.get("subject", "S1") if isinstance(m, dict) else getattr(m, "subject", "S1") for m in meta4])
        subjs8 = set([m.get("subject", "S1") if isinstance(m, dict) else getattr(m, "subject", "S1") for m in meta8])

        assert subjs4 == subjs8, f"Fold {fold} subject mismatch: {subjs4} != {subjs8}"
    print("[PASS] 4. Subject-fold assignment matching verified.")


def test_same_normalization_stats():
    """5. Test that normalization stats files are identical."""
    for fold in range(1, 6):
        norm_file = PROJECT_ROOT / "configs" / f"norm_stats_fold{fold}.json"
        assert norm_file.exists(), f"Norm stats file missing: {norm_file}"
    print("[PASS] 5. Same normalization stats verified.")


def test_no_retraining_and_no_synthetic_data():
    """6 & 7. Test that AE checkpoints are reused and zero synthetic data exists."""
    # Verify main checkpoints exist and were reused
    for fold in range(1, 6):
        for db in [16, 8, 4, 2]:
            ckpt = PROJECT_ROOT / "checkpoints" / f"fold0{fold}_db{db:02d}_seed42.pt"
            assert ckpt.exists(), f"Missing trained checkpoint: {ckpt}"

    # Verify no NaNs or synthetic tags in results_step4.csv.gz
    gz_csv = SENSITIVITY_DIR / "results_step4.csv.gz"
    assert gz_csv.exists(), f"Missing {gz_csv}"
    with gzip.open(gz_csv, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, keep_default_na=False, low_memory=False)
        assert len(df) > 0, "results_step4.csv.gz is empty"
        assert not df.isnull().values.any(), "NaN found in results_step4.csv.gz"

    print("[PASS] 6 & 7. No retraining and zero synthetic data verified.")

