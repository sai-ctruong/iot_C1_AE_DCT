"""Unit tests for TASK 19 — Per-Channel Distortion Metrics Evaluation (evaluate.py)."""

import sys
import math
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    compute_channel_metrics,
    evaluate_window_metrics,
    evaluate_dataset_metrics,
    compute_joint_valid_mask,
)
from src.normalize import (
    compute_norm_stats,
    normalize,
    denormalize,
)


def test_case_1_prediction_equals_reference():
    """
    CASE 1: prediction == reference.
    PRD=0, PRDN=0, RMSE=0 if denominator is valid.
    """
    t = np.linspace(0, 10, 512)
    x_ref = np.sin(t) + 10.0  # Non-zero, non-constant physical signal
    x_pred = x_ref.copy()     # Exact identical reconstruction

    m = compute_channel_metrics(x_ref, x_pred, channel_name="PPG")

    assert m["rmse"] == 0.0, f"Expected RMSE=0, got {m['rmse']}"
    assert m["prd"] == 0.0, f"Expected PRD=0, got {m['prd']}"
    assert m["prdn"] == 0.0, f"Expected PRDN=0, got {m['prdn']}"
    assert m["valid_prd"] is True
    assert m["valid_prdn"] is True

    print("[PASS] Case 1: prediction == reference -> PRD=0, PRDN=0, RMSE=0")


def test_case_2_near_constant_zero_energy_reference():
    """
    CASE 2: reference near constant / zero-energy.
    Metrics marked undefined according to rules, no fake Inf or extreme values.
    """
    # 2a: Near-constant signal (variance = 0) -> PRDN denominator = 0
    x_ref_const = np.full(512, 5.0, dtype=np.float64)
    x_pred_const = x_ref_const + 0.1

    m_const = compute_channel_metrics(x_ref_const, x_pred_const, channel_name="ACCx")
    assert math.isnan(m_const["prdn"]), f"Expected NaN PRDN, got {m_const['prdn']}"
    assert m_const["valid_prdn"] is False
    assert not math.isinf(m_const["prdn"]), "PRDN should be NaN, not Inf"
    assert m_const["valid_prd"] is True  # sum(x^2) = 512 * 25 > 0

    # 2b: Zero-energy signal (x = 0) -> PRD and PRDN denominators = 0
    x_ref_zero = np.zeros(512, dtype=np.float64)
    x_pred_zero = np.ones(512, dtype=np.float64) * 0.05

    m_zero = compute_channel_metrics(x_ref_zero, x_pred_zero, channel_name="ACCy")
    assert math.isnan(m_zero["prd"]), f"Expected NaN PRD, got {m_zero['prd']}"
    assert math.isnan(m_zero["prdn"]), f"Expected NaN PRDN, got {m_zero['prdn']}"
    assert m_zero["valid_prd"] is False
    assert m_zero["valid_prdn"] is False
    assert not math.isinf(m_zero["prd"]), "PRD should be NaN, not Inf"
    assert not math.isinf(m_zero["prdn"]), "PRDN should be NaN, not Inf"

    print("[PASS] Case 2: Near-constant / zero-energy reference correctly marked invalid without fake Inf")


def test_case_3_ae_dct_shared_valid_mask():
    """
    CASE 3: AE and DCT must use the exact same valid mask.
    """
    t = np.linspace(0, 10, 512)
    ppg_valid = np.sin(t) + 5.0
    accx_const_zero = np.zeros(512)
    accy_valid = np.cos(t) + 2.0
    accz_valid = np.sin(2 * t) + 9.8

    ref_windows = np.array([
        [ppg_valid, accx_const_zero, accy_valid, accz_valid],
        [ppg_valid, accy_valid, accx_const_zero, accz_valid],
    ], dtype=np.float64)

    # AE and DCT reconstructions with different noise levels
    preds_ae = ref_windows + 0.01 * np.random.randn(*ref_windows.shape)
    preds_dct = ref_windows + 0.05 * np.random.randn(*ref_windows.shape)

    df_ae = evaluate_dataset_metrics(ref_windows, preds_ae)
    df_dct = evaluate_dataset_metrics(ref_windows, preds_dct)

    mask_joint = compute_joint_valid_mask(df_ae, df_dct, metric_col="valid_prdn")

    if hasattr(df_ae, "values"):
        valid_ae = df_ae["valid_prdn"].values
        valid_dct = df_dct["valid_prdn"].values
        mask_val = mask_joint.values
    else:
        valid_ae = [r["valid_prdn"] for r in df_ae]
        valid_dct = [r["valid_prdn"] for r in df_dct]
        mask_val = mask_joint

    np.testing.assert_array_equal(valid_ae, valid_dct)
    np.testing.assert_array_equal(mask_val, valid_ae)

    print("[PASS] Case 3: AE and DCT use identical shared valid mask across dataset")


def test_case_4_per_channel_independent_calculation():
    """
    CASE 4: Metrics must be computed separately for all 4 channels (PPG, ACCx, ACCy, ACCz).
    """
    t = np.linspace(0, 10, 512)
    ref = np.array([
        np.sin(t) + 10.0,
        np.cos(t) + 5.0,
        np.sin(2 * t) + 2.0,
        np.cos(2 * t) + 9.8
    ], dtype=np.float64)

    # Errors: PPG=0, ACCx=0.1, ACCy=0.2, ACCz=0.3
    pred = np.array([
        ref[0],
        ref[1] + 0.1,
        ref[2] + 0.2,
        ref[3] + 0.3
    ], dtype=np.float64)

    results = evaluate_window_metrics(ref, pred)

    assert len(results) == 4
    channels = [r["channel"] for r in results]
    assert channels == ["PPG", "ACCx", "ACCy", "ACCz"]

    # Each channel has distinct RMSE matching its specific channel noise
    assert abs(results[0]["rmse"] - 0.0) < 1e-6, "PPG channel RMSE mismatch"
    assert abs(results[1]["rmse"] - 0.1) < 1e-4, "ACCx channel RMSE mismatch"
    assert abs(results[2]["rmse"] - 0.2) < 1e-4, "ACCy channel RMSE mismatch"
    assert abs(results[3]["rmse"] - 0.3) < 1e-4, "ACCz channel RMSE mismatch"

    print("[PASS] Case 4: Metrics calculated independently for PPG, ACCx, ACCy, ACCz")


def test_case_5_normalize_identity_denormalize_pipeline():
    """
    CASE 5: normalize -> model identity -> denormalize must match reference within tolerance.
    """
    np.random.seed(42)
    # 5 windows of 4x512 physical signals
    t = np.linspace(0, 10, 512)
    x_phys = np.zeros((5, 4, 512), dtype=np.float32)
    for i in range(5):
        x_phys[i, 0] = np.sin(t + i) * 2.0 + 50.0   # PPG
        x_phys[i, 1] = np.cos(t + i) * 0.5 + 0.1    # ACCx
        x_phys[i, 2] = np.sin(2*t + i) * 0.5 - 0.2  # ACCy
        x_phys[i, 3] = np.cos(2*t + i) * 1.0 + 9.8  # ACCz

    # Step 1: compute norm stats
    stats = compute_norm_stats(x_phys)

    # Step 2: normalize
    x_norm = normalize(x_phys, stats)

    # Step 3: identity model prediction
    x_rec_norm = x_norm.copy()

    # Step 4: denormalize back to physical domain
    x_rec_phys = denormalize(x_rec_norm, stats)

    # Check maximum physical reconstruction error
    max_diff = np.max(np.abs(x_phys - x_rec_phys))
    assert max_diff < 1e-5, f"Expected physical reconstruction diff < 1e-5, got {max_diff}"

    # Step 5: Evaluate dataset metrics with norm_stats
    df_or_list = evaluate_dataset_metrics(x_norm, x_rec_norm, norm_stats=stats)

    records = df_or_list.to_dict("records") if hasattr(df_or_list, "to_dict") else df_or_list
    for row in records:
        assert row["rmse"] < 1e-5, f"Expected RMSE ~ 0, got {row['rmse']}"
        assert row["prd"] < 1e-4, f"Expected PRD ~ 0, got {row['prd']}"
        assert row["prdn"] < 1e-4, f"Expected PRDN ~ 0, got {row['prdn']}"
        assert row["valid_prd"] is True
        assert row["valid_prdn"] is True

    print(f"[PASS] Case 5: normalize -> identity -> denormalize pipeline matched reference (max_diff={max_diff:.2e})")



def test_c1_denominator_validity_threshold_with_sigma_train():
    """
    Test C1 Spec Denominator-Near-Zero Threshold Rule:
    threshold_c = N * 1e-12 * (sigma_c ** 2)
    where N = len(x_ref) = 512, sigma_c = Train std of channel c.
    """
    n = 512
    sigma_c = 10.0  # Train std = 10.0
    threshold_expected = 512 * 1e-12 * (10.0 ** 2)  # 512 * 1e-12 * 100 = 5.12e-9

    # Signal with centered variance = 4.0e-9 (below threshold_expected -> invalid PRDN)
    t = np.linspace(0, 1, n)
    x_ref_small = np.sin(2 * np.pi * t) * math.sqrt(2.0 * 4.0e-9 / n) + 5.0
    x_pred_small = x_ref_small + 1e-5

    m_small = compute_channel_metrics(x_ref_small, x_pred_small, channel_name="PPG", sigma_train=sigma_c)
    assert m_small["valid_prdn"] is False, f"Expected valid_prdn=False for centered energy below threshold"
    assert math.isnan(m_small["prdn"]), f"Expected NaN PRDN"

    # Signal with centered variance = 1.0e-7 (above threshold_expected -> valid PRDN)
    x_ref_large = np.sin(2 * np.pi * t) * math.sqrt(2.0 * 1.0e-7 / n) + 5.0
    x_pred_large = x_ref_large + 1e-5

    m_large = compute_channel_metrics(x_ref_large, x_pred_large, channel_name="PPG", sigma_train=sigma_c)
    assert m_large["valid_prdn"] is True, f"Expected valid_prdn=True for centered energy above threshold"
    assert math.isfinite(m_large["prdn"]), f"Expected finite PRDN"

    # Verify AE and DCT produce identical validity flags on same reference signal
    ae_pred = x_ref_large + 0.01
    dct_pred = x_ref_large + 0.05
    m_ae = compute_channel_metrics(x_ref_large, ae_pred, channel_name="PPG", sigma_train=sigma_c)
    m_dct = compute_channel_metrics(x_ref_large, dct_pred, channel_name="PPG", sigma_train=sigma_c)
    assert m_ae["valid_prd"] == m_dct["valid_prd"]
    assert m_ae["valid_prdn"] == m_dct["valid_prdn"]


def run_all_task_19_tests():
    """Run all test cases and print clear PASS/FAIL report."""
    print("\n==========================================================")
    print("      TASK 19 — EVALUATION METRICS VERIFICATION SUITE     ")
    print("==========================================================")

    test_cases = [
        ("Case 1: prediction == reference (PRD=0, PRDN=0, RMSE=0)", test_case_1_prediction_equals_reference),
        ("Case 2: reference near constant / zero-energy", test_case_2_near_constant_zero_energy_reference),
        ("Case 3: AE and DCT shared valid mask", test_case_3_ae_dct_shared_valid_mask),
        ("Case 4: 4 channels calculated independently", test_case_4_per_channel_independent_calculation),
        ("Case 5: normalize -> model identity -> denormalize pipeline", test_case_5_normalize_identity_denormalize_pipeline),
        ("Case 6: C1 spec denominator threshold with sigma_train", test_c1_denominator_validity_threshold_with_sigma_train),
    ]

    passed_count = 0
    failed_count = 0

    for title, test_func in test_cases:
        try:
            test_func()
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {title}: {e}")
            failed_count += 1

    print("==========================================================")
    print(f"RESULTS SUMMARY: {passed_count}/{len(test_cases)} PASSED | {failed_count} FAILED")
    print("==========================================================")
    if failed_count > 0:
        raise RuntimeError(f"{failed_count} test cases failed in TASK 19 suite!")


if __name__ == "__main__":
    run_all_task_19_tests()


