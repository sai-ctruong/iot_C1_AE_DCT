"""Comprehensive Automated Unit Test Suite for TASK 7 — DCT-II Baseline."""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_dct import dct_encode_topk, dct_decode, dct_reconstruct

def test_1_full_k_exact_reconstruction():
    """
    Test 1: K = 2048.
    DCT -> IDCT must reconstruct input almost exactly within floating point error tolerance.
    """
    np.random.seed(42)
    window = np.random.randn(4, 512).astype(np.float32)

    rec_signal, topk_vals, topk_idx, ch_counts = dct_reconstruct(window, k=2048)

    mse = np.mean((window - rec_signal) ** 2)
    print(f"[TEST 1] K=2048 Reconstruction MSE: {mse:.8e}")

    assert mse < 1e-9, f"Test 1 Failed: K=2048 MSE ({mse}) exceeds 1e-9 tolerance"
    np.testing.assert_allclose(
        window, rec_signal, rtol=1e-4, atol=1e-4,
        err_msg="Test 1 Failed: Floating point reconstruction tolerance exceeded!"
    )
    print("[TEST 1 PASSED] K=2048 exact reconstruction verified.")


def test_2_monotonicity_k1_gt_k2():
    """
    Test 2: Monotonicity check.
    For the same input window, if K1 > K2, then MSE(K1) must NOT be larger than MSE(K2)
    outside numerical floating-point tolerance.
    """
    np.random.seed(123)
    window = np.random.randn(4, 512).astype(np.float32)

    k_pairs = [(512, 256), (256, 128), (128, 64), (64, 32), (2048, 512)]

    for k1, k2 in k_pairs:
        assert k1 > k2, "Test setup error: k1 must be greater than k2"

        rec_k1, _, _, _ = dct_reconstruct(window, k=k1)
        rec_k2, _, _, _ = dct_reconstruct(window, k=k2)

        mse_k1 = np.mean((window - rec_k1) ** 2)
        mse_k2 = np.mean((window - rec_k2) ** 2)

        print(f"[TEST 2] K1={k1:<4} (MSE={mse_k1:.6f}) vs K2={k2:<4} (MSE={mse_k2:.6f})")

        # MSE(K1) <= MSE(K2) + 1e-7 tolerance
        assert mse_k1 <= mse_k2 + 1e-7, (
            f"Test 2 Failed: Monotonicity violated! MSE(K1={k1})={mse_k1:.6f} > MSE(K2={k2})={mse_k2:.6f}"
        )

    print("[TEST 2 PASSED] Monotonicity (MSE(K1) <= MSE(K2) for K1 > K2) verified.")


def test_3_deterministic_tie_break():
    """
    Test 3: Top-K deterministic tie-breaking.
    When coefficients have equal absolute value, smaller flat index j must be prioritized.
    """
    # Create window resulting in equal DCT magnitude across channels
    window = np.zeros((4, 512), dtype=np.float32)
    window[0, 0] = 10.0
    window[1, 0] = 10.0
    window[2, 0] = 10.0
    window[3, 0] = 10.0

    k = 3
    topk_vals, topk_idx, ch_counts, sparse_dct = dct_encode_topk(window, k=k)

    print(f"[TEST 3] Selected indices for equal magnitude coefficients: {topk_idx}")

    assert len(topk_idx) == k
    # Check that indices are strictly ordered ascending for equal magnitudes (smaller index first)
    for i in range(len(topk_idx) - 1):
        assert topk_idx[i] < topk_idx[i + 1], (
            f"Test 3 Failed: Tie-break rule violated! Index {topk_idx[i]} comes before {topk_idx[i+1]}"
        )

    print("[TEST 3 PASSED] Deterministic tie-breaking (smaller index j first) verified.")


def test_4_shape_preservation():
    """
    Test 4: Shape preservation.
    Encode -> decode must not change the shape [4, 512].
    """
    np.random.seed(99)
    single_window = np.random.randn(4, 512).astype(np.float32)
    batch_window = np.random.randn(5, 4, 512).astype(np.float32)

    # Test single window shape [4, 512]
    rec_single, _, _, _ = dct_reconstruct(single_window, k=128)
    assert rec_single.shape == (4, 512), (
        f"Test 4 Failed: Expected single window shape (4, 512), got {rec_single.shape}"
    )

    # Test batch windows shape [B, 4, 512]
    topk_vals, topk_idx, ch_counts, sparse_dct = dct_encode_topk(batch_window, k=128)
    rec_batch = dct_decode(sparse_dct)
    assert rec_batch.shape == (5, 4, 512), (
        f"Test 4 Failed: Expected batch window shape (5, 4, 512), got {rec_batch.shape}"
    )

    print(f"[TEST 4 PASSED] Shape preservation verified for [4, 512] and batch [B, 4, 512].")


def test_5_retained_coefficients_count():
    """
    Test 5: Retained coefficient count.
    Verify that total number of coefficients kept across channels equals exactly K.
    """
    np.random.seed(2026)
    window = np.random.randn(4, 512).astype(np.float32)

    test_k_values = [1, 16, 64, 128, 256, 512, 1024, 2048]

    for k in test_k_values:
        topk_vals, topk_idx, ch_counts, sparse_dct = dct_encode_topk(window, k=k)

        total_kept = sum(ch_counts.values())
        print(f"[TEST 5] Target K={k:<4} -> Retained count: {total_kept}, TopK values len: {len(topk_vals)}")

        assert len(topk_vals) == k, f"Test 5 Failed: len(topk_vals) ({len(topk_vals)}) != K ({k})"
        assert len(topk_idx) == k, f"Test 5 Failed: len(topk_idx) ({len(topk_idx)}) != K ({k})"
        assert total_kept == k, f"Test 5 Failed: sum(channel_counts) ({total_kept}) != K ({k})"

    print("[TEST 5 PASSED] Retained coefficient count == K verified across all test K values.")


if __name__ == "__main__":
    print("======================================================")
    print("    TASK 7 — AUTOMATED UNIT TEST SUITE FOR DCT-II     ")
    print("======================================================")
    
    tests = [
        test_1_full_k_exact_reconstruction,
        test_2_monotonicity_k1_gt_k2,
        test_3_deterministic_tie_break,
        test_4_shape_preservation,
        test_5_retained_coefficients_count,
    ]

    failed_tests = []
    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"\n[FAILURE DETECTED in {test_fn.__name__}]: {e}")
            failed_tests.append((test_fn.__name__, str(e)))
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR in {test_fn.__name__}]: {e}")
            failed_tests.append((test_fn.__name__, str(e)))

    print("======================================================")
    if not failed_tests:
        print(">>> ALL 5 DCT UNIT TESTS PASSED SUCCESSFULLY! <<<")
        print("======================================================")
    else:
        print(f">>> {len(failed_tests)} TEST(S) FAILED! <<<")
        for fn_name, err in failed_tests:
            print(f" - {fn_name}: {err}")
        print("======================================================")
        sys.exit(1)
