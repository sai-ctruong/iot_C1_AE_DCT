"""Unit test for TASK 8 — CR_dim and Budget Allocation Mapping."""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    compute_cr_dim,
    get_budget_mapping_table,
    print_budget_mapping_table,
    RAW_SAMPLES_PER_WINDOW,
    RAW_64HZ_VALUES,
)

def test_cr_dim_budget_mapping_table():
    """Verify budget configurations for db = [16, 8, 4, 2] and corresponding M, K, CR_dim."""
    table = get_budget_mapping_table()
    assert len(table) == 4

    expected = [
        {"db": 16, "M": 512, "K": 512, "CR_dim": 4.0},
        {"db": 8,  "M": 256, "K": 256, "CR_dim": 8.0},
        {"db": 4,  "M": 128, "K": 128, "CR_dim": 16.0},
        {"db": 2,  "M": 64,  "K": 64,  "CR_dim": 32.0},
    ]

    for row, exp in zip(table, expected):
        assert row["db"] == exp["db"]
        assert row["M"] == exp["M"]
        assert row["K"] == exp["K"]
        assert row["CR_dim"] == exp["CR_dim"]

        # Calculate using compute_cr_dim
        cr_calc = compute_cr_dim(RAW_SAMPLES_PER_WINDOW, row["M"])
        assert cr_calc == exp["CR_dim"], f"Mismatch for db={row['db']}: {cr_calc} != {exp['CR_dim']}"

    print("[TEST] Budget mapping table verification passed!")


def test_c1_spec_cr_dim_exact_assertions():
    """
    Explicitly test C1 specification for CR_dim calculation:
    RAW_64HZ_VALUES = 4 * 512 = 2048.
    AE: CR_dim = 2048 / M
      M=512 -> 4.0
      M=256 -> 8.0
      M=128 -> 16.0
      M=64  -> 32.0
    DCT: CR_dim = 2048 / K
    """
    assert RAW_64HZ_VALUES == 2048
    assert RAW_SAMPLES_PER_WINDOW == 2048

    # AE assertions
    ae_map = {512: 4.0, 256: 8.0, 128: 16.0, 64: 32.0}
    for M, expected_cr_dim in ae_map.items():
        cr = compute_cr_dim(RAW_64HZ_VALUES, M)
        assert cr == expected_cr_dim, f"AE CR_dim mismatch for M={M}: {cr} != {expected_cr_dim}"

    # DCT assertions (Equal-dim K)
    dct_map = {512: 4.0, 256: 8.0, 128: 16.0, 64: 32.0}
    for K, expected_cr_dim in dct_map.items():
        cr = compute_cr_dim(RAW_64HZ_VALUES, K)
        assert cr == expected_cr_dim, f"DCT CR_dim mismatch for K={K}: {cr} != {expected_cr_dim}"

    # DCT assertions (Equal-byte K)
    # K_equal_byte = [341, 170, 85, 42]
    eq_byte_map = {
        341: 2048.0 / 341.0,
        170: 2048.0 / 170.0,
        85: 2048.0 / 85.0,
        42: 2048.0 / 42.0,
    }
    for K, expected_cr_dim in eq_byte_map.items():
        cr = compute_cr_dim(RAW_64HZ_VALUES, K)
        assert abs(cr - expected_cr_dim) < 1e-6

    print("[TEST] C1 spec CR_dim exact assertions passed!")


def test_compute_cr_dim_arrays_and_scalars():
    """Verify compute_cr_dim works with integer counts and numpy arrays."""
    raw_window = np.zeros((4, 512), dtype=np.float32)  # 2048 elements
    encoded_latent = np.zeros((128,), dtype=np.float32)  # 128 elements

    cr_array = compute_cr_dim(raw_window, encoded_latent)
    assert cr_array == 16.0

    cr_scalar = compute_cr_dim(2048, 128)
    assert cr_scalar == 16.0

    print("[TEST] compute_cr_dim for arrays and scalars passed!")


def test_invalid_encoded_values_raises_error():
    """Verify compute_cr_dim raises ValueError for zero or negative encoded size."""
    with pytest.raises(ValueError):
        compute_cr_dim(2048, 0)

    print("[TEST] Invalid encoded values error handling passed!")


if __name__ == "__main__":
    print("=== TASK 8 - CR_dim AND BUDGET MAPPING VERIFICATION ===")
    test_cr_dim_budget_mapping_table()
    test_c1_spec_cr_dim_exact_assertions()
    test_compute_cr_dim_arrays_and_scalars()
    test_invalid_encoded_values_raises_error()
    print_budget_mapping_table()
    print("=== ALL TASK 8 METRICS TESTS PASSED CLEANLY ===")
