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
    test_compute_cr_dim_arrays_and_scalars()
    test_invalid_encoded_values_raises_error()
    print_budget_mapping_table()
    print("=== ALL TASK 8 METRICS TESTS PASSED CLEANLY ===")
