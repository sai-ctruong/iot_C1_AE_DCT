"""Unit test for TASK 10 — AE vs DCT Equal Byte Budget Matching."""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    compute_equal_byte_budget,
    get_equal_byte_budget_table,
    print_equal_byte_budget_table,
)
from src.codec import encode_ae_bytes, encode_dct_bytes, decode_dct_bytes

def test_equal_byte_budget_calculations():
    """Verify M=512->K=341, M=256->K=170, M=128->K=85, M=64->K=42 and B_AE == B_DCT."""
    table = get_equal_byte_budget_table()
    assert len(table) == 4

    expected = [
        {"db": 16, "M": 512, "K": 341, "pad": 2, "B": 2064},
        {"db": 8,  "M": 256, "K": 170, "pad": 4, "B": 1040},
        {"db": 4,  "M": 128, "K": 85,  "pad": 2, "B": 528},
        {"db": 2,  "M": 64,  "K": 42,  "pad": 4, "B": 272},
    ]

    for row, exp in zip(table, expected):
        assert row["db"] == exp["db"]
        assert row["M"] == exp["M"]
        assert row["K_equal_byte"] == exp["K"]
        assert row["padding"] == exp["pad"]
        assert row["B_AE"] == exp["B"]
        assert row["B_DCT"] == exp["B"]
        assert row["B_AE"] == row["B_DCT"]

    print("[TEST] Equal byte budget calculations verified!")


def test_codec_bitstream_length_equality():
    """Verify actual binary bitstreams for AE and padded DCT have identical length."""
    db_list = [16, 8, 4, 2]

    for db in db_list:
        budget = compute_equal_byte_budget(db)
        m = budget["M"]
        k = budget["K_equal_byte"]
        pad = budget["padding"]

        # Dummy AE latent and DCT topk
        latent = np.random.randn(m).astype(np.float32)
        topk_vals = np.random.randn(k).astype(np.float32)
        topk_idxs = np.random.randint(0, 2048, size=k).astype(np.uint16)

        b_ae = encode_ae_bytes(latent, d_b=db)
        b_dct = encode_dct_bytes(topk_vals, topk_idxs, d_b=db, pad_bytes=pad)

        assert len(b_ae) == budget["B_AE"], f"AE bitstream len {len(b_ae)} != {budget['B_AE']}"
        assert len(b_dct) == budget["B_DCT"], f"DCT bitstream len {len(b_dct)} != {budget['B_DCT']}"
        assert len(b_ae) == len(b_dct), f"Equal byte size failure: len(AE)={len(b_ae)} != len(DCT)={len(b_dct)}"

        # Verify decode ignores padding cleanly
        dec_vals, dec_idxs, header = decode_dct_bytes(b_dct)
        assert len(dec_vals) == k
        assert len(dec_idxs) == k
        assert header["pad_bytes"] == pad
        np.testing.assert_allclose(topk_vals, dec_vals)
        np.testing.assert_array_equal(topk_idxs, dec_idxs)

    print("[TEST] Bitstream binary byte equality (len(b_ae) == len(b_dct)) verified!")


if __name__ == "__main__":
    print("=== TASK 10 - AE vs DCT EQUAL BYTE BUDGET VERIFICATION ===")
    test_equal_byte_budget_calculations()
    test_codec_bitstream_length_equality()
    print_equal_byte_budget_table()
    print("=== ALL TASK 10 TESTS PASSED CLEANLY ===")
