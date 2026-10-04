"""Unit & Integration Tests for Task 4: Fixed Low-Frequency DCT Baseline (DCT-Fixed-LF)."""

import os
import sys
import gzip
import pytest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_dct import (
    get_fixed_lf_channel_allocation,
    get_fixed_lf_indices,
    dct_encode_fixed_lf,
    dct_decode_fixed_lf,
    dct_reconstruct_fixed_lf,
    BaselineDCTFixedLF,
    CHANNEL_NAMES
)
from src.codec import (
    encode_dct_fixed_lf_bytes,
    decode_dct_fixed_lf_bytes,
    C1Codec,
    MAGIC_DCT_FIXED,
    CODEC_ID_DCT_FIXED
)


def test_fixed_positions_deterministic():
    """1. Test that get_fixed_lf_indices is 100% deterministic across multiple invocations."""
    for K in [512, 256, 128, 64, 42]:
        idx1 = get_fixed_lf_indices(K)
        idx2 = get_fixed_lf_indices(K)
        np.testing.assert_array_equal(idx1, idx2)
        assert len(idx1) == K


def test_positions_data_independent():
    """2. Test that positions are static and independent of input data values."""
    np.random.seed(42)
    win1 = np.random.randn(4, 512)
    win2 = np.random.randn(4, 512) * 100.0

    K = 128
    val1, counts1, _ = dct_encode_fixed_lf(win1, k=K)
    val2, counts2, _ = dct_encode_fixed_lf(win2, k=K)

    # Indices are identical regardless of signal values
    idx1 = get_fixed_lf_indices(K)
    idx2 = get_fixed_lf_indices(K)
    np.testing.assert_array_equal(idx1, idx2)
    assert counts1 == counts2 == {"PPG": 32, "ACCx": 32, "ACCy": 32, "ACCz": 32}


def test_k_exact_and_channel_allocation():
    """3. Test exact K counts and fixed channel allocation remainder logic."""
    # Test even K=256
    alloc256 = get_fixed_lf_channel_allocation(256)
    assert sum(alloc256.values()) == 256
    assert alloc256 == {"PPG": 64, "ACCx": 64, "ACCy": 64, "ACCz": 64}

    # Test odd remainder K=42 (42 // 4 = 10, rem = 2 -> PPG=11, ACCx=11, ACCy=10, ACCz=10)
    alloc42 = get_fixed_lf_channel_allocation(42)
    assert sum(alloc42.values()) == 42
    assert alloc42 == {"PPG": 11, "ACCx": 11, "ACCy": 10, "ACCz": 10}


def test_dct_idct_full_spectrum_reconstruction():
    """4. Test that DCT/IDCT with K=2048 reconstructs signal with numerical precision."""
    np.random.seed(123)
    orig_signal = np.random.randn(4, 512).astype(np.float32)

    rec_signal, vals, counts = dct_reconstruct_fixed_lf(orig_signal, k=2048)
    assert len(vals) == 2048
    np.testing.assert_allclose(rec_signal, orig_signal, rtol=1e-5, atol=1e-5)


def test_serialized_bytes_formula_and_no_indices():
    """5 & 6. Test binary bitstream format, header, exact payload length formula 16 + 4K, and zero index payload."""
    for K in [512, 256, 128, 64]:
        vals = np.random.randn(K).astype(np.float32)
        bitstream = encode_dct_fixed_lf_bytes(vals, d_b=8, profile_id=0)

        expected_bytes = 16 + 4 * K
        assert len(bitstream) == expected_bytes, f"Bitstream size {len(bitstream)} != expected {expected_bytes}"

        # Decode bitstream
        decoded_vals, header_info = decode_dct_fixed_lf_bytes(bitstream)
        np.testing.assert_allclose(decoded_vals, vals, rtol=1e-6)
        assert header_info["magic"] == "C1DF"
        assert header_info["codec_id"] == CODEC_ID_DCT_FIXED
        assert header_info["count"] == K
        assert header_info["payload_bytes"] == 4 * K


def test_class_wrapper():
    """Test BaselineDCTFixedLF class wrapper methods."""
    win = np.random.randn(4, 512)
    model = BaselineDCTFixedLF(k=128)
    vals, counts, sparse_dct = model.encode(win)
    rec = model.decode(sparse_dct)
    rec2, vals2, counts2 = model.reconstruct(win)

    np.testing.assert_allclose(rec, rec2, rtol=1e-6)
    assert len(vals) == 128


def test_no_synthetic_data_in_fixed_lf_results():
    """7. Test that real evaluation results exist, have valid schema, and contain no NaNs."""
    ext_dir = PROJECT_ROOT / "results" / "extensions"
    gz_detail = ext_dir / "dct_fixed_lf_detail.csv.gz"

    if not gz_detail.exists():
        pytest.skip("Run python -m src.run_dct_fixed_lf first to generate results")

    with gzip.open(gz_detail, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, keep_default_na=False, low_memory=False)

    assert len(df) > 0, "Fixed-LF detail CSV is empty!"
    assert not df["PRD"].isnull().any(), "NaNs found in PRD results!"
    assert not df["RMSE"].isnull().any(), "NaNs found in RMSE results!"
    assert set(df["method"].unique()) == {"DCT-Fixed-LF"}

