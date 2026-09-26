"""Unit test for TASK 9 — Reference Bitstream Codec & Binary Serialization."""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codec import (
    HEADER_SIZE,
    encode_ae_bytes,
    decode_ae_bytes,
    encode_dct_bytes,
    decode_dct_bytes,
    C1Codec,
)

def test_header_and_ae_byte_length():
    """Verify AE header unpacking and byte length formula: B_AE = 16 + 4 * M."""
    m_values = [64, 128, 256, 512]
    d_b_values = [2, 4, 8, 16]

    for m, db in zip(m_values, d_b_values):
        latent = np.random.randn(m).astype(np.float32)
        bitstream = encode_ae_bytes(latent, d_b=db, profile_id=42)

        expected_bytes = 16 + 4 * m
        assert len(bitstream) == expected_bytes, f"AE bitstream length mismatch: {len(bitstream)} != {expected_bytes}"

        decoded_latent, header = decode_ae_bytes(bitstream)

        assert header["magic"] == "C1AE"
        assert header["codec_id"] == 1
        assert header["d_b"] == db
        assert header["count"] == m
        assert header["profile_id"] == 42
        assert header["payload_bytes"] == 4 * m
        assert header["header_size"] == 16

        np.testing.assert_allclose(latent, decoded_latent, rtol=1e-6, atol=1e-6)

    print("[TEST] AE header and exact byte length B_AE = 16 + 4*M passed!")


def test_dct_header_and_byte_length():
    """Verify DCT header unpacking and byte length formula: B_DCT = 16 + 6 * K."""
    k_values = [64, 128, 256, 512]
    d_b_values = [2, 4, 8, 16]

    for k, db in zip(k_values, d_b_values):
        topk_vals = np.random.randn(k).astype(np.float32)
        topk_idxs = np.random.randint(0, 2048, size=k).astype(np.uint16)

        bitstream = encode_dct_bytes(topk_vals, topk_idxs, d_b=db, profile_id=101)

        expected_bytes = 16 + 6 * k
        assert len(bitstream) == expected_bytes, f"DCT bitstream length mismatch: {len(bitstream)} != {expected_bytes}"

        decoded_vals, decoded_idxs, header = decode_dct_bytes(bitstream)

        assert header["magic"] == "C1DC"
        assert header["codec_id"] == 2
        assert header["d_b"] == db
        assert header["count"] == k
        assert header["profile_id"] == 101
        assert header["payload_bytes"] == 6 * k
        assert header["header_size"] == 16

        np.testing.assert_allclose(topk_vals, decoded_vals, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(topk_idxs, decoded_idxs)

    print("[TEST] DCT header and exact byte length B_DCT = 16 + 6*K passed!")


def test_codec_wrapper_class():
    """Test C1Codec class helper methods."""
    latent = np.array([1.5, -0.5, 3.2, 0.0], dtype=np.float32)
    b_ae = C1Codec.encode_ae(latent, d_b=2)
    rec_latent, _ = C1Codec.decode_ae(b_ae)
    np.testing.assert_allclose(latent, rec_latent)

    vals = np.array([10.0, 20.0], dtype=np.float32)
    idxs = np.array([5, 12], dtype=np.uint16)
    b_dct = C1Codec.encode_dct(vals, idxs, d_b=2)
    rec_vals, rec_idxs, _ = C1Codec.decode_dct(b_dct)
    np.testing.assert_allclose(vals, rec_vals)
    np.testing.assert_array_equal(idxs, rec_idxs)

    print("[TEST] C1Codec wrapper class passed!")


if __name__ == "__main__":
    print("=== TASK 9 - REFERENCE BITSTREAM CODEC VERIFICATION ===")
    test_header_and_ae_byte_length()
    test_dct_header_and_byte_length()
    test_codec_wrapper_class()
    print("=== ALL CODEC TESTS PASSED CLEANLY ===")
