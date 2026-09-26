"""Reference Bitstream Codec and Binary Serialization module for C1_AE_DCT."""

import struct
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np

# Header specification constants
HEADER_FORMAT = "<4sBBHII"  # Little-endian: magic[4], codec_id[uint8], d_b[uint8], count[uint16], profile_id[uint32], payload_bytes[uint32]
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # Exactly 16 bytes

MAGIC_AE = b"C1AE"
MAGIC_DCT = b"C1DC"

CODEC_ID_AE = 1
CODEC_ID_DCT = 2


def encode_ae_bytes(
    latent: np.ndarray,
    d_b: int,
    profile_id: int = 0
) -> bytes:
    """
    Encode Autoencoder latent vector into binary bitstream with 16-byte header.

    Binary Format:
    - Header (16 bytes): magic="C1AE", codec_id=1, d_b, count=M, profile_id, payload_bytes=4*M
    - Payload (4*M bytes): M float32 values in little-endian

    Total Bitstream Length: B_AE = 16 + 4 * M bytes

    Parameters:
    -----------
    latent : np.ndarray
        AE latent vector of shape (M,) or (1, M) in float32.
    d_b : int
        Budget factor (e.g., 16, 8, 4, 2).
    profile_id : int
        Profile or configuration identifier (default: 0).

    Returns:
    --------
    bitstream : bytes
        Binary payload with exact length 16 + 4 * M.
    """
    arr = np.asarray(latent, dtype="<f4").ravel()
    m_count = len(arr)

    payload_bytes = m_count * 4
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_AE,
        CODEC_ID_AE,
        int(d_b),
        int(m_count),
        int(profile_id),
        int(payload_bytes)
    )

    payload = arr.tobytes()
    bitstream = header + payload

    expected_len = 16 + 4 * m_count
    assert len(bitstream) == expected_len, f"AE bitstream length mismatch: {len(bitstream)} != {expected_len}"

    return bitstream


def decode_ae_bytes(bitstream: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Decode binary bitstream into AE latent float32 vector and header metadata.

    Parameters:
    -----------
    bitstream : bytes
        Binary payload containing 16-byte header + float32 latent values.

    Returns:
    --------
    latent : np.ndarray
        Decoded 1D float32 array of shape (M,).
    header_info : Dict[str, Any]
        Header metadata dictionary.
    """
    if len(bitstream) < HEADER_SIZE:
        raise ValueError(f"Bitstream length ({len(bitstream)}) is smaller than header size ({HEADER_SIZE})")

    magic, codec_id, d_b, count, profile_id, payload_bytes = struct.unpack(HEADER_FORMAT, bitstream[:HEADER_SIZE])

    if magic != MAGIC_AE or codec_id != CODEC_ID_AE:
        raise ValueError(f"Invalid AE header magic/codec_id: magic={magic}, codec_id={codec_id}")

    expected_len = HEADER_SIZE + payload_bytes
    if len(bitstream) != expected_len:
        raise ValueError(f"Bitstream payload length mismatch: actual {len(bitstream)} != expected {expected_len}")

    latent = np.frombuffer(bitstream[HEADER_SIZE:expected_len], dtype="<f4").astype(np.float32)

    header_info = {
        "magic": magic.decode("ascii", errors="ignore"),
        "codec_id": codec_id,
        "d_b": d_b,
        "count": count,
        "profile_id": profile_id,
        "payload_bytes": payload_bytes,
        "header_size": HEADER_SIZE,
        "total_bytes": len(bitstream),
    }

    return latent, header_info


def encode_dct_bytes(
    topk_values: np.ndarray,
    topk_indices: np.ndarray,
    d_b: int,
    profile_id: int = 0,
    pad_bytes: int = 0
) -> bytes:
    """
    Encode DCT Top-K values and indices into binary bitstream with 16-byte header and optional byte padding.

    Binary Format:
    - Header (16 bytes): magic="C1DC", codec_id=2, d_b, count=K, profile_id, payload_bytes=6*K + pad_bytes
    - Payload (6*K + pad_bytes bytes):
      - K float32 values (4*K bytes)
      - K uint16 indices (2*K bytes)
      - pad_bytes zeros (pad_bytes bytes)

    Total Bitstream Length: B_DCT = 16 + 6 * K + pad_bytes bytes

    Parameters:
    -----------
    topk_values : np.ndarray
        Top-K coefficient values of shape (K,) in float32.
    topk_indices : np.ndarray
        Top-K coefficient indices of shape (K,) in uint16 (0..2047).
    d_b : int
        Budget factor (e.g., 16, 8, 4, 2).
    profile_id : int
        Profile identifier (default: 0).
    pad_bytes : int
        Number of padding bytes to align total byte length with AE budget.

    Returns:
    --------
    bitstream : bytes
        Binary payload with exact length 16 + 6 * K + pad_bytes.
    """
    vals = np.asarray(topk_values, dtype="<f4").ravel()
    idxs = np.asarray(topk_indices, dtype="<u2").ravel()

    k_count = len(vals)
    if len(idxs) != k_count:
        raise ValueError(f"Length of topk_values ({len(vals)}) != topk_indices ({len(idxs)})")

    payload_bytes = k_count * 6 + pad_bytes  # 4 bytes val + 2 bytes idx + pad_bytes
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_DCT,
        CODEC_ID_DCT,
        int(d_b),
        int(k_count),
        int(profile_id),
        int(payload_bytes)
    )

    vals_bytes = vals.tobytes()
    idxs_bytes = idxs.tobytes()
    padding_bytes = b"\x00" * pad_bytes

    bitstream = header + vals_bytes + idxs_bytes + padding_bytes

    expected_len = 16 + 6 * k_count + pad_bytes
    assert len(bitstream) == expected_len, f"DCT bitstream length mismatch: {len(bitstream)} != {expected_len}"

    return bitstream


def decode_dct_bytes(bitstream: bytes) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Decode binary bitstream into DCT Top-K float32 values, uint16 indices, and header metadata.

    Parameters:
    -----------
    bitstream : bytes
        Binary payload containing 16-byte header + float32 values + uint16 indices (+ optional padding).

    Returns:
    --------
    topk_values : np.ndarray
        Decoded 1D float32 array of values.
    topk_indices : np.ndarray
        Decoded 1D uint16 array of indices.
    header_info : Dict[str, Any]
        Header metadata dictionary.
    """
    if len(bitstream) < HEADER_SIZE:
        raise ValueError(f"Bitstream length ({len(bitstream)}) is smaller than header size ({HEADER_SIZE})")

    magic, codec_id, d_b, count, profile_id, payload_bytes = struct.unpack(HEADER_FORMAT, bitstream[:HEADER_SIZE])

    if magic != MAGIC_DCT or codec_id != CODEC_ID_DCT:
        raise ValueError(f"Invalid DCT header magic/codec_id: magic={magic}, codec_id={codec_id}")

    expected_len = HEADER_SIZE + payload_bytes
    if len(bitstream) != expected_len:
        raise ValueError(f"Bitstream payload length mismatch: actual {len(bitstream)} != expected {expected_len}")

    k_count = count
    vals_len = 4 * k_count
    idxs_len = 2 * k_count

    vals_start = HEADER_SIZE
    vals_end = vals_start + vals_len
    idxs_end = vals_end + idxs_len

    topk_values = np.frombuffer(bitstream[vals_start:vals_end], dtype="<f4").astype(np.float32)
    topk_indices = np.frombuffer(bitstream[vals_end:idxs_end], dtype="<u2").astype(np.uint16)

    pad_bytes = payload_bytes - (vals_len + idxs_len)

    header_info = {
        "magic": magic.decode("ascii", errors="ignore"),
        "codec_id": codec_id,
        "d_b": d_b,
        "count": k_count,
        "profile_id": profile_id,
        "payload_bytes": payload_bytes,
        "pad_bytes": pad_bytes,
        "header_size": HEADER_SIZE,
        "total_bytes": len(bitstream),
    }

    return topk_values, topk_indices, header_info


class C1Codec:
    """Class wrapper for reference binary codec."""
    @staticmethod
    def encode_ae(latent: np.ndarray, d_b: int) -> bytes:
        return encode_ae_bytes(latent, d_b)

    @staticmethod
    def decode_ae(bitstream: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        return decode_ae_bytes(bitstream)

    @staticmethod
    def encode_dct(values: np.ndarray, indices: np.ndarray, d_b: int, pad_bytes: int = 0) -> bytes:
        return encode_dct_bytes(values, indices, d_b, pad_bytes=pad_bytes)

    @staticmethod
    def decode_dct(bitstream: bytes) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return decode_dct_bytes(bitstream)
