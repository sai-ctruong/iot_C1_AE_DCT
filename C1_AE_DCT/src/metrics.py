"""Compression Metrics and Dimension/Byte Budget Allocation module for C1_AE_DCT."""

from typing import Dict, Any, List, Tuple, Union
import numpy as np

RAW_SAMPLES_PER_WINDOW = 4 * 512  # 2048 raw values
RAW_64HZ_VALUES = 4 * 512        # 2048 raw values (4 channels x 512 samples)
RAW_BYTES_64HZ = 4 * 512 * 4       # 8192 bytes (float32 at 64Hz)
RAW_BYTES_NATIVE = (512 + 256 * 3) * 4  # 5120 bytes (PPG 64Hz + ACC 32Hz float32)

BUDGET_CONFIGS = [
    {"db": 16, "M": 512, "K": 512, "CR_dim": 4.0},
    {"db": 8,  "M": 256, "K": 256, "CR_dim": 8.0},
    {"db": 4,  "M": 128, "K": 128, "CR_dim": 16.0},
    {"db": 2,  "M": 64,  "K": 64,  "CR_dim": 32.0},
]

def compute_cr_dim(
    raw_values: Union[int, np.ndarray],
    encoded_values: Union[int, np.ndarray]
) -> float:
    """
    Compute Dimension Compression Ratio (CR_dim).

    CR_dim = N_raw / N_encoded

    NOTE ON INTENT & LIMITATIONS:
    CR_dim compares ONLY the count of numeric values/elements (N_raw / N_encoded).
    It DOES NOT represent actual physical transmission payload byte size (CR_bit),
    as DCT requires coefficient index encoding while Autoencoders generate continuous
    dense latent vectors. Bit-level compression ratio (CR_bit) is computed separately.
    """
    if isinstance(raw_values, np.ndarray):
        n_raw = raw_values.size
    else:
        n_raw = int(raw_values)

    if isinstance(encoded_values, np.ndarray):
        n_encoded = encoded_values.size
    else:
        n_encoded = int(encoded_values)

    if n_encoded <= 0:
        raise ValueError(f"Encoded values count must be positive, got {n_encoded}")

    cr_dim = float(n_raw) / float(n_encoded)
    return cr_dim


def compute_equal_byte_budget(db: int) -> Dict[str, Any]:
    """
    Compute exact byte-matched budget for AE and DCT for a given budget factor d_b.

    Formulas:
    - M = 32 * d_b
    - B_AE = 16 + 4 * M bytes
    - K_equal_byte = floor(4 * M / 6)
    - padding = (4 * M) - (6 * K_equal_byte)
    - B_DCT = 16 + 6 * K_equal_byte + padding = B_AE
    """
    M = 32 * db
    K_equal_dim = M
    payload_ae = 4 * M
    B_AE = 16 + payload_ae

    K_equal_byte = (4 * M) // 6
    pad_bytes = (4 * M) - (6 * K_equal_byte)
    B_DCT = 16 + (6 * K_equal_byte) + pad_bytes

    assert B_AE == B_DCT, f"Byte budget mismatch: B_AE ({B_AE}) != B_DCT ({B_DCT})"

    CR_byte_64 = RAW_BYTES_64HZ / float(B_AE)
    CR_byte_native = RAW_BYTES_NATIVE / float(B_AE)

    return {
        "db": db,
        "M": M,
        "K_equal_dim": K_equal_dim,
        "K_equal_byte": K_equal_byte,
        "padding": pad_bytes,
        "B_AE": B_AE,
        "B_DCT": B_DCT,
        "CR_byte_64": CR_byte_64,
        "CR_byte_native": CR_byte_native,
    }


def get_equal_byte_budget_table() -> List[Dict[str, Any]]:
    """Return complete equal-byte budget comparison table for d_b in [16, 8, 4, 2]."""
    return [compute_equal_byte_budget(db) for db in [16, 8, 4, 2]]


def print_equal_byte_budget_table() -> None:
    """Print formatted ASCII equal byte budget table for AE vs DCT."""
    table = get_equal_byte_budget_table()

    header = (
        f"| {'d_b':<4} | {'M':<5} | {'K_dim':<6} | {'K_byte':<6} | {'Pad':<4} | "
        f"{'B_AE':<6} | {'B_DCT':<6} | {'CR_byte_64':<11} | {'CR_byte_native':<15} |"
    )
    separator = (
        "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 6 + "+"
        + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 13 + "+" + "-" * 17 + "+"
    )

    print("\n=== EQUAL BYTE BUDGET MATCHING TABLE (AE vs DCT) ===")
    print("NOTE: CR_byte reflects ACTUAL physical bitstream size including 16-byte header & padding.")
    print("      It is slightly smaller than nominal 4x/8x/16x/32x due to header overhead.")
    print(separator)
    print(header)
    print(separator)

    for row in table:
        print(
            f"| {row['db']:<4} | {row['M']:<5} | {row['K_equal_dim']:<6} | {row['K_equal_byte']:<6} | "
            f"{row['padding']:<4} | {row['B_AE']:<6} | {row['B_DCT']:<6} | "
            f"{row['CR_byte_64']:<10.2f}x | {row['CR_byte_native']:<14.2f}x |"
        )

    print(separator)


def get_budget_mapping_table() -> List[Dict[str, Any]]:
    """Return budget mapping table comparing db, M (AE latent dim), K (DCT top-K), and CR_dim."""
    return BUDGET_CONFIGS


def print_budget_mapping_table() -> None:
    """Print formatted ASCII budget mapping table for AE vs DCT dimension compression ratio."""
    table = get_budget_mapping_table()
    header = f"| {'d_b':<6} | {'M (AE Latent)':<15} | {'K (DCT Top-K)':<15} | {'CR_dim':<10} |"
    separator = "+" + "-" * 8 + "+" + "-" * 17 + "+" + "-" * 17 + "+" + "-" * 12 + "+"

    print("\n=== BUDGET ALLOCATION TABLE (AE vs DCT CR_dim) ===")
    print("[DISCLAIMER] CR_dim ONLY compares element counts (N_raw / N_encoded).")
    print("             It DOES NOT reflect actual transmission bitstream byte size (CR_bit).")
    print(separator)
    print(header)
    print(separator)

    for row in table:
        print(f"| {row['db']:<6} | {row['M']:<15} | {row['K']:<15} | {row['CR_dim']:<9.1f}x |")

    print(separator)


if __name__ == "__main__":
    print_budget_mapping_table()
    print_equal_byte_budget_table()
