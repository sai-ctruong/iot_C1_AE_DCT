"""Discrete Cosine Transform (DCT-II) Baseline module for C1_AE_DCT project."""

from typing import Tuple, Dict, Any, Union, List, Optional
import numpy as np

try:
    import scipy.fft as fft
except ImportError:
    try:
        import scipy.fftpack as fft
    except ImportError:
        fft = None

CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]

def dct_encode_topk(
    window: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    """
    Perform DCT-II on 4-channel window signal and select Top-K global coefficients.

    Parameters:
    -----------
    window : np.ndarray
        Signal window matrix of shape (4, 512) or (N, 4, 512).
    k : int
        Number of Top-K coefficients to retain across all 2048 pooled coefficients (1 <= k <= 2048).

    Returns:
    --------
    topk_values : np.ndarray
        Array of top-K coefficient values.
    topk_indices : np.ndarray
        Array of flat indices [0..2047] corresponding to top-K coefficients.
    channel_counts : Dict[str, int]
        Number of retained Top-K coefficients per channel.
    sparse_dct : np.ndarray
        Sparse DCT array of shape (4, 512) with unselected coefficients zeroed out.
    """
    if fft is None:
        raise ImportError("scipy is required for DCT operations. Please install requirements.txt.")

    arr = np.asarray(window, dtype=np.float64)
    is_batch = False

    if arr.ndim == 2:
        if arr.shape != (4, 512):
            if arr.shape == (512, 4):
                arr = arr.T
            else:
                raise ValueError(f"Expected window shape (4, 512), got {arr.shape}")
    elif arr.ndim == 3:
        is_batch = True
        if arr.shape[1:] != (4, 512):
            raise ValueError(f"Expected batch shape (B, 4, 512), got {arr.shape}")
    else:
        raise ValueError(f"Input must be 2D (4, 512) or 3D (B, 4, 512), got shape {arr.shape}")

    num_channels, num_samples = 4, 512
    total_coeffs = num_channels * num_samples  # 2048

    if not (1 <= k <= total_coeffs):
        raise ValueError(f"k must be between 1 and {total_coeffs}, got {k}")

    # 1. Apply DCT-II independently on time axis for each channel
    dct_coeffs = fft.dct(arr, type=2, norm="ortho", axis=-1)

    if not is_batch:
        # Flatten 4x512 into 2048 1D array
        flat_coeffs = dct_coeffs.ravel()  # Shape: (2048,)
        abs_coeffs = np.abs(flat_coeffs)

        # Stable sort on negated absolute values (tie-break: smaller index j comes first)
        sorted_indices = np.argsort(-abs_coeffs, kind="stable")
        topk_indices = sorted_indices[:k]
        topk_values = flat_coeffs[topk_indices]

        # 5. Create sparse DCT matrix with zeroed non-selected coefficients
        sparse_dct = np.zeros_like(dct_coeffs)
        np.put(sparse_dct, topk_indices, topk_values)

        # Count retained coefficients per channel
        ch_indices = topk_indices // num_samples
        channel_counts = {
            ch: int(np.sum(ch_indices == ch_idx))
            for ch_idx, ch in enumerate(CHANNEL_NAMES)
        }

        return topk_values.astype(np.float32), topk_indices, channel_counts, sparse_dct.astype(np.float32)

    else:
        # Batch processing (B, 4, 512)
        batch_size = arr.shape[0]
        topk_values_list = []
        topk_indices_list = []
        sparse_dct_list = []
        channel_counts_list = []

        for b in range(batch_size):
            flat = dct_coeffs[b].ravel()
            abs_flat = np.abs(flat)
            sorted_idx = np.argsort(-abs_flat, kind="stable")
            top_idx = sorted_idx[:k]
            top_val = flat[top_idx]

            sp_dct = np.zeros_like(dct_coeffs[b])
            np.put(sp_dct, top_idx, top_val)

            ch_idx = top_idx // num_samples
            ch_counts = {
                ch: int(np.sum(ch_idx == idx))
                for idx, ch in enumerate(CHANNEL_NAMES)
            }

            topk_values_list.append(top_val)
            topk_indices_list.append(top_idx)
            sparse_dct_list.append(sp_dct)
            channel_counts_list.append(ch_counts)

        return (
            np.array(topk_values_list, dtype=np.float32),
            np.array(topk_indices_list),
            channel_counts_list,
            np.array(sparse_dct_list, dtype=np.float32),
        )


def dct_decode(sparse_dct: np.ndarray) -> np.ndarray:
    """
    Perform IDCT-II on sparse DCT matrix to reconstruct continuous time domain signal.

    Parameters:
    -----------
    sparse_dct : np.ndarray
        Sparse DCT coefficient matrix of shape (4, 512) or (B, 4, 512).

    Returns:
    --------
    reconstructed_signal : np.ndarray
        Reconstructed time domain signal matrix of shape (4, 512) or (B, 4, 512).
    """
    if fft is None:
        raise ImportError("scipy is required for IDCT operations. Please install requirements.txt.")

    arr = np.asarray(sparse_dct, dtype=np.float64)

    # 6. Inverse DCT-II on each channel to reconstruct signal
    reconstructed = fft.idct(arr, type=2, norm="ortho", axis=-1)
    return reconstructed.astype(np.float32)


def dct_reconstruct(
    window: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Full DCT-II Top-K compression and reconstruction pipeline.

    Parameters:
    -----------
    window : np.ndarray
        Input signal window matrix of shape (4, 512).
    k : int
        Number of Top-K coefficients to retain.

    Returns:
    --------
    reconstructed_signal : np.ndarray
        Reconstructed signal of shape (4, 512).
    topk_values : np.ndarray
        Top-K coefficient values.
    topk_indices : np.ndarray
        Top-K coefficient indices [0..2047].
    channel_counts : Dict[str, int]
        Count of retained coefficients per channel.
    """
    topk_values, topk_indices, channel_counts, sparse_dct = dct_encode_topk(window, k)
    reconstructed_signal = dct_decode(sparse_dct)
    return reconstructed_signal, topk_values, topk_indices, channel_counts


class BaselineDCT:
    """Class wrapper for DCT baseline model."""
    def __init__(self, k: int = 128):
        self.k = k

    def encode(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
        return dct_encode_topk(window, self.k)

    def decode(self, sparse_dct: np.ndarray) -> np.ndarray:
        return dct_decode(sparse_dct)

    def reconstruct(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        return dct_reconstruct(window, self.k)


def get_fixed_lf_channel_allocation(k: int) -> Dict[str, int]:
    """
    Distribute total budget K across 4 channels (PPG, ACCx, ACCy, ACCz) as evenly as possible.
    Remainder r = K % 4 is assigned to channels in fixed order: PPG, ACCx, ACCy, ACCz.

    Parameters:
    -----------
    k : int
        Total number of DCT coefficients to retain (1 <= k <= 2048).

    Returns:
    --------
    channel_counts : Dict[str, int]
        Dictionary mapping each channel name to its allocated coefficient count.
    """
    num_channels = len(CHANNEL_NAMES)
    base = k // num_channels
    rem = k % num_channels

    counts = {
        ch: base + (1 if i < rem else 0)
        for i, ch in enumerate(CHANNEL_NAMES)
    }
    return counts


def get_fixed_lf_indices(k: int) -> np.ndarray:
    """
    Get deterministic 1D array of flat coefficient indices [0..2047] corresponding to
    the lowest-frequency coefficients assigned to each channel.

    For channel c with count K_c, the retained coefficient indices within that channel are 0..K_c-1.
    Flat index = c * 512 + index_in_channel.

    Parameters:
    -----------
    k : int
        Total number of DCT coefficients.

    Returns:
    --------
    fixed_indices : np.ndarray
        1D uint16 array of shape (K,) containing flat indices.
    """
    counts = get_fixed_lf_channel_allocation(k)
    indices_list = []
    num_samples = 512

    for ch_idx, ch in enumerate(CHANNEL_NAMES):
        k_c = counts[ch]
        if k_c > 0:
            ch_indices = ch_idx * num_samples + np.arange(k_c, dtype=np.uint16)
            indices_list.append(ch_indices)

    if not indices_list:
        return np.array([], dtype=np.uint16)

    return np.concatenate(indices_list).astype(np.uint16)


def dct_encode_fixed_lf(
    window: np.ndarray,
    k: int
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """
    Perform DCT-II on 4-channel window signal and extract fixed low-frequency coefficients per channel.

    Parameters:
    -----------
    window : np.ndarray
        Signal window matrix of shape (4, 512) or (B, 4, 512).
    k : int
        Total number of lowest-frequency coefficients to retain (1 <= k <= 2048).

    Returns:
    --------
    fixed_values : np.ndarray
        Array of shape (K,) or (B, K) containing retained float32 low-frequency values.
    channel_counts : Dict[str, int]
        Number of retained coefficients per channel.
    sparse_dct : np.ndarray
        Sparse DCT array of shape (4, 512) or (B, 4, 512) with unselected coefficients zeroed out.
    """
    if fft is None:
        raise ImportError("scipy is required for DCT operations. Please install requirements.txt.")

    arr = np.asarray(window, dtype=np.float64)
    is_batch = False

    if arr.ndim == 2:
        if arr.shape != (4, 512):
            if arr.shape == (512, 4):
                arr = arr.T
            else:
                raise ValueError(f"Expected window shape (4, 512), got {arr.shape}")
    elif arr.ndim == 3:
        is_batch = True
        if arr.shape[1:] != (4, 512):
            raise ValueError(f"Expected batch shape (B, 4, 512), got {arr.shape}")
    else:
        raise ValueError(f"Input must be 2D (4, 512) or 3D (B, 4, 512), got shape {arr.shape}")

    total_coeffs = 4 * 512
    if not (1 <= k <= total_coeffs):
        raise ValueError(f"k must be between 1 and {total_coeffs}, got {k}")

    channel_counts = get_fixed_lf_channel_allocation(k)
    fixed_indices = get_fixed_lf_indices(k)

    # 1. Apply DCT-II independently on time axis for each channel
    dct_coeffs = fft.dct(arr, type=2, norm="ortho", axis=-1)

    if not is_batch:
        flat_coeffs = dct_coeffs.ravel()
        fixed_values = flat_coeffs[fixed_indices]

        sparse_dct = np.zeros_like(dct_coeffs)
        np.put(sparse_dct, fixed_indices, fixed_values)

        return fixed_values.astype(np.float32), channel_counts, sparse_dct.astype(np.float32)
    else:
        batch_size = arr.shape[0]
        fixed_values_list = []
        sparse_dct_list = []

        for b in range(batch_size):
            flat = dct_coeffs[b].ravel()
            vals = flat[fixed_indices]
            sp_dct = np.zeros_like(dct_coeffs[b])
            np.put(sp_dct, fixed_indices, vals)

            fixed_values_list.append(vals)
            sparse_dct_list.append(sp_dct)

        return (
            np.array(fixed_values_list, dtype=np.float32),
            channel_counts,
            np.array(sparse_dct_list, dtype=np.float32),
        )


def dct_decode_fixed_lf(sparse_dct: np.ndarray) -> np.ndarray:
    """
    Perform IDCT-II on sparse DCT-Fixed-LF matrix to reconstruct time-domain signal.
    """
    return dct_decode(sparse_dct)


def dct_reconstruct_fixed_lf(
    window: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Full DCT-Fixed-LF compression and reconstruction pipeline.

    Parameters:
    -----------
    window : np.ndarray
        Input signal window matrix of shape (4, 512).
    k : int
        Total low-frequency DCT coefficients to retain.

    Returns:
    --------
    reconstructed_signal : np.ndarray
        Reconstructed signal of shape (4, 512).
    fixed_values : np.ndarray
        Fixed low-frequency coefficient values of shape (K,).
    channel_counts : Dict[str, int]
        Count of retained coefficients per channel.
    """
    fixed_values, channel_counts, sparse_dct = dct_encode_fixed_lf(window, k)
    reconstructed_signal = dct_decode_fixed_lf(sparse_dct)
    return reconstructed_signal, fixed_values, channel_counts


class BaselineDCTFixedLF:
    """Class wrapper for DCT Fixed Low-Frequency baseline model."""
    def __init__(self, k: int = 128):
        self.k = k

    def encode(self, window: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
        return dct_encode_fixed_lf(window, self.k)

    def decode(self, sparse_dct: np.ndarray) -> np.ndarray:
        return dct_decode_fixed_lf(sparse_dct)

    def reconstruct(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        return dct_reconstruct_fixed_lf(window, self.k)

