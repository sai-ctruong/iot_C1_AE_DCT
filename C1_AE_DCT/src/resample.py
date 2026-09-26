"""Resampling and alignment module for PPG and ACC sensor signals in C1_AE_DCT."""

import math
from typing import Tuple, Optional, Union, Dict, Any

try:
    import numpy as np
    import scipy.signal as signal
except ImportError:
    np = None
    signal = None


def resample_acc(
    acc: Any,
    fs_in: float = 32.0,
    fs_out: float = 64.0,
    timestamps: Optional[Any] = None,
    max_gap_seconds: float = 1.0,
) -> Tuple[Any, Optional[Any]]:
    """
    Resample Accelerometer (ACC) signal from fs_in (32 Hz) to fs_out (64 Hz) using scipy.signal.resample_poly.

    Parameters:
    -----------
    acc : np.ndarray
        Array of shape (N, 3) or (N,) representing ACC channels (ACCx, ACCy, ACCz).
    fs_in : float
        Input sampling rate in Hz (default: 32.0 Hz).
    fs_out : float
        Target sampling rate in Hz (default: 64.0 Hz).
    timestamps : Optional[np.ndarray]
        Timestamps corresponding to input acc samples in seconds.
    max_gap_seconds : float
        Threshold duration in seconds to detect data gaps.

    Returns:
    --------
    acc_resampled : np.ndarray
        Resampled ACC data array with shape (N_resampled, 3) or (N_resampled,).
    timestamps_resampled : Optional[np.ndarray]
        Resampled timestamps if timestamps input was provided.
    """
    if signal is None or np is None:
        raise ImportError("numpy and scipy are required for resample_acc. Please install requirements.txt.")

    acc = np.asarray(acc, dtype=np.float64)

    # Compute up/down factors (32 -> 64 Hz means up=2, down=1)
    gcd_val = math.gcd(int(fs_in), int(fs_out))
    up = int(fs_out) // gcd_val
    down = int(fs_in) // gcd_val

    # Verify duration before resampling
    duration_before = len(acc) / fs_in

    # Check for data gaps if timestamps are provided
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        time_diffs = np.diff(timestamps)
        gap_indices = np.where(time_diffs > max_gap_seconds)[0]

        if len(gap_indices) > 0:
            # Split continuous segments across missing gaps without interpolating long gaps
            split_indices = gap_indices + 1
            acc_segments = np.split(acc, split_indices, axis=0)
            time_segments = np.split(timestamps, split_indices)

            resampled_acc_list = []
            resampled_time_list = []

            for acc_seg, time_seg in zip(acc_segments, time_segments):
                if len(acc_seg) == 0:
                    continue
                # Resample each continuous segment independently
                acc_res = signal.resample_poly(
                    acc_seg,
                    up=up,
                    down=down,
                    axis=0,
                    window=("kaiser", 5.0),
                    padtype="line",
                )
                time_res = np.linspace(
                    time_seg[0],
                    time_seg[-1],
                    num=len(acc_res),
                    endpoint=True,
                )
                resampled_acc_list.append(acc_res)
                resampled_time_list.append(time_res)

            acc_resampled = np.vstack(resampled_acc_list) if acc.ndim > 1 else np.concatenate(resampled_acc_list)
            timestamps_resampled = np.concatenate(resampled_time_list)
        else:
            # Continuous signal without long gaps
            acc_resampled = signal.resample_poly(
                acc,
                up=up,
                down=down,
                axis=0,
                window=("kaiser", 5.0),
                padtype="line",
            )
            timestamps_resampled = np.linspace(
                timestamps[0],
                timestamps[-1],
                num=len(acc_resampled),
                endpoint=True,
            )
    else:
        # Standard continuous resampling
        acc_resampled = signal.resample_poly(
            acc,
            up=up,
            down=down,
            axis=0,
            window=("kaiser", 5.0),
            padtype="line",
        )
        timestamps_resampled = None

    # Check duration after resampling
    duration_after = len(acc_resampled) / fs_out
    
    # Assert duration consistency within 1 sample tolerance
    assert abs(duration_before - duration_after) <= (1.0 / fs_in), (
        f"Duration mismatch! Before: {duration_before:.4f}s, After: {duration_after:.4f}s"
    )

    return acc_resampled, timestamps_resampled


def align_ppg_acc(
    ppg: Any,
    acc_resampled: Any,
    fs: float = 64.0,
    ppg_timestamps: Optional[Any] = None,
    acc_timestamps: Optional[Any] = None,
) -> Any:
    """
    Align PPG signal (64 Hz) and resampled ACC signal (64 Hz) into synchronized 4-channel matrix.

    Output Channel Ordering: [PPG, ACCx, ACCy, ACCz]

    Parameters:
    -----------
    ppg : np.ndarray
        PPG sensor signal array of shape (N,) or (N, 1) at 64 Hz.
    acc_resampled : np.ndarray
        Resampled ACC sensor signal array of shape (M, 3) at 64 Hz.
    fs : float
        Common sampling rate in Hz (default: 64.0 Hz).
    ppg_timestamps : Optional[np.ndarray]
        Timestamps for PPG samples.
    acc_timestamps : Optional[np.ndarray]
        Timestamps for resampled ACC samples.

    Returns:
    --------
    output_4ch : np.ndarray
        Aligned 4-channel array of shape (min_length, 4) containing [PPG, ACCx, ACCy, ACCz].
    """
    if np is None:
        raise ImportError("numpy is required for align_ppg_acc. Please install requirements.txt.")

    ppg = np.asarray(ppg, dtype=np.float64)
    acc_resampled = np.asarray(acc_resampled, dtype=np.float64)

    # Ensure 1D PPG
    if ppg.ndim == 2 and ppg.shape[1] == 1:
        ppg = ppg.squeeze(axis=1)
    elif ppg.ndim != 1:
        raise ValueError(f"PPG input must be 1D array or (N, 1), got shape {ppg.shape}")

    # Ensure 2D ACC (N, 3)
    if acc_resampled.ndim == 1:
        raise ValueError(f"ACC resampled must be 3-channel array (N, 3), got shape {acc_resampled.shape}")
    elif acc_resampled.ndim == 2 and acc_resampled.shape[1] != 3:
        if acc_resampled.shape[0] == 3:
            acc_resampled = acc_resampled.T  # Transpose to (N, 3)

    # Handle timestamp-based window alignment if timestamps provided
    if ppg_timestamps is not None and acc_timestamps is not None:
        t_start = max(ppg_timestamps[0], acc_timestamps[0])
        t_end = min(ppg_timestamps[-1], acc_timestamps[-1])

        ppg_mask = (ppg_timestamps >= t_start) & (ppg_timestamps <= t_end)
        acc_mask = (acc_timestamps >= t_start) & (acc_timestamps <= t_end)

        ppg = ppg[ppg_mask]
        acc_resampled = acc_resampled[acc_mask]

    # Align length to common minimum length
    min_len = min(len(ppg), len(acc_resampled))
    ppg_aligned = ppg[:min_len]
    acc_aligned = acc_resampled[:min_len]

    # Combine into 4 channels: [PPG, ACCx, ACCy, ACCz]
    output_4ch = np.column_stack([
        ppg_aligned,
        acc_aligned[:, 0],
        acc_aligned[:, 1],
        acc_aligned[:, 2],
    ])

    return output_4ch
