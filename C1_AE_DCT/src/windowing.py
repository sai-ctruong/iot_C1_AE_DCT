"""Windowing module for C1_AE_DCT project (8s window, 4s/8s step, [4, 512] output shape)."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np

def create_windows(
    signal_data: np.ndarray,
    fs: float = 64.0,
    window_sec: float = 8.0,
    step_sec: Optional[float] = None,
    subject: Optional[str] = None,
    fold: Optional[int] = None,
    split: str = "train",
    timestamps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate fixed 8-second sliding windows from 4-channel signal data [PPG, ACCx, ACCy, ACCz].

    Window parameters:
    - Target sampling rate: 64 Hz
    - Window length: 8 seconds (512 samples)
    - Train / Val step: 4 seconds (256 samples, 50% overlap)
    - Test step: 8 seconds (512 samples, 0% overlap)
    - Output window shape: [4, 512] (Channels x Time)

    Parameters:
    -----------
    signal_data : np.ndarray
        Signal matrix of shape (N, 4) or (4, N).
    fs : float
        Sampling rate in Hz (default: 64.0 Hz).
    window_sec : float
        Window duration in seconds (default: 8.0 s -> 512 samples).
    step_sec : Optional[float]
        Step duration in seconds.
        If None:
          - Train / Val: step_sec = 4.0 s (256 samples, 50% overlap)
          - Test: step_sec = 8.0 s (512 samples, no overlap)
    subject : Optional[str]
        Subject identifier (e.g., "S1").
    fold : Optional[int]
        Fold index (1..5).
    split : str
        Split type ("train", "val", "test").
    timestamps : Optional[np.ndarray]
        Timestamps corresponding to signal_data samples.

    Returns:
    --------
    windows : np.ndarray
        Array of shape (num_windows, 4, 512) and float32 dtype.
    metadata_list : List[Dict[str, Any]]
        List of metadata dictionaries for each generated window.
    summary_stats : Dict[str, Any]
        Summary containing window count before/after filtering and leftover samples dropped.
    """
    arr = np.asarray(signal_data, dtype=np.float32)

    # Format shape to (N, 4)
    if arr.ndim == 2:
        if arr.shape[0] == 4 and arr.shape[1] != 4:
            arr = arr.T
        elif arr.shape[1] != 4:
            raise ValueError(f"Expected 4 channels [PPG, ACCx, ACCy, ACCz], got shape {arr.shape}")
    else:
        raise ValueError(f"Signal data must be 2D matrix (N, 4) or (4, N), got shape {arr.shape}")

    num_samples, num_channels = arr.shape
    win_size = int(round(window_sec * fs))  # 8 * 64 = 512

    # Determine step size based on split type if step_sec not explicitly provided
    if step_sec is None:
        if split.lower() in ["train", "val"]:
            step_sec = 4.0  # 50% overlap
        elif split.lower() == "test":
            step_sec = 8.0  # No overlap
        else:
            step_sec = 4.0

    step_size = int(round(step_sec * fs))  # 4 * 64 = 256 or 8 * 64 = 512

    windows_list = []
    metadata_list = []

    start_idx = 0
    window_counter = 0

    while start_idx + win_size <= num_samples:
        end_idx = start_idx + win_size
        
        # Extract window of shape (512, 4) and transpose to (4, 512)
        win = arr[start_idx:end_idx, :].T  # Shape: (4, 512)

        # Timestamps metadata calculation
        if timestamps is not None:
            start_time = float(timestamps[start_idx])
            end_time = float(timestamps[end_idx - 1])
        else:
            start_time = float(start_idx / fs)
            end_time = float((end_idx - 1) / fs)

        subj_str = subject if subject is not None else "Unknown"
        fold_int = fold if fold is not None else 0

        win_id = f"{subj_str}_{split}_f{fold_int}_w{window_counter:04d}"

        meta = {
            "window_id": win_id,
            "subject": subj_str,
            "fold": fold_int,
            "split": split,
            "start_index": start_idx,
            "end_index": end_idx,
            "start_time": start_time,
            "end_time": end_time,
            "num_samples": win_size,
            "fs": fs,
        }

        windows_list.append(win)
        metadata_list.append(meta)

        window_counter += 1
        start_idx += step_size

    # Calculate leftover samples dropped (incomplete trailing window)
    if len(windows_list) > 0:
        last_covered_end = (len(windows_list) - 1) * step_size + win_size
        leftover_samples = max(0, num_samples - last_covered_end)
    else:
        leftover_samples = num_samples

    summary_stats = {
        "subject": subject,
        "split": split,
        "total_input_samples": num_samples,
        "window_size_samples": win_size,
        "step_size_samples": step_size,
        "windows_generated": len(windows_list),
        "leftover_samples_dropped": leftover_samples,
        "leftover_seconds_dropped": float(leftover_samples / fs),
    }

    if len(windows_list) > 0:
        windows_array = np.stack(windows_list, axis=0).astype(np.float32)  # Shape: (N_win, 4, 512)
    else:
        windows_array = np.empty((0, num_channels, win_size), dtype=np.float32)

    return windows_array, metadata_list, summary_stats
