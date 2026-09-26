"""Z-Score Normalization module for C1_AE_DCT project (Fold-Wise Train-Only Statistics)."""

import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None

CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]

def compute_norm_stats(
    train_data: Any,
    channel_names: Optional[List[str]] = None,
    ddof: int = 0
) -> Dict[str, Any]:
    """
    Compute channel-wise mean and standard deviation from training data only.

    Parameters:
    -----------
    train_data : np.ndarray
        Continuous 2D array of shape (N, C) representing training signal samples.
        Columns correspond to channels [PPG, ACCx, ACCy, ACCz].
    channel_names : Optional[List[str]]
        List of channel names. Defaults to ["PPG", "ACCx", "ACCy", "ACCz"].
    ddof : int
        Delta Degrees of Freedom for std calculation (default: 0).

    Returns:
    --------
    stats : Dict[str, Any]
        Dictionary containing channel names, means, stds, and ddof.
    """
    if np is None:
        raise ImportError("numpy is required for compute_norm_stats. Run 'pip install numpy'.")

    data = np.asarray(train_data, dtype=np.float32)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim > 2:
        # Reshape (B, T, C) or (B, C, T) to (-1, C) if windowed data is passed
        data = data.reshape(-1, data.shape[-1])

    num_channels = data.shape[1]
    if channel_names is None:
        channel_names = CHANNEL_NAMES[:num_channels]

    means = np.mean(data, axis=0, dtype=np.float64)
    stds = np.std(data, axis=0, ddof=ddof, dtype=np.float64)

    # Check for zero or non-positive standard deviation
    for idx, (ch_name, sigma) in enumerate(zip(channel_names, stds)):
        if sigma <= 0 or np.isnan(sigma):
            raise ValueError(
                f"Invalid standard deviation for channel '{ch_name}' (channel {idx}): std = {sigma}. "
                "Cannot normalize constant or invalid signal."
            )

    stats = {
        "channels": channel_names,
        "mean": [float(m) for m in means],
        "std": [float(s) for s in stds],
        "ddof": ddof
    }

    return stats


def normalize(
    data: Any,
    stats: Dict[str, Any]
) -> Any:
    """
    Normalize signal data using pre-computed Z-score statistics (x_norm = (x - mu) / sigma).

    Parameters:
    -----------
    data : np.ndarray
        Signal array of shape (N, C), (B, T, C), or (B, C, T).
    stats : Dict[str, Any]
        Dictionary containing "mean" and "std" arrays/lists.

    Returns:
    --------
    data_norm : np.ndarray
        Z-score normalized signal array of same shape and float32 dtype.
    """
    if np is None:
        raise ImportError("numpy is required for normalize. Run 'pip install numpy'.")

    arr = np.asarray(data, dtype=np.float32)
    means = np.array(stats["mean"], dtype=np.float32)
    stds = np.array(stats["std"], dtype=np.float32)

    # Check dimension matching
    if arr.ndim == 2:
        if arr.shape[1] == len(means):
            # Shape: (N, C)
            data_norm = (arr - means) / stds
        elif arr.shape[0] == len(means):
            # Shape: (C, N)
            data_norm = (arr - means[:, np.newaxis]) / stds[:, np.newaxis]
        else:
            raise ValueError(f"Array shape {arr.shape} does not match channel count {len(means)}")
    elif arr.ndim == 3:
        if arr.shape[-1] == len(means):
            # Shape: (B, T, C)
            data_norm = (arr - means) / stds
        elif arr.shape[1] == len(means):
            # Shape: (B, C, T)
            data_norm = (arr - means[:, np.newaxis]) / stds[:, np.newaxis]
        else:
            raise ValueError(f"Array shape {arr.shape} does not match number of channels {len(means)}")
    elif arr.ndim == 1 and len(means) == 1:
        data_norm = (arr - means[0]) / stds[0]
    else:
        raise ValueError(f"Unsupported data shape {arr.shape} for channel count {len(means)}")

    return data_norm.astype(np.float32)


def denormalize(
    data_norm: Any,
    stats: Dict[str, Any]
) -> Any:
    """
    Denormalize Z-score normalized signal data back to original physical scale (x = x_norm * sigma + mu).

    Parameters:
    -----------
    data_norm : np.ndarray
        Z-score normalized signal array of shape (N, C), (C, N), (B, T, C), or (B, C, T).
    stats : Dict[str, Any]
        Dictionary containing "mean" and "std" arrays/lists.

    Returns:
    --------
    data_rec : np.ndarray
        Reconstructed signal array in original physical scale.
    """
    if np is None:
        raise ImportError("numpy is required for denormalize. Run 'pip install numpy'.")

    arr = np.asarray(data_norm, dtype=np.float32)
    means = np.array(stats["mean"], dtype=np.float32)
    stds = np.array(stats["std"], dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[1] == len(means):
            # Shape: (N, C)
            data_rec = arr * stds + means
        elif arr.shape[0] == len(means):
            # Shape: (C, N)
            data_rec = arr * stds[:, np.newaxis] + means[:, np.newaxis]
        else:
            raise ValueError(f"Array shape {arr.shape} does not match channel count {len(means)}")
    elif arr.ndim == 3:
        if arr.shape[-1] == len(means):
            # Shape: (B, T, C)
            data_rec = arr * stds + means
        elif arr.shape[1] == len(means):
            # Shape: (B, C, T)
            data_rec = arr * stds[:, np.newaxis] + means[:, np.newaxis]
        else:
            raise ValueError(f"Array shape {arr.shape} does not match number of channels {len(means)}")
    elif arr.ndim == 1 and len(means) == 1:
        data_rec = arr * stds[0] + means[0]
    else:
        raise ValueError(f"Unsupported data shape {arr.shape} for channel count {len(means)}")

    return data_rec.astype(np.float32)



def save_norm_stats(stats: Dict[str, Any], file_path: Union[str, Path]) -> Path:
    """Save norm_stats dictionary to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return path


def load_norm_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load norm_stats dictionary from JSON file."""
    path = Path(file_path)
    if not path.exists():
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / file_path

    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats
