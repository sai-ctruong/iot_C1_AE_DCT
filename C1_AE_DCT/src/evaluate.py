"""Evaluation module for computing per-channel PRD, PRDN, and RMSE distortion metrics in C1_AE_DCT (TASK 18)."""

import math
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from src.normalize import denormalize
except ImportError:
    try:
        from normalize import denormalize
    except ImportError:
        denormalize = None

CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]
DENOMINATOR_EPS_THRESHOLD = 1e-12


def compute_channel_metrics(
    x_ref: np.ndarray,
    x_pred: np.ndarray,
    channel_name: str = "PPG"
) -> Dict[str, Any]:
    """
    Compute RMSE, PRD, and PRDN for a single 1D physical signal channel of length N=512.

    IMPORTANT RULES (TASK 18):
    1. x_ref and x_pred MUST be in original physical units (denormalized).
    2. Computed independently per channel (PPG, ACCx, ACCy, ACCz).
    3. PRDN uses mean-centered reference energy in denominator: sum((x_ref - mean_x)^2),
       while numerator is full reconstruction error: sum((x_ref - x_pred)^2).
    4. Denominator <= 1e-12 sets valid_prd/valid_prdn to False and metric to NaN (no arbitrary epsilon added).

    Parameters:
    -----------
    x_ref : np.ndarray
        1D reference physical signal array (length 512).
    x_pred : np.ndarray
        1D reconstructed physical signal array (length 512).
    channel_name : str
        Name of channel ("PPG", "ACCx", "ACCy", "ACCz").

    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary containing channel, rmse, prd, prdn, valid_prd, valid_prdn.
    """
    x_ref = np.asarray(x_ref, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64)

    err = x_ref - x_pred
    sse = np.sum(err ** 2)  # Sum of squared errors (full reconstruction error)
    n = len(x_ref)

    # 1. RMSE (Root Mean Squared Error)
    rmse = math.sqrt(sse / n)

    # 2. PRD (Percentage Relative Distortion)
    # Denominator: total energy of physical signal
    ref_energy = np.sum(x_ref ** 2)
    if ref_energy > DENOMINATOR_EPS_THRESHOLD:
        prd = math.sqrt(sse / ref_energy) * 100.0
        valid_prd = True
    else:
        prd = float("nan")
        valid_prd = False

    # 3. PRDN (Normalized Percentage Relative Distortion)
    # Denominator: mean-centered reference energy sum((x_ref - mean_x)^2)
    # Numerator: full reconstruction error (sse)
    ref_mean = np.mean(x_ref)
    ref_centered_energy = np.sum((x_ref - ref_mean) ** 2)
    if ref_centered_energy > DENOMINATOR_EPS_THRESHOLD:
        prdn = math.sqrt(sse / ref_centered_energy) * 100.0
        valid_prdn = True
    else:
        prdn = float("nan")
        valid_prdn = False

    return {
        "channel": channel_name,
        "rmse": float(rmse),
        "prd": float(prd),
        "prdn": float(prdn),
        "valid_prd": valid_prd,
        "valid_prdn": valid_prdn,
    }


def evaluate_window_metrics(
    window_ref: np.ndarray,
    window_pred: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    norm_stats: Optional[Dict[str, Any]] = None,
    channel_names: List[str] = CHANNEL_NAMES
) -> List[Dict[str, Any]]:
    """
    Compute per-channel distortion metrics for a single 4x512 signal window.

    If norm_stats is provided, signals are automatically denormalized first.

    Parameters:
    -----------
    window_ref : np.ndarray
        Shape (4, 512) reference signal.
    window_pred : np.ndarray
        Shape (4, 512) reconstructed signal.
    meta : Optional[Dict[str, Any]]
        Window metadata dict (window_id, subject, fold, split).
    norm_stats : Optional[Dict[str, Any]]
        Normalization statistics dict for denormalizing signals back to physical scale.
    channel_names : List[str]
        List of 4 channel names.

    Returns:
    --------
    results : List[Dict[str, Any]]
        List of 4 dictionaries (one per channel).
    """
    ref = np.asarray(window_ref, dtype=np.float64)
    pred = np.asarray(window_pred, dtype=np.float64)

    if norm_stats is not None:
        if denormalize is None:
            raise ImportError("denormalize function is not available.")
        ref = denormalize(ref, norm_stats).astype(np.float64)
        pred = denormalize(pred, norm_stats).astype(np.float64)

    if ref.shape != (4, 512):
        if ref.shape == (512, 4):
            ref = ref.T
        else:
            raise ValueError(f"Expected reference window shape (4, 512), got {ref.shape}")

    if pred.shape != (4, 512):
        if pred.shape == (512, 4):
            pred = pred.T
        else:
            raise ValueError(f"Expected prediction window shape (4, 512), got {pred.shape}")

    meta = meta or {}
    window_results = []

    for c_idx, ch_name in enumerate(channel_names):
        ch_ref = ref[c_idx]
        ch_pred = pred[c_idx]

        ch_metrics = compute_channel_metrics(ch_ref, ch_pred, channel_name=ch_name)

        row = {
            "window_id": meta.get("window_id", "win_unknown"),
            "subject": meta.get("subject", "sub_unknown"),
            "fold": meta.get("fold", 0),
            "split": meta.get("split", "test"),
            "channel": ch_name,
            "rmse": ch_metrics["rmse"],
            "prd": ch_metrics["prd"],
            "prdn": ch_metrics["prdn"],
            "valid_prd": ch_metrics["valid_prd"],
            "valid_prdn": ch_metrics["valid_prdn"],
        }
        window_results.append(row)

    return window_results


def evaluate_dataset_metrics(
    windows_ref: np.ndarray,
    windows_pred: np.ndarray,
    norm_stats: Optional[Dict[str, Any]] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    channel_names: List[str] = CHANNEL_NAMES,
) -> Union[Any, List[Dict[str, Any]]]:
    """
    Compute per-channel metrics across all windows in a dataset split.

    If norm_stats is provided, denormalization is applied to all windows prior to metric computation.

    Returns Pandas DataFrame (if pandas installed) or List of Dicts.
    """
    refs = np.asarray(windows_ref, dtype=np.float64)
    preds = np.asarray(windows_pred, dtype=np.float64)

    if norm_stats is not None:
        if denormalize is None:
            raise ImportError("denormalize function is not available.")
        refs = denormalize(refs, norm_stats).astype(np.float64)
        preds = denormalize(preds, norm_stats).astype(np.float64)

    num_windows = len(refs)
    if metadata_list is None:
        metadata_list = [{"window_id": f"win_{i:04d}"} for i in range(num_windows)]

    all_rows = []
    for i in range(num_windows):
        meta = metadata_list[i] if i < len(metadata_list) else {}
        w_rows = evaluate_window_metrics(refs[i], preds[i], meta=meta, channel_names=channel_names)
        all_rows.extend(w_rows)

    if pd is not None:
        return pd.DataFrame(all_rows)
    return all_rows


def compute_joint_valid_mask(
    df_ae: Any,
    df_dct: Any,
    metric_col: str = "valid_prdn"
) -> Any:
    """
    Compute shared valid mask across AE and DCT results to ensure strictly equal sample evaluation.

    Parameters:
    -----------
    df_ae : pd.DataFrame
        Detailed metrics dataframe for AE model.
    df_dct : pd.DataFrame
        Detailed metrics dataframe for DCT baseline.
    metric_col : str
        Validity boolean column ('valid_prd' or 'valid_prdn').

    Returns:
    --------
    joint_mask : pd.Series / np.ndarray
        Boolean mask True where BOTH AE and DCT are valid for the window x channel pair.
    """
    if pd is not None and isinstance(df_ae, pd.DataFrame) and isinstance(df_dct, pd.DataFrame):
        return df_ae[metric_col] & df_dct[metric_col]
    elif isinstance(df_ae, list) and isinstance(df_dct, list):
        return [row_ae[metric_col] and row_dct[metric_col] for row_ae, row_dct in zip(df_ae, df_dct)]
    else:
        raise ValueError("df_ae and df_dct must both be DataFrames or Lists of Dicts.")


def evaluate_model(model=None, test_loader=None) -> Dict[str, float]:
    """Placeholder evaluation model function for project stub integration."""
    print("[C1_AE_DCT] Evaluation module initialized.")
    return {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}

