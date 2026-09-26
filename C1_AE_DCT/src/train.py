"""Training module for 1D-CNN Autoencoder in C1_AE_DCT project."""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    np = None
    torch = None

from src.model import C1Autoencoder
from src.loss import reconstruction_mse
from src.utils import load_folds, validate_folds


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, PyTorch (CPU & GPU) for exact reproducibility."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def evaluate_fair_val_loss(
    model: Any,
    val_windows_by_subject: Dict[str, Any],
    device: Any
) -> Dict[str, Any]:
    """
    Compute subject-fair validation MSE loss.

    Prevents subjects with larger number of windows from dominating validation score:
    Val_MSE_Fair = (1 / N_subjects) * sum(MSE_subject_i)
    """
    model.eval()
    subject_mses = {}

    with torch.no_grad():
        for subj, windows in val_windows_by_subject.items():
            if len(windows) == 0:
                continue
            windows_tensor = torch.as_tensor(windows, dtype=torch.float32, device=device)
            rec_x = model(windows_tensor)
            mse_val = float(torch.mean((rec_x - windows_tensor) ** 2).item())
            subject_mses[subj] = mse_val

    if len(subject_mses) == 0:
        val_fair_loss = 0.0
    else:
        val_fair_loss = float(sum(subject_mses.values()) / len(subject_mses))

    return {"val_fair_loss": val_fair_loss, "subject_mses": subject_mses}


def train_model(
    fold: int = 1,
    d_b: int = 16,
    train_windows: Optional[Any] = None,
    val_windows_by_subject: Optional[Dict[str, Any]] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    max_epoch: int = 100,
    early_stopping_patience: int = 10,
    seed: int = 42,
    checkpoints_dir: str = "checkpoints",
    logs_dir: str = "logs",
) -> Dict[str, Any]:
    """
    Train 1D-CNN Autoencoder model for a specific fold and budget factor d_b.

    Parameters:
    -----------
    fold : int
        Fold index (1..5).
    d_b : int
        Budget factor (16, 8, 4, 2).
    train_windows : Optional[np.ndarray / torch.Tensor]
        Training windows array of shape (N_train, 4, 512).
    val_windows_by_subject : Optional[Dict[str, np.ndarray]]
        Dictionary mapping val subject ID (e.g., 'S4', 'S5') to windows (N_subj, 4, 512).
    lr : float
        Learning rate (default: 1e-3).
    weight_decay : float
        Weight decay (default: 0.0).
    batch_size : int
        Batch size (default: 128).
    max_epoch : int
        Maximum training epochs (default: 100).
    early_stopping_patience : int
        Patience epochs for early stopping (default: 10).
    seed : int
        Random seed (default: 42).
    checkpoints_dir : str
        Directory to save best checkpoint weights.
    logs_dir : str
        Directory to save training logs.

    Returns:
    --------
    results : Dict[str, Any]
        Dictionary containing best_val_loss, best_epoch, checkpoint_path, and history log.
    """
    if torch is None or np is None:
        raise ImportError("PyTorch and NumPy are required for training. Please install requirements.txt.")

    # 1. Set Seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[C1_AE_DCT] Starting Training | Fold: {fold} | d_b: {d_b} | Device: {device} | Seed: {seed}")

    # Prepare directories
    ckpt_path_dir = Path(checkpoints_dir)
    ckpt_path_dir.mkdir(parents=True, exist_ok=True)
    
    log_path_dir = Path(logs_dir)
    log_path_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_path_dir / f"best_model_fold{fold}_db{d_b}.pt"
    log_file = log_path_dir / f"train_log_fold{fold}_db{d_b}.json"

    # Handle synthetic/dummy data fallback for dry-run testing if no windows provided
    if train_windows is None:
        print("[WARNING] No train_windows provided. Generating synthetic training data for dry-run verification.")
        train_windows = np.random.randn(256, 4, 512).astype(np.float32)

    if val_windows_by_subject is None:
        folds_info = load_folds()
        val_subjs = folds_info[f"fold_{fold}"]["val"]
        val_windows_by_subject = {
            subj: np.random.randn(32, 4, 512).astype(np.float32)
            for subj in val_subjs
        }

    # Prepare Train DataLoader (SHUFFLE ONLY TRAIN)
    train_tensor = torch.as_tensor(train_windows, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle ONLY Train
        drop_last=False
    )

    # Instantiate Model & Optimizer
    model = C1Autoencoder(d_b=d_b).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, max_epoch + 1):
        start_time = time.time()

        # Training Phase
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            rec_x = model(batch_x)
            loss = reconstruction_mse(rec_x, batch_x)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            num_batches += 1

        avg_train_loss = train_loss_sum / max(1, num_batches)

        # Validation Phase (Subject-Fair Aggregation)
        val_res = evaluate_fair_val_loss(model, val_windows_by_subject, device)
        avg_val_loss = val_res["val_fair_loss"]

        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr,
            "elapsed_time": elapsed_time,
            "subject_val_mses": val_res["subject_mses"],
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:03d}/{max_epoch:03d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss (Fair): {avg_val_loss:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed_time:.2f}s"
        )

        # Early Stopping & Checkpoint Selection on Validation MSE
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "d_b": d_b,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                ckpt_file,
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[C1_AE_DCT] Early stopping triggered at epoch {epoch}. Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.6f})")
                break

    # Save training history to JSON
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}, f, indent=2)

    return {
        "fold": fold,
        "d_b": d_b,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_file),
        "log_path": str(log_file),
    }


if __name__ == "__main__":
    train_model(fold=1, d_b=16, max_epoch=3, batch_size=32)
