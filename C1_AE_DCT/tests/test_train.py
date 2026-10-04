"""Unit test for TASK 14 — Training Loop & Validation Evaluation."""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_set_seed_reproducibility():
    """Verify set_seed sets Python, NumPy, and PyTorch seeds deterministically."""
    import torch
    from src.train import set_seed

    set_seed(42)
    val1 = torch.randn(5)
    np_val1 = np.random.randn(5)

    set_seed(42)
    val2 = torch.randn(5)
    np_val2 = np.random.randn(5)

    torch.testing.assert_close(val1, val2)
    np.testing.assert_array_equal(np_val1, np_val2)
    print("[TEST] Seed reproducibility verified!")


def test_fair_val_loss_aggregation():
    """Verify fair validation MSE aggregation averages subject MSEs without window count bias."""
    import torch

    from src.model import C1Autoencoder
    from src.train import evaluate_fair_val_loss

    device = torch.device("cpu")
    model = C1Autoencoder(d_b=16)

    # Subject 4 has 10 windows, Subject 5 has 100 windows
    val_windows_by_subject = {
        "S4": np.random.randn(10, 4, 512).astype(np.float32),
        "S5": np.random.randn(100, 4, 512).astype(np.float32),
    }

    res = evaluate_fair_val_loss(model, val_windows_by_subject, device)

    val_fair = res["val_fair_loss"]
    sub_mses = res["subject_mses"]

    expected_fair = (sub_mses["S4"] + sub_mses["S5"]) / 2.0
    assert abs(val_fair - expected_fair) < 1e-6, f"Fair loss calculation mismatch: {val_fair} != {expected_fair}"
    print(f"[TEST] Fair validation loss (S4: {sub_mses['S4']:.4f}, S5: {sub_mses['S5']:.4f}) -> Fair: {val_fair:.4f} passed!")


def test_training_loop_execution():
    """Verify end-to-end training loop execution, checkpoint saving, and log writing."""
    import torch
    from src.train import train_model

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"

    # Create test fixture windows for test suite dry run
    train_windows = np.random.randn(32, 4, 512).astype(np.float32)
    val_windows_by_subject = {
        "S4": np.random.randn(8, 4, 512).astype(np.float32),
        "S5": np.random.randn(8, 4, 512).astype(np.float32),
    }

    # Run 3 epochs dry run for Fold 1, d_b = 16
    results = train_model(
        fold=1,
        d_b=16,
        train_windows=train_windows,
        val_windows_by_subject=val_windows_by_subject,
        max_epoch=3,
        batch_size=16,
        early_stopping_patience=2,
        seed=42,
        checkpoints_dir=str(ckpt_dir),
        logs_dir=str(log_dir)
    )

    assert results["fold"] == 1
    assert results["d_b"] == 16
    assert results["best_epoch"] >= 1
    assert os.path.exists(results["checkpoint_path"]), "Best checkpoint file was not saved!"
    assert os.path.exists(results["log_path"]), "Train log file was not saved!"

    # Verify loaded checkpoint
    ckpt = torch.load(results["checkpoint_path"], map_location="cpu")
    assert "model_state_dict" in ckpt
    assert ckpt["d_b"] == 16
    assert ckpt["fold"] == 1

    print("[SUCCESS] End-to-end training loop verification passed cleanly!")


if __name__ == "__main__":
    print("=== TASK 14 — TRAINING LOOP VERIFICATION ===")
    test_set_seed_reproducibility()
    test_fair_val_loss_aggregation()
    test_training_loop_execution()
    print("=== ALL TRAINING LOOP TESTS PASSED CLEANLY ===")
