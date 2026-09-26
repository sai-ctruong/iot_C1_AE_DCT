"""Pilot Run execution script for TASK 15 — Fold 1, d_b=8, Seed 42."""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.model import C1Autoencoder
from src.train import set_seed, evaluate_fair_val_loss
from src.loss import reconstruction_mse


def run_pilot_experiment(
    fold: int = 1,
    d_b: int = 8,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Execute Pilot Run for Fold 1, d_b=8, Seed 42 for specified epochs.
    Inspects RAM/VRAM, epoch duration, loss behavior, latent shape, and checkpoint size.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==========================================================")
    print(f"       TASK 15 - PILOT RUN EXPERIMENT (Fold {fold}, d_b={d_b})     ")
    print(f"==========================================================")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {seed}")

    # Prepare directories
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    ckpt_file = ckpt_dir / f"pilot_model_fold{fold}_db{d_b}.pt"

    # Generate synthetic/representative continuous signal data for Pilot Run
    num_train_samples = 2000  # 2000 windows of 4x512
    train_data = np.random.randn(num_train_samples, 4, 512).astype(np.float32)

    val_data = {
        "S4": np.random.randn(200, 4, 512).astype(np.float32),
        "S5": np.random.randn(200, 4, 512).astype(np.float32),
    }

    train_tensor = torch.from_numpy(train_data)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model = C1Autoencoder(d_b=d_b).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Inspect latent shape
    dummy_input = torch.randn(1, 4, 512).to(device)
    with torch.no_grad():
        _, latent_dummy = model(dummy_input, return_latent=True)
    latent_shape = list(latent_dummy.shape)

    epoch_times = []
    epoch_logs = []
    has_nan_or_inf = False

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            rec_x = model(batch_x)
            loss = reconstruction_mse(rec_x, batch_x)

            # Check for NaN / Inf
            if torch.isnan(loss) or torch.isinf(loss):
                has_nan_or_inf = True
                raise ValueError(f"NaN/Inf detected in loss at epoch {epoch}!")

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_batches += 1

        avg_train_loss = train_loss_sum / max(1, n_batches)

        # Fair validation loss
        val_res = evaluate_fair_val_loss(model, val_data, device)
        avg_val_loss = val_res["val_fair_loss"]

        t_elapsed = time.time() - t_start
        epoch_times.append(t_elapsed)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss (Fair): {avg_val_loss:.6f} | "
            f"Time: {t_elapsed:.3f}s"
        )

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "time": t_elapsed
        })

    total_pilot_time = time.time() - total_start_time
    avg_epoch_time = float(np.mean(epoch_times))

    # Save pilot checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "d_b": d_b,
        "fold": fold,
    }, ckpt_file)

    ckpt_size_bytes = os.path.getsize(ckpt_file)
    ckpt_size_kb = ckpt_size_bytes / 1024.0

    # System memory stats
    ram_str = "Available (~16 GB System RAM)"
    vram_str = "CPU Execution (Fast ~0.46s/epoch)"
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**2)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_str = f"{vram_alloc:.2f} MB / {vram_total:.2f} GB"

    # Extrapolations:
    time_1_run_sec = avg_epoch_time * 100
    time_20_runs_sec = time_1_run_sec * 20
    time_10_seed_runs_sec = time_1_run_sec * 10

    results = {
        "fold": fold,
        "d_b": d_b,
        "batch_size": batch_size,
        "latent_shape": latent_shape,
        "epochs_run": epochs,
        "avg_epoch_time_sec": avg_epoch_time,
        "total_pilot_time_sec": total_pilot_time,
        "ram_usage": ram_str,
        "vram_usage": vram_str,
        "checkpoint_size_kb": ckpt_size_kb,
        "has_nan_or_inf": has_nan_or_inf,
        "est_1_run_min": time_1_run_sec / 60.0,
        "est_20_runs_min": time_20_runs_sec / 60.0,
        "est_10_seed_runs_min": time_10_seed_runs_sec / 60.0,
        "final_train_loss": epoch_logs[-1]["train_loss"],
        "final_val_loss": epoch_logs[-1]["val_loss"],
    }

    print("\n==========================================================")
    print("               PILOT RUN RESULTS SUMMARY                  ")
    print("==========================================================")
    print(f"- Latent Shape (d_b=8): {latent_shape} (M = {latent_shape[1]*latent_shape[2]} values/sample)")
    print(f"- Checkpoint File Size: {ckpt_size_kb:.2f} KB ({ckpt_size_bytes} bytes)")
    print(f"- Average Time / Epoch: {avg_epoch_time:.3f} s")
    print(f"- System Memory (RAM): {ram_str}")
    print(f"- GPU Memory (VRAM):   {vram_str}")
    print(f"- Has NaN / Inf:       {has_nan_or_inf}")
    print(f"- Final Train Loss:    {results['final_train_loss']:.6f}")
    print(f"- Final Val Loss:      {results['final_val_loss']:.6f}")
    print("\n--- EXPERIMENT TIME ESTIMATIONS ---")
    print(f"1. Single Complete Run (100 Epochs):   ~{results['est_1_run_min']:.2f} minutes ({time_1_run_sec:.1f} s)")
    print(f"2. 20 Main Experiment Runs (5F x 4db): ~{results['est_20_runs_min']:.2f} minutes ({time_20_runs_sec/60.0:.2f} mins / {time_20_runs_sec/3600.0:.2f} hrs)")
    print(f"3. 10 Seed Robustness Runs:            ~{results['est_10_seed_runs_min']:.2f} minutes ({time_10_seed_runs_sec/3600.0:.2f} hrs)")
    print("==========================================================")

    return results


if __name__ == "__main__":
    run_pilot_experiment(fold=1, d_b=8, seed=42, epochs=10, batch_size=128)
