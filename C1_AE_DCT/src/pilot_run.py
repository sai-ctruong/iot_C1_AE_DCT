"""Pilot Run execution script for real PPG-DaLiA dataset — Fold 1, d_b=8, Seed 42."""

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
from src.prepare_data import find_dataset_root, preprocess_all_subjects, prepare_folds_processed_data


def load_real_fold_data(fold: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load real preprocessed fold data, running preprocessing if missing."""
    proc_dir = PROJECT_ROOT / "data" / "processed" / f"fold{fold}"
    train_npz = proc_dir / "train.npz"
    val_npz = proc_dir / "val.npz"

    if not train_npz.exists() or not val_npz.exists():
        print(f"[PILOT] Processed data for fold {fold} not found. Running preprocessing pipeline...")
        root = find_dataset_root()
        subj_map = preprocess_all_subjects(dataset_dir=root)
        prepare_folds_processed_data(subj_map)

    train_data = np.load(train_npz, allow_pickle=True)["windows"].astype(np.float32)
    val_npz_data = np.load(val_npz, allow_pickle=True)
    val_windows = val_npz_data["windows"].astype(np.float32)
    val_meta = val_npz_data["metadata"]
    if val_meta.ndim == 0:
        val_meta_list = val_meta.item()
    else:
        val_meta_list = list(val_meta)

    # Group validation windows by subject for fair val loss computation
    val_data_by_subj = {}
    for win, meta in zip(val_windows, val_meta_list):
        subj = meta["subject"] if isinstance(meta, dict) and "subject" in meta else "val_unknown"
        if subj not in val_data_by_subj:
            val_data_by_subj[subj] = []
        val_data_by_subj[subj].append(win)

    for subj in val_data_by_subj:
        val_data_by_subj[subj] = np.stack(val_data_by_subj[subj], axis=0).astype(np.float32)

    return train_data, val_data_by_subj


def run_pilot_experiment(
    fold: int = 1,
    d_b: int = 8,
    seed: int = 42,
    epochs: int = 10,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Execute Pilot Run on real PPG-DaLiA dataset for Fold 1, d_b=8, Seed 42."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==========================================================")
    print(f"   REAL PPG-DaLiA PILOT EXPERIMENT (Fold {fold}, d_b={d_b}, Seed {seed}) ")
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

    ckpt_file = ckpt_dir / f"pilot_real_fold{fold:02d}_db{d_b:02d}_seed{seed}.pt"

    # Load REAL PPG-DaLiA dataset for Fold 1
    train_data, val_data = load_real_fold_data(fold=fold)
    print(f"[PILOT] Loaded REAL Train windows: {train_data.shape}")
    for s_name, s_arr in val_data.items():
        print(f"[PILOT] Loaded REAL Val {s_name} windows: {s_arr.shape}")

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

    # Save real pilot checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "d_b": d_b,
        "fold": fold,
        "seed": seed,
        "dataset": "PPG-DaLiA",
        "data_type": "REAL_DATA",
    }, ckpt_file)

    ckpt_size_bytes = os.path.getsize(ckpt_file)
    ckpt_size_kb = ckpt_size_bytes / 1024.0

    # System memory stats
    ram_str = "Available"
    vram_str = "CPU Execution"
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().used / (1024**3), 2)
        ram_total = round(psutil.virtual_memory().total / (1024**3), 2)
        ram_str = f"{ram_gb} GB / {ram_total} GB"
    except ImportError:
        pass

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
        "seed": seed,
        "data_type": "REAL_DATA",
        "dataset": "PPG-DaLiA",
        "batch_size": batch_size,
        "latent_shape": latent_shape,
        "epochs_run": epochs,
        "avg_epoch_time_sec": avg_epoch_time,
        "total_pilot_time_sec": total_pilot_time,
        "ram_usage": ram_str,
        "vram_usage": vram_str,
        "checkpoint_size_kb": ckpt_size_kb,
        "checkpoint_path": str(ckpt_file),
        "has_nan_or_inf": has_nan_or_inf,
        "est_1_run_min": time_1_run_sec / 60.0,
        "est_20_runs_min": time_20_runs_sec / 60.0,
        "est_10_seed_runs_min": time_10_seed_runs_sec / 60.0,
        "final_train_loss": epoch_logs[-1]["train_loss"],
        "final_val_loss": epoch_logs[-1]["val_loss"],
        "epoch_logs": epoch_logs,
    }

    print("\n==========================================================")
    print("            REAL PILOT RUN RESULTS SUMMARY                ")
    print("==========================================================")
    print(f"- Latent Shape (d_b={d_b}): {latent_shape} (M = {latent_shape[1]*latent_shape[2]} values/sample)")
    print(f"- Checkpoint Path:     {ckpt_file.name}")
    print(f"- Checkpoint Size:     {ckpt_size_kb:.2f} KB ({ckpt_size_bytes} bytes)")
    print(f"- Average Time / Epoch: {avg_epoch_time:.3f} s")
    print(f"- System Memory (RAM): {ram_str}")
    print(f"- GPU Memory (VRAM):   {vram_str}")
    print(f"- Has NaN / Inf:       {has_nan_or_inf}")
    print(f"- Final Train Loss:    {results['final_train_loss']:.6f}")
    print(f"- Final Val Loss:      {results['final_val_loss']:.6f}")
    print("\n--- EXPERIMENT TIME ESTIMATIONS (REAL DATA) ---")
    print(f"1. Single Complete Run (100 Epochs):   ~{results['est_1_run_min']:.2f} minutes")
    print(f"2. 20 Main Experiment Runs (5F x 4db): ~{results['est_20_runs_min']:.2f} minutes ({time_20_runs_sec/3600.0:.2f} hrs)")
    print(f"3. 10 Seed Robustness Runs:            ~{results['est_10_seed_runs_min']:.2f} minutes ({time_10_seed_runs_sec/3600.0:.2f} hrs)")
    print("==========================================================")

    return results


if __name__ == "__main__":
    run_pilot_experiment(fold=1, d_b=8, seed=42, epochs=10, batch_size=128)
