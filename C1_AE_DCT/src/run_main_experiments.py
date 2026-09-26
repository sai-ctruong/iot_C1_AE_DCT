"""Automated Launcher for Main Experiment: 20 AE Runs (5 Folds x 4 d_b x Seed 42) on Real PPG-DaLiA."""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.utils import load_folds, validate_folds
from src.normalize import load_norm_stats
from src.train import train_model, set_seed
from src.prepare_data import find_dataset_root, preprocess_all_subjects, prepare_folds_processed_data

FOLDS = [1, 2, 3, 4, 5]
DB_FACTORS = [16, 8, 4, 2]
DEFAULT_SEED = 42


def get_experiment_name(fold: int, d_b: int, seed: int = DEFAULT_SEED) -> str:
    """Generate standardized non-overlapping experiment run name."""
    return f"fold{fold:02d}_db{d_b:02d}_seed{seed}"


def init_manifest_csv(manifest_path: Path) -> None:
    """Initialize CSV manifest file with header if it does not exist."""
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold",
                "db",
                "seed",
                "status",
                "best_epoch",
                "best_val_loss",
                "time_sec",
                "checkpoint_path",
                "dataset",
                "data_type"
            ])


def read_completed_experiments(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read existing manifest CSV to support automatic resume capability."""
    completed = {}
    if not manifest_path.exists():
        return completed

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_key = f"fold{int(row['fold']):02d}_db{int(row['db']):02d}_seed{int(row['seed'])}"
            completed[exp_key] = row
    return completed


def update_manifest_csv(manifest_path: Path, row_dict: Dict[str, Any]) -> None:
    """Append or update experiment run result in CSV manifest."""
    completed = read_completed_experiments(manifest_path)
    exp_key = get_experiment_name(row_dict["fold"], row_dict["db"], row_dict["seed"])

    completed[exp_key] = {
        "fold": str(row_dict["fold"]),
        "db": str(row_dict["db"]),
        "seed": str(row_dict["seed"]),
        "status": str(row_dict["status"]),
        "best_epoch": str(row_dict["best_epoch"]),
        "best_val_loss": f"{float(row_dict['best_val_loss']):.6f}",
        "time_sec": f"{float(row_dict['time_sec']):.2f}",
        "checkpoint_path": str(row_dict["checkpoint_path"]),
        "dataset": str(row_dict.get("dataset", "PPG-DaLiA")),
        "data_type": str(row_dict.get("data_type", "real")),
    }

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "db",
            "seed",
            "status",
            "best_epoch",
            "best_val_loss",
            "time_sec",
            "checkpoint_path",
            "dataset",
            "data_type"
        ])
        for record in completed.values():
            writer.writerow([
                record["fold"],
                record["db"],
                record["seed"],
                record["status"],
                record["best_epoch"],
                record["best_val_loss"],
                record["time_sec"],
                record["checkpoint_path"],
                record["dataset"],
                record["data_type"]
            ])


def load_fold_real_data(fold: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load preprocessed real PPG-DaLiA train data and val windows grouped by subject."""
    proc_dir = PROJECT_ROOT / "data" / "processed" / f"fold{fold}"
    train_npz = proc_dir / "train.npz"
    val_npz = proc_dir / "val.npz"

    if not train_npz.exists() or not val_npz.exists():
        print(f"[MAIN EXP] Processed data for fold {fold} not found. Running preprocessing pipeline...")
        root = find_dataset_root()
        subj_map = preprocess_all_subjects(dataset_dir=root)
        prepare_folds_processed_data(subj_map)

    train_windows = np.load(train_npz, allow_pickle=True)["windows"].astype(np.float32)
    val_npz_data = np.load(val_npz, allow_pickle=True)
    val_windows = val_npz_data["windows"].astype(np.float32)
    val_meta = val_npz_data["metadata"]
    if val_meta.ndim == 0:
        val_meta_list = val_meta.item()
    else:
        val_meta_list = list(val_meta)

    val_windows_by_subject = {}
    for win, meta in zip(val_windows, val_meta_list):
        subj = meta["subject"] if isinstance(meta, dict) and "subject" in meta else "val_unknown"
        if subj not in val_windows_by_subject:
            val_windows_by_subject[subj] = []
        val_windows_by_subject[subj].append(win)

    for subj in val_windows_by_subject:
        val_windows_by_subject[subj] = np.stack(val_windows_by_subject[subj], axis=0).astype(np.float32)

    return train_windows, val_windows_by_subject


def run_main_experiments(
    max_epoch: int = 100,
    early_stopping_patience: int = 10,
    batch_size: int = 128,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Run all 20 main Autoencoder experiment combinations (5 folds x 4 d_b x seed 42) on real PPG-DaLiA data.
    """
    print("==========================================================================")
    print(f"      MAIN EXPERIMENT LAUNCHER: 20 AE RUNS (5F x 4db x S{seed}) ON PPG-DaLiA     ")
    print(f"      max_epoch={max_epoch}, patience={early_stopping_patience}, batch_size={batch_size}")
    print("==========================================================================")

    folds_info = load_folds()
    validate_folds(folds_info)

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"
    configs_dir = PROJECT_ROOT / "configs"
    exp_config_dir = configs_dir / "experiment_configs"
    results_dir = PROJECT_ROOT / "results"

    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    exp_config_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    manifest_path = results_dir / "experiment_manifest.csv"
    init_manifest_csv(manifest_path)

    completed_dict = read_completed_experiments(manifest_path)

    run_results = []
    total_runs = len(FOLDS) * len(DB_FACTORS)
    run_counter = 0
    total_start_time = time.time()

    for fold in FOLDS:
        norm_stats_file = configs_dir / f"norm_stats_fold{fold}.json"
        train_windows, val_windows_by_subject = load_fold_real_data(fold)

        for d_b in DB_FACTORS:
            run_counter += 1
            exp_name = get_experiment_name(fold, d_b, seed)
            ckpt_path = ckpt_dir / f"{exp_name}.pt"
            log_path = log_dir / f"{exp_name}.json"
            exp_config_file = exp_config_dir / f"{exp_name}.json"

            # Check Resume Condition
            if exp_name in completed_dict and completed_dict[exp_name]["status"] == "COMPLETED" and ckpt_path.exists():
                print(f"[{run_counter:02d}/{total_runs:02d}] SKIPPING (Already Completed & Verified): {exp_name}")
                continue

            print(f"\n[{run_counter:02d}/{total_runs:02d}] RUNNING REAL EXPERIMENT: {exp_name} (Fold {fold}, d_b={d_b}, Seed {seed})")

            # Save experiment config JSON
            exp_config_data = {
                "experiment_name": exp_name,
                "fold": fold,
                "d_b": d_b,
                "seed": seed,
                "batch_size": batch_size,
                "max_epoch": max_epoch,
                "early_stopping_patience": early_stopping_patience,
                "lr": 1e-3,
                "dataset": "PPG-DaLiA",
                "data_type": "real",
                "norm_stats_file": str(norm_stats_file),
                "val_subjects": list(val_windows_by_subject.keys()),
            }
            with open(exp_config_file, "w", encoding="utf-8") as f:
                json.dump(exp_config_data, f, indent=2)

            t_run_start = time.time()

            try:
                train_res = train_model(
                    fold=fold,
                    d_b=d_b,
                    train_windows=train_windows,
                    val_windows_by_subject=val_windows_by_subject,
                    lr=1e-3,
                    weight_decay=0.0,
                    batch_size=batch_size,
                    max_epoch=max_epoch,
                    early_stopping_patience=early_stopping_patience,
                    seed=seed,
                    checkpoints_dir=str(ckpt_dir),
                    logs_dir=str(log_dir)
                )

                default_ckpt = ckpt_dir / f"best_model_fold{fold}_db{d_b}.pt"
                default_log = log_dir / f"train_log_fold{fold}_db{d_b}.json"

                if default_ckpt.exists():
                    if ckpt_path.exists():
                        ckpt_path.unlink()
                    default_ckpt.rename(ckpt_path)

                if default_log.exists():
                    if log_path.exists():
                        log_path.unlink()
                    default_log.rename(log_path)

                t_run_elapsed = time.time() - t_run_start

                manifest_entry = {
                    "fold": fold,
                    "db": d_b,
                    "seed": seed,
                    "status": "COMPLETED",
                    "best_epoch": train_res["best_epoch"],
                    "best_val_loss": train_res["best_val_loss"],
                    "time_sec": t_run_elapsed,
                    "checkpoint_path": str(ckpt_path),
                    "dataset": "PPG-DaLiA",
                    "data_type": "real",
                }

                update_manifest_csv(manifest_path, manifest_entry)
                run_results.append(manifest_entry)

                print(f"[COMPLETED] {exp_name} | Best Epoch: {train_res['best_epoch']} | Best Val Loss: {train_res['best_val_loss']:.6f} | Time: {t_run_elapsed:.2f}s")

            except Exception as e:
                print(f"[FAILED] {exp_name} encountered an error: {e}")
                manifest_entry = {
                    "fold": fold,
                    "db": d_b,
                    "seed": seed,
                    "status": f"FAILED: {str(e)}",
                    "best_epoch": 0,
                    "best_val_loss": -1.0,
                    "time_sec": time.time() - t_run_start,
                    "checkpoint_path": str(ckpt_path),
                    "dataset": "PPG-DaLiA",
                    "data_type": "real",
                }
                update_manifest_csv(manifest_path, manifest_entry)
                raise e

    total_elapsed = time.time() - total_start_time
    print("\n==========================================================================")
    print(f"   ALL 20 REAL MAIN AUTOENCODER RUNS EXECUTED / VERIFIED IN {total_elapsed:.2f} SECONDS!  ")
    print(f"   Manifest CSV saved at: {manifest_path}")
    print("==========================================================================")

    return run_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 20 Main Autoencoder Experiments on PPG-DaLiA dataset.")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 5-epoch smoke test.")
    args = parser.parse_args()

    if args.smoke_test:
        print("[MODE] Running in --smoke-test mode (max_epoch=5, patience=3)")
        run_main_experiments(max_epoch=5, early_stopping_patience=3, batch_size=128, seed=42)
    else:
        print("[MODE] Running in standard MAIN mode (max_epoch=100, patience=10)")
        run_main_experiments(max_epoch=100, early_stopping_patience=10, batch_size=128, seed=42)
