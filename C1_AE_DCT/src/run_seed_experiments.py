"""Automated Launcher for Seed Stability Experiments (d_b=8, Seeds 42, 123 & 999) on Real PPG-DaLiA."""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.utils import load_folds, validate_folds
from src.normalize import load_norm_stats
from src.train import train_model, set_seed
from src.run_main_experiments import load_fold_real_data

FOLDS = [1, 2, 3, 4, 5]
TARGET_DB = 8
SEED_PRIMARY = 42
SEED_ADDITIONAL = [123, 999]
ALL_SEEDS = [42, 123, 999]


def get_seed_exp_name(fold: int, d_b: int, seed: int) -> str:
    """Generate standardized non-overlapping checkpoint name for seed run."""
    return f"fold{fold:02d}_db{d_b:02d}_seed{seed}"


def init_seed_manifest_csv(manifest_path: Path) -> None:
    """Initialize CSV seed manifest file with header if it does not exist."""
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


def read_completed_seed_runs(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read existing seed manifest CSV to support automatic resume capability."""
    completed = {}
    if not manifest_path.exists():
        return completed

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_key = f"fold{int(row['fold']):02d}_db{int(row['db']):02d}_seed{int(row['seed'])}"
            completed[exp_key] = row
    return completed


def update_seed_manifest_csv(manifest_path: Path, row_dict: Dict[str, Any]) -> None:
    """Append or update seed experiment run result in CSV manifest."""
    completed = read_completed_seed_runs(manifest_path)
    exp_key = get_seed_exp_name(row_dict["fold"], row_dict["db"], row_dict["seed"])

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


def compute_seed_stability_summary(
    manifest_path: Path,
    results_csv_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Compute seed stability summary across all 15 test subjects for seeds [42, 123, 999].

    CRITICAL RULE:
    Does NOT treat 15 subjects x 3 seeds as 45 independent subjects.
    Subject identity is fixed across 15 subjects S1..S15 across the 5 test folds.
    If results.csv with test PRD/PRDN/RMSE metrics is available, uses actual test metrics.
    Otherwise reports validation loss by fold.
    """
    completed = read_completed_seed_runs(manifest_path)
    summary_list = []

    # Check if results.csv with actual Test metrics exists
    has_test_results = False
    test_metrics_by_seed = {seed: {} for seed in ALL_SEEDS}

    if results_csv_path is not None and results_csv_path.exists():
        with open(results_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == "AE" and int(row.get("db", 0)) == TARGET_DB:
                    s_seed = int(row.get("seed", 42))
                    s_subj = row.get("subject", "")
                    if s_seed in ALL_SEEDS and row.get("valid_prd") in ["True", "1", "true"]:
                        prd = float(row.get("PRD", 0.0))
                        if s_subj not in test_metrics_by_seed[s_seed]:
                            test_metrics_by_seed[s_seed][s_subj] = []
                        test_metrics_by_seed[s_seed][s_subj].append(prd)

        has_test_results = any(len(v) > 0 for v in test_metrics_by_seed.values())

    for seed in ALL_SEEDS:
        if has_test_results and len(test_metrics_by_seed[seed]) > 0:
            subj_means = [np.mean(val_list) for val_list in test_metrics_by_seed[seed].values()]
            mean_metric = float(np.mean(subj_means))
            std_metric = float(np.std(subj_means, ddof=0))
            metric_name = "PRD_Test (%)"
            count_eval = len(subj_means)
        else:
            seed_val_losses_by_fold = []
            for fold in FOLDS:
                exp_key = get_seed_exp_name(fold, TARGET_DB, seed)
                if exp_key in completed:
                    val_loss = float(completed[exp_key]["best_val_loss"])
                    seed_val_losses_by_fold.append(val_loss)

            if len(seed_val_losses_by_fold) > 0:
                mean_metric = float(np.mean(seed_val_losses_by_fold))
                std_metric = float(np.std(seed_val_losses_by_fold, ddof=0))
            else:
                mean_metric = 0.0
                std_metric = 0.0
            metric_name = "Val_Loss"
            count_eval = len(seed_val_losses_by_fold)

        summary_list.append({
            "seed": seed,
            "d_b": TARGET_DB,
            "metric_type": metric_name,
            "count_evaluated": count_eval,
            "mean_metric_across_subjects": mean_metric,
            "std_across_subjects": std_metric,
        })

    return summary_list


def print_seed_stability_summary(summary_list: List[Dict[str, Any]]) -> None:
    """Print formatted ASCII table of seed stability summary."""
    header = f"| {'Seed':<6} | {'d_b':<4} | {'Metric Type':<12} | {'Count':<6} | {'Mean Metric':<18} | {'Std Across Subjects':<20} |"
    separator = "+" + "-" * 8 + "+" + "-" * 6 + "+" + "-" * 14 + "+" + "-" * 8 + "+" + "-" * 20 + "+" + "-" * 22 + "+"

    print("\n=== SEED STABILITY SUMMARY TABLE (d_b=8) ===")
    print("NOTE: Evaluated across 15 fixed subjects (5 Folds x 3 Test Subjects/Fold).")
    print("      Does NOT treat 15 subjects x 3 seeds as 45 independent subjects.")
    print(separator)
    print(header)
    print(separator)

    for row in summary_list:
        print(
            f"| {row['seed']:<6} | {row['d_b']:<4} | {row['metric_type']:<12} | {row['count_evaluated']:<6} | "
            f"{row['mean_metric_across_subjects']:<18.6f} | {row['std_across_subjects']:<20.6f} |"
        )

    print(separator)


def run_seed_experiments(
    max_epoch: int = 100,
    early_stopping_patience: int = 10,
    batch_size: int = 128,
) -> List[Dict[str, Any]]:
    """
    Run the 10 additional seed stability runs (d_b=8, seeds 123 & 999 on 5 folds).
    Includes seed 42 from Task 16 into final stability summary.
    """
    print("==========================================================================")
    print("  SEED STABILITY EXPERIMENTS (d_b=8, Seeds 123 & 999 on 5 Folds) ON PPG-DaLiA  ")
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

    manifest_path = results_dir / "seed_manifest.csv"
    init_seed_manifest_csv(manifest_path)

    main_manifest_path = results_dir / "experiment_manifest.csv"
    if main_manifest_path.exists():
        with open(main_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["db"]) == TARGET_DB and int(row["seed"]) == SEED_PRIMARY:
                    update_seed_manifest_csv(manifest_path, {
                        "fold": int(row["fold"]),
                        "db": int(row["db"]),
                        "seed": int(row["seed"]),
                        "status": row["status"],
                        "best_epoch": int(row["best_epoch"]),
                        "best_val_loss": float(row["best_val_loss"]),
                        "time_sec": float(row["time_sec"]),
                        "checkpoint_path": row["checkpoint_path"],
                        "dataset": "PPG-DaLiA",
                        "data_type": "real",
                    })

    completed_dict = read_completed_seed_runs(manifest_path)

    run_counter = 0
    total_runs = len(SEED_ADDITIONAL) * len(FOLDS)
    total_start_time = time.time()

    for seed in SEED_ADDITIONAL:
        for fold in FOLDS:
            run_counter += 1
            exp_name = get_seed_exp_name(fold, TARGET_DB, seed)
            ckpt_path = ckpt_dir / f"{exp_name}.pt"
            log_path = log_dir / f"{exp_name}.json"
            exp_config_file = exp_config_dir / f"{exp_name}.json"

            # Check Resume Condition
            if exp_name in completed_dict and completed_dict[exp_name]["status"] == "COMPLETED" and ckpt_path.exists():
                print(f"[{run_counter:02d}/{total_runs:02d}] SKIPPING SEED RUN: {exp_name} (Already Completed)")
                continue

            print(f"\n[{run_counter:02d}/{total_runs:02d}] RUNNING SEED EXPERIMENT: {exp_name} (Fold {fold}, d_b={TARGET_DB}, Seed {seed})")

            norm_stats_file = configs_dir / f"norm_stats_fold{fold}.json"
            train_windows, val_windows_by_subject = load_fold_real_data(fold)

            exp_config_data = {
                "experiment_name": exp_name,
                "fold": fold,
                "d_b": TARGET_DB,
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
                    d_b=TARGET_DB,
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

                default_ckpt = ckpt_dir / f"best_model_fold{fold}_db{TARGET_DB}.pt"
                default_log = log_dir / f"train_log_fold{fold}_db{TARGET_DB}.json"

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
                    "db": TARGET_DB,
                    "seed": seed,
                    "status": "COMPLETED",
                    "best_epoch": train_res["best_epoch"],
                    "best_val_loss": train_res["best_val_loss"],
                    "time_sec": t_run_elapsed,
                    "checkpoint_path": str(ckpt_path),
                    "dataset": "PPG-DaLiA",
                    "data_type": "real",
                }

                update_seed_manifest_csv(manifest_path, manifest_entry)
                print(f"[COMPLETED SEED RUN] {exp_name} | Best Val Loss: {train_res['best_val_loss']:.6f} | Time: {t_run_elapsed:.2f}s")

            except Exception as e:
                print(f"[FAILED SEED RUN] {exp_name} encountered an error: {e}")
                manifest_entry = {
                    "fold": fold,
                    "db": TARGET_DB,
                    "seed": seed,
                    "status": f"FAILED: {str(e)}",
                    "best_epoch": 0,
                    "best_val_loss": -1.0,
                    "time_sec": time.time() - t_run_start,
                    "checkpoint_path": str(ckpt_path),
                    "dataset": "PPG-DaLiA",
                    "data_type": "real",
                }
                update_seed_manifest_csv(manifest_path, manifest_entry)
                raise e

    results_csv_file = results_dir / "results.csv"
    summary_list = compute_seed_stability_summary(manifest_path, results_csv_file)
    print_seed_stability_summary(summary_list)

    summary_json = results_dir / "seed_stability_summary.json"
    summary_csv = results_dir / "seed_stability_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "d_b", "metric_type", "count_evaluated", "mean_metric_across_subjects", "std_across_subjects"])
        for row in summary_list:
            writer.writerow([
                row["seed"],
                row["d_b"],
                row["metric_type"],
                row["count_evaluated"],
                f"{row['mean_metric_across_subjects']:.6f}",
                f"{row['std_across_subjects']:.6f}"
            ])

    total_elapsed = time.time() - total_start_time
    print(f"\n[SUCCESS] Seed stability experiments completed in {total_elapsed:.2f} seconds!")
    print(f"Summary JSON saved at: {summary_json}")
    print(f"Summary CSV saved at:  {summary_csv}")

    return summary_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Seed Stability Experiments on PPG-DaLiA dataset.")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 5-epoch smoke test.")
    args = parser.parse_args()

    if args.smoke_test:
        print("[MODE] Running in --smoke-test mode (max_epoch=5, patience=3)")
        run_seed_experiments(max_epoch=5, early_stopping_patience=3, batch_size=128)
    else:
        print("[MODE] Running in standard SEED mode (max_epoch=100, patience=10)")
        run_seed_experiments(max_epoch=100, early_stopping_patience=10, batch_size=128)
