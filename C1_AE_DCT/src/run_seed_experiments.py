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
from src.normalize import load_norm_stats, denormalize
from src.train import train_model, set_seed
from src.run_main_experiments import load_fold_real_data
from src.model import C1Autoencoder
from src.codec import encode_ae_bytes, decode_ae_bytes
from src.evaluate import compute_channel_metrics
from src.results_schema import create_result_row, validate_results_schema

FOLDS = [1, 2, 3, 4, 5]
TARGET_DB = 8
ALL_SEEDS = [42, 123, 999]
CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]


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


def evaluate_seed_test_metrics(
    seeds: List[int] = ALL_SEEDS,
    d_b: int = TARGET_DB,
    output_dir: Path = PROJECT_ROOT / "results"
) -> Tuple[Path, Path, Path]:
    """
    Evaluate all 15 checkpoints (5 Folds x 3 Seeds for d_b=8) on real Test set windows.

    Generates:
    - seed_results_detail.csv
    - seed_summary_by_subject.csv
    - seed_stability_summary.csv
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating Seed Stability Test Metrics across {len(seeds)} seeds on device: {device}...")

    detail_rows = []

    for seed in seeds:
        for fold in FOLDS:
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"fold{fold:02d}_db{d_b:02d}_seed{seed}.pt"
            proc_test_path = PROJECT_ROOT / "data" / "processed" / f"fold{fold}" / "test.npz"
            norm_stats_path = PROJECT_ROOT / "configs" / f"norm_stats_fold{fold}.json"

            if not ckpt_path.exists():
                raise FileNotFoundError(f"[ERROR] Missing checkpoint for seed evaluation: {ckpt_path}")
            if not proc_test_path.exists():
                raise FileNotFoundError(f"[ERROR] Missing test data: {proc_test_path}")
            if not norm_stats_path.exists():
                raise FileNotFoundError(f"[ERROR] Missing norm stats: {norm_stats_path}")

            npz_data = np.load(proc_test_path, allow_pickle=True)
            test_windows_norm = npz_data["windows"].astype(np.float32)
            test_meta_arr = npz_data["metadata"]
            test_meta_list = test_meta_arr.item() if test_meta_arr.ndim == 0 else list(test_meta_arr)
            norm_stats = load_norm_stats(norm_stats_path)

            ckpt_data = torch.load(ckpt_path, map_location=device)
            model = C1Autoencoder(d_b=d_b).to(device)
            model.load_state_dict(ckpt_data["model_state_dict"])
            model.eval()

            m_dim = 32 * d_b
            cr_dim_ae = 2048.0 / float(m_dim)

            with torch.no_grad():
                for w_idx in range(len(test_windows_norm)):
                    w_norm = test_windows_norm[w_idx]
                    w_meta = test_meta_list[w_idx]

                    x_in = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)
                    latent_tensor = model.encode(x_in)
                    latent_np = latent_tensor.cpu().numpy()

                    ae_bytes = encode_ae_bytes(latent_np, d_b=d_b, profile_id=0)
                    nbytes_ae = len(ae_bytes)

                    latent_dec_np, _ = decode_ae_bytes(ae_bytes)
                    latent_dec_tensor = torch.from_numpy(latent_dec_np.reshape(1, d_b, 32)).float().to(device)

                    rec_norm_tensor = model.decode(latent_dec_tensor)
                    rec_norm_ae = rec_norm_tensor.squeeze(0).cpu().numpy()

                    ref_phys = denormalize(w_norm, norm_stats)
                    rec_phys = denormalize(rec_norm_ae, norm_stats)

                    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                        sigma_c = float(norm_stats["std"][c_idx])
                        ch_m = compute_channel_metrics(ref_phys[c_idx], rec_phys[c_idx], channel_name=ch_name, sigma_train=sigma_c)

                        r_ae = create_result_row(
                            fold=fold, subject=w_meta["subject"], seed=seed, dataset="PPG-DaLiA", method="AE",
                            comparison_type="equal_byte", db=d_b, M=m_dim, K=0, representation_count=m_dim,
                            channel=ch_name, window_id=w_meta["window_id"], start_index=w_meta["start_index"],
                            nbytes=nbytes_ae, CR_dim=cr_dim_ae, CR_byte_64=8192.0 / float(nbytes_ae),
                            CR_byte_native=5120.0 / float(nbytes_ae), PRD=ch_m["prd"], PRDN=ch_m["prdn"],
                            RMSE=ch_m["rmse"], valid_prd=ch_m["valid_prd"], valid_prdn=ch_m["valid_prdn"],
                            metric_valid=ch_m["valid_prdn"], checkpoint=ckpt_path.name,
                            config_id=f"ae_f{fold}_db{d_b}_seed{seed}", budget=d_b
                        )
                        detail_rows.append(r_ae)

    # 1. Save results/seed_results_detail.csv
    detail_csv = output_dir / "seed_results_detail.csv"
    validate_results_schema(detail_rows)
    fieldnames = list(detail_rows[0].keys())
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    # 2. Subject Aggregation -> results/seed_summary_by_subject.csv
    subj_group = {}
    for r in detail_rows:
        key = (int(r["seed"]), str(r["subject"]), str(r["channel"]))
        if key not in subj_group:
            subj_group[key] = {"fold": int(r["fold"]), "db": int(r["db"]), "prd": [], "prdn": [], "rmse": []}
        if r["valid_prdn"]:
            subj_group[key]["prd"].append(float(r["PRD"]))
            subj_group[key]["prdn"].append(float(r["PRDN"]))
            subj_group[key]["rmse"].append(float(r["RMSE"]))

    subj_rows = []
    for (seed, subj, ch), val_dict in subj_group.items():
        if val_dict["prd"]:
            subj_rows.append({
                "seed": seed,
                "subject": subj,
                "fold": val_dict["fold"],
                "channel": ch,
                "db": val_dict["db"],
                "PRD_mean": round(float(np.mean(val_dict["prd"])), 6),
                "PRDN_mean": round(float(np.mean(val_dict["prdn"])), 6),
                "RMSE_mean": round(float(np.mean(val_dict["rmse"])), 6),
                "valid_windows": len(val_dict["prd"]),
            })

    subj_csv = output_dir / "seed_summary_by_subject.csv"
    with open(subj_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "subject", "fold", "channel", "db", "PRD_mean", "PRDN_mean", "RMSE_mean", "valid_windows"])
        writer.writeheader()
        writer.writerows(subj_rows)

    # 3. Overall Seed Stability Summary -> results/seed_stability_summary.csv
    # Equal weight (1/15) across all 15 subjects S1..S15 for each seed & channel
    summary_rows = []
    for seed in seeds:
        for ch in CHANNEL_NAMES:
            seed_ch_rows = [r for r in subj_rows if r["seed"] == seed and r["channel"] == ch]
            n_subjs = len(seed_ch_rows)

            prd_vals = [r["PRD_mean"] for r in seed_ch_rows]
            prdn_vals = [r["PRDN_mean"] for r in seed_ch_rows]
            rmse_vals = [r["RMSE_mean"] for r in seed_ch_rows]

            summary_rows.append({
                "seed": seed, "channel": ch, "metric": "PRD (%)", "n_subjects": n_subjs,
                "mean": round(float(np.mean(prd_vals)), 6), "std": round(float(np.std(prd_vals, ddof=0)), 6)
            })
            summary_rows.append({
                "seed": seed, "channel": ch, "metric": "PRDN (%)", "n_subjects": n_subjs,
                "mean": round(float(np.mean(prdn_vals)), 6), "std": round(float(np.std(prdn_vals, ddof=0)), 6)
            })
            summary_rows.append({
                "seed": seed, "channel": ch, "metric": "RMSE", "n_subjects": n_subjs,
                "mean": round(float(np.mean(rmse_vals)), 6), "std": round(float(np.std(rmse_vals, ddof=0)), 6)
            })

    summary_csv = output_dir / "seed_stability_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "channel", "metric", "n_subjects", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[SUCCESS] Exported seed test evaluation CSVs:")
    print(f"  - Detail rows ({len(detail_rows)}): {detail_csv.name}")
    print(f"  - Subject summary ({len(subj_rows)}): {subj_csv.name}")
    print(f"  - Stability summary ({len(summary_rows)}): {summary_csv.name}")

    return detail_csv, subj_csv, summary_csv


def print_seed_stability_test_summary(summary_csv: Path) -> None:
    """Print formatted ASCII table of seed stability test metrics across 15 subjects."""
    rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    header = f"| {'Seed':<6} | {'Channel':<7} | {'Metric':<10} | {'N_Subj':<6} | {'Mean (Test)':<14} | {'Std Across Subjects':<20} |"
    separator = "+" + "-" * 8 + "+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 8 + "+" + "-" * 16 + "+" + "-" * 22 + "+"

    print("\n==========================================================================")
    print("      FINAL SEED STABILITY TEST METRICS SUMMARY (d_b=8, PPG-DaLiA)       ")
    print("==========================================================================")
    print("NOTE: Evaluated across 15 Test Subjects S1..S15 with equal weight (1/15).")
    print(separator)
    print(header)
    print(separator)

    for r in rows:
        mean_val = float(r["mean"])
        std_val = float(r["std"])
        print(
            f"| {r['seed']:<6} | {r['channel']:<7} | {r['metric']:<10} | {r['n_subjects']:<6} | "
            f"{mean_val:<14.4f} | {std_val:<20.4f} |"
        )

    print(separator)


def run_seed_experiments(
    max_epoch: int = 100,
    early_stopping_patience: int = 10,
    batch_size: int = 128,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Run seed stability experiments (d_b=8, seeds 42, 123 & 999 across 5 folds).

    - Reuses existing completed checkpoints (seed 42 and completed seed 123 runs).
    - Only executes missing training runs.
    - Evaluates all 15 checkpoints on 15 Test subjects.
    - Generates seed_results_detail.csv, seed_summary_by_subject.csv, seed_stability_summary.csv.
    """
    print("==========================================================================")
    print("  SEED STABILITY EXPERIMENTS (d_b=8, Seeds 42, 123 & 999 on 5 Folds) ON PPG-DaLiA  ")
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

    # Register seed 42 checkpoints from main experiment manifest into seed manifest
    main_manifest_path = results_dir / "experiment_manifest.csv"
    if main_manifest_path.exists():
        with open(main_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["db"]) == TARGET_DB and int(row["seed"]) == 42:
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

    reused_runs = []
    newly_run = []

    # Iterate through all 15 target seed runs (3 seeds x 5 folds)
    for seed in ALL_SEEDS:
        for fold in FOLDS:
            exp_name = get_seed_exp_name(fold, TARGET_DB, seed)
            ckpt_path = ckpt_dir / f"{exp_name}.pt"
            log_path = log_dir / f"{exp_name}.json"
            exp_config_file = exp_config_dir / f"{exp_name}.json"

            # Check if checkpoint already exists and is completed
            if ckpt_path.exists() and (exp_name in completed_dict and completed_dict[exp_name]["status"] == "COMPLETED"):
                reused_runs.append(exp_name)
                print(f"[REUSE CHECKPOINT] Skipping training for: {exp_name} (Already completed & verified)")
                continue

            print(f"\n[NEW SEED TRAINING RUN] Launching training for: {exp_name} (Fold {fold}, d_b={TARGET_DB}, Seed {seed})")
            newly_run.append(exp_name)

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

    # Verify all 15 checkpoints for d_b=8 exist
    all_ckpt_files = [ckpt_dir / f"{get_seed_exp_name(f, TARGET_DB, s)}.pt" for s in ALL_SEEDS for f in FOLDS]
    existing_ckpts = [p for p in all_ckpt_files if p.exists()]
    assert len(existing_ckpts) == 15, f"[ERROR] Expected 15 checkpoints for d_b=8, found {len(existing_ckpts)}!"

    # Run Test evaluation on all 15 checkpoints
    detail_csv, subj_csv, summary_csv = evaluate_seed_test_metrics(ALL_SEEDS, TARGET_DB, results_dir)
    print_seed_stability_test_summary(summary_csv)

    print("\n==========================================================================")
    print("                    SEED STABILITY RUN SUMMARY                           ")
    print("==========================================================================")
    print(f"Reused Checkpoints ({len(reused_runs)}): {', '.join(reused_runs)}")
    print(f"Newly Trained Runs ({len(newly_run)}): {', '.join(newly_run) if newly_run else 'None (All resumed)'}")
    print(f"Verified d_b=8 Checkpoints: {len(existing_ckpts)}/15 present.")
    print("==========================================================================")

    return reused_runs, newly_run, read_completed_seed_runs(manifest_path)


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
