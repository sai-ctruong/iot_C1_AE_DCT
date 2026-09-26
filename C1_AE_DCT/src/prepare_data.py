"""End-to-end Data Preprocessing Pipeline for PPG-DaLiA dataset.

Pipeline Steps:
1. Locate PPG-DaLiA raw pickle files (S1..S15).
2. Extract wrist PPG (64 Hz) and wrist ACC (32 Hz, 3 channels).
3. Polyphase resample ACC from 32 Hz to 64 Hz using scipy.signal.resample_poly.
4. Align PPG and resampled ACC into continuous 4-channel matrix [PPG, ACCx, ACCy, ACCz] of shape (N, 4).
5. Load 5-fold subject-wise split.
6. For each fold (1..5):
   - Calculate Z-score norm stats (mean, std) ONLY from continuous Train subjects (ddof=0).
   - Save configs/norm_stats_fold{1..5}.json.
   - Normalize subject continuous signals using train-only norm stats.
   - Generate 8s windows (512 samples): 4s step (50% overlap) for Train/Val, 8s step (0% overlap) for Test.
   - Save processed arrays and window metadata to data/processed/fold{1..5}/{train|val|test}.npz.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.dataset import find_dataset_root, load_subject_pickle, extract_wrist_signals, build_subject_inventory
from src.resample import resample_acc, align_ppg_acc
from src.normalize import compute_norm_stats, normalize, save_norm_stats
from src.windowing import create_windows
from src.utils import load_folds, validate_folds


def preprocess_all_subjects(dataset_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Read, resample, and align all 15 subjects into continuous 4-channel arrays (64 Hz)."""
    root = find_dataset_root(dataset_dir)
    print(f"[PREPROC] Reading raw PPG-DaLiA dataset from: {root}")

    subject_continuous_map = {}

    for i in range(1, 16):
        subj_str = f"S{i}"
        data_dict = load_subject_pickle(subj_str, dataset_dir=root)
        ppg, acc, meta = extract_wrist_signals(data_dict)

        # Polyphase resample ACC 32 -> 64 Hz
        acc_resampled, _ = resample_acc(acc, fs_in=32.0, fs_out=64.0)

        # Align PPG and ACC into 4 channels: [PPG, ACCx, ACCy, ACCz]
        signal_4ch = align_ppg_acc(ppg, acc_resampled, fs=64.0)

        assert signal_4ch.shape[1] == 4, f"{subj_str}: Expected 4 channels, got shape {signal_4ch.shape}"
        assert not np.isnan(signal_4ch).any(), f"{subj_str}: NaN detected in continuous signal!"
        assert not np.isinf(signal_4ch).any(), f"{subj_str}: Inf detected in continuous signal!"

        subject_continuous_map[subj_str] = signal_4ch
        duration_sec = len(signal_4ch) / 64.0
        print(f"  - {subj_str}: Shape {signal_4ch.shape}, Duration {duration_sec:.2f}s ({len(signal_4ch)} samples)")

    return subject_continuous_map


def prepare_folds_processed_data(
    subject_continuous_map: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Process folds, compute train-only norm stats, generate windows, and export .npz files."""
    base_dir = Path(__file__).resolve().parent.parent

    if output_dir is None:
        output_dir = base_dir / "data" / "processed"
    if configs_dir is None:
        configs_dir = base_dir / "configs"

    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    folds = load_folds()
    validate_folds(folds)

    summary_report = {}

    for fold_idx in range(1, 6):
        fold_key = f"fold_{fold_idx}"
        split_dict = folds[fold_key]

        train_subjs = split_dict["train"]
        val_subjs = split_dict["val"]
        test_subjs = split_dict["test"]

        print(f"\n==================== PROCESSING FOLD {fold_idx} ====================")
        print(f"  Train ({len(train_subjs)}): {train_subjs}")
        print(f"  Val   ({len(val_subjs)}):   {val_subjs}")
        print(f"  Test  ({len(test_subjs)}):  {test_subjs}")

        # 1. Combine continuous Train signals to compute train-only norm stats
        train_signals_list = [subject_continuous_map[s] for s in train_subjs]
        train_continuous_all = np.vstack(train_signals_list)

        norm_stats = compute_norm_stats(train_continuous_all, ddof=0)
        norm_stats_file = configs_dir / f"norm_stats_fold{fold_idx}.json"
        save_norm_stats(norm_stats, norm_stats_file)
        print(f"  Saved norm stats to: {norm_stats_file}")
        print(f"    Means: {[round(m, 4) for m in norm_stats['mean']]}")
        print(f"    Stds:  {[round(s, 4) for s in norm_stats['std']]}")

        fold_dir = output_dir / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_summary = {"fold": fold_idx, "norm_stats": norm_stats}

        for split_name, subjs in [("train", train_subjs), ("val", val_subjs), ("test", test_subjs)]:
            split_windows_list = []
            split_meta_list = []

            step_sec = 4.0 if split_name in ["train", "val"] else 8.0

            for subj in subjs:
                raw_sig = subject_continuous_map[subj]
                # Normalize continuous signal using train-only norm stats
                norm_sig = normalize(raw_sig, norm_stats)

                # Window normalized signal
                subj_wins, subj_meta, _ = create_windows(
                    signal_data=norm_sig,
                    fs=64.0,
                    window_sec=8.0,
                    step_sec=step_sec,
                    subject=subj,
                    fold=fold_idx,
                    split=split_name,
                )

                if len(subj_wins) > 0:
                    split_windows_list.append(subj_wins)
                    split_meta_list.extend(subj_meta)

            if len(split_windows_list) > 0:
                all_split_windows = np.concatenate(split_windows_list, axis=0).astype(np.float32)
            else:
                all_split_windows = np.empty((0, 4, 512), dtype=np.float32)

            save_path = fold_dir / f"{split_name}.npz"
            np.savez_compressed(
                save_path,
                windows=all_split_windows,
                metadata=np.array(split_meta_list, dtype=object),
            )

            print(
                f"  Saved {split_name.upper():<5}: {len(all_split_windows):>5} windows of shape {all_split_windows.shape[1:]} to {save_path.name}"
            )
            fold_summary[f"{split_name}_windows"] = len(all_split_windows)

        summary_report[f"fold_{fold_idx}"] = fold_summary

    print("\n==================== PREPROCESSING COMPLETED ====================")
    return summary_report


def main():
    print("=== STARTING PPG-DaLiA END-TO-END PREPROCESSING PIPELINE ===")
    root = find_dataset_root()
    # 1. Build & save data inventory
    build_subject_inventory(dataset_dir=root)

    # 2. Extract, resample, align continuous signals
    subject_map = preprocess_all_subjects(dataset_dir=root)

    # 3. Create fold splits, calculate norm stats, generate windows & save
    summary = prepare_folds_processed_data(subject_map)
    print("\nPreprocessed dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: Train={v['train_windows']}, Val={v['val_windows']}, Test={v['test_windows']}")


if __name__ == "__main__":
    main()
