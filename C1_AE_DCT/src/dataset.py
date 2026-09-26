"""Dataset module for C1_AE_DCT task with real PPG-DaLiA dataset support."""

import os
import sys
import pickle
import csv
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict, Union

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import load_folds, validate_folds



def find_dataset_root(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Dynamically locate the PPG-DaLiA dataset root directory containing S1..S15.

    Candidate relative paths are checked from project root and workspace root.
    """
    candidates = []
    if base_dir is not None:
        candidates.append(Path(base_dir))

    # Current working directory / script directory relative paths
    curr_dir = Path.cwd()
    script_dir = Path(__file__).resolve().parent.parent

    relative_paths = [
        Path("C1_AE_DCT/data/raw"),
        Path("data/raw"),
        Path("ppg+dalia/data/PPG_FieldStudy"),
        Path("../ppg+dalia/data/PPG_FieldStudy"),
        Path("ppg+dalia/PPG_FieldStudy"),
        Path("data/PPG_FieldStudy"),
        Path("PPG_FieldStudy"),
    ]

    for root in [script_dir, curr_dir, script_dir.parent]:
        for rel in relative_paths:
            candidates.append((root / rel).resolve())

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            # Check if S1.pkl or S1/S1.pkl exists
            has_s1 = (cand / "S1.pkl").exists() or (cand / "S1" / "S1.pkl").exists()
            if has_s1:
                return cand

    raise FileNotFoundError(
        "PPG-DaLiA dataset root not found. Please verify dataset folder placement "
        "containing S1.pkl .. S15.pkl or S1/S1.pkl .. S15/S15.pkl."
    )


def load_subject_pickle(
    subject_id: Union[str, int],
    dataset_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load raw pickle file for a specific subject (S1 to S15)."""
    if isinstance(subject_id, int):
        subj_str = f"S{subject_id}"
    else:
        subj_str = str(subject_id).upper()
        if not subj_str.startswith("S"):
            subj_str = f"S{subj_str}"

    root = find_dataset_root(dataset_dir)
    p1 = root / f"{subj_str}.pkl"
    p2 = root / subj_str / f"{subj_str}.pkl"

    if p1.exists():
        file_path = p1
    elif p2.exists():
        file_path = p2
    else:
        raise FileNotFoundError(f"Subject pickle file for {subj_str} not found under {root}")

    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")

    return data_dict


def extract_wrist_signals(data_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract wrist PPG (BVP, 64 Hz) and wrist ACC (32 Hz, 3 channels) from PPG-DaLiA subject dict.

    Returns:
    --------
    ppg : np.ndarray
        Wrist BVP signal array of shape (N,) at 64 Hz.
    acc : np.ndarray
        Wrist ACC signal array of shape (M, 3) at 32 Hz.
    meta : Dict[str, Any]
        Subject metadata dictionary.
    """
    signal_dict = data_dict["signal"]
    wrist_dict = signal_dict["wrist"]

    bvp = wrist_dict["BVP"]  # Shape (N, 1)
    acc = wrist_dict["ACC"]  # Shape (M, 3)

    ppg = np.asarray(bvp, dtype=np.float64).flatten()
    acc = np.asarray(acc, dtype=np.float64)

    meta = {
        "subject": str(data_dict.get("subject", "Unknown")),
        "ppg_fs": 64.0,
        "acc_fs": 32.0,
        "activity": data_dict.get("activity", None),
        "questionnaire": data_dict.get("questionnaire", None),
        "rpeaks": data_dict.get("rpeaks", None),
    }

    return ppg, acc, meta


def inspect_subject(
    subject_id: Union[str, int],
    dataset_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Inspect subject data integrity, shape, duration, NaN/Inf presence."""
    if isinstance(subject_id, int):
        subj_str = f"S{subject_id}"
    else:
        subj_str = str(subject_id).upper()
        if not subj_str.startswith("S"):
            subj_str = f"S{subj_str}"

    data_dict = load_subject_pickle(subj_str, dataset_dir=dataset_dir)
    ppg, acc, meta = extract_wrist_signals(data_dict)

    ppg_samples = len(ppg)
    acc_samples = len(acc)
    ppg_duration = ppg_samples / 64.0
    acc_duration = acc_samples / 32.0

    has_nan = bool(np.isnan(ppg).any() or np.isnan(acc).any())
    has_inf = bool(np.isinf(ppg).any() or np.isinf(acc).any())

    duration_diff = abs(ppg_duration - acc_duration)
    notes = []
    if has_nan:
        notes.append("NaN detected")
    if has_inf:
        notes.append("Inf detected")
    if duration_diff > 2.0:
        notes.append(f"Duration diff {duration_diff:.2f}s")
    if not notes:
        notes.append("OK")

    return {
        "subject": subj_str,
        "ppg_samples": ppg_samples,
        "acc_samples": acc_samples,
        "ppg_duration_sec": round(ppg_duration, 3),
        "acc_duration_sec": round(acc_duration, 3),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "notes": "; ".join(notes),
    }


def build_subject_inventory(
    dataset_dir: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = "results/data_inventory.csv"
) -> List[Dict[str, Any]]:
    """Inspect all 15 subjects S1..S15 and export inventory CSV report."""
    inventory = []
    missing_subjects = []

    for i in range(1, 16):
        subj_str = f"S{i}"
        try:
            row = inspect_subject(subj_str, dataset_dir=dataset_dir)
            inventory.append(row)
        except Exception as e:
            missing_subjects.append(subj_str)
            inventory.append({
                "subject": subj_str,
                "ppg_samples": 0,
                "acc_samples": 0,
                "ppg_duration_sec": 0.0,
                "acc_duration_sec": 0.0,
                "has_nan": False,
                "has_inf": False,
                "notes": f"MISSING / ERROR: {e}",
            })

    if missing_subjects:
        print(f"[WARNING] Missing subjects in dataset inventory: {missing_subjects}")

    if output_csv is not None:
        out_path = Path(output_csv)
        if not out_path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent
            out_path = base_dir / output_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "subject",
            "ppg_samples",
            "acc_samples",
            "ppg_duration_sec",
            "acc_duration_sec",
            "has_nan",
            "has_inf",
            "notes",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory)

        print(f"[INFO] Data inventory saved to: {out_path}")

    return inventory


class C1Dataset(Dataset):
    """PyTorch Dataset class for C1 Autoencoder & DCT baseline with subject-wise fold partition."""

    def __init__(
        self,
        fold: int = 1,
        split_type: str = "train",
        data_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
    ):
        assert 1 <= fold <= 5, f"Fold index must be between 1 and 5, got {fold}"
        assert split_type in ["train", "val", "test"], f"Invalid split_type: {split_type}"

        self.fold = fold
        self.split_type = split_type
        self.transform = transform

        # Load and validate folds
        self.all_folds = load_folds()
        validate_folds(self.all_folds)

        fold_key = f"fold_{fold}"
        self.subjects = self.all_folds[fold_key][split_type]

        # Determine processed directory
        if data_dir is not None:
            self.processed_dir = Path(data_dir)
        else:
            base_dir = Path(__file__).resolve().parent.parent
            self.processed_dir = base_dir / "data" / "processed"

        processed_file = self.processed_dir / f"fold{fold}" / f"{split_type}.npz"
        self.windows: np.ndarray = np.empty((0, 4, 512), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []

        if processed_file.exists():
            npz_data = np.load(processed_file, allow_pickle=True)
            self.windows = npz_data["windows"].astype(np.float32)
            if "metadata" in npz_data:
                meta_arr = npz_data["metadata"]
                if meta_arr.ndim == 0:
                    self.metadata = meta_arr.item()
                else:
                    self.metadata = list(meta_arr)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.windows[idx]  # Shape: (4, 512)
        if self.transform is not None:
            sample = self.transform(sample)

        tensor_val = torch.from_numpy(sample).float()
        # For reconstruction autoencoder, target == input
        return tensor_val, tensor_val


if __name__ == "__main__":
    folds = load_folds()
    validate_folds(folds)
    root = find_dataset_root()
    print(f"Dataset root identified at: {root}")
    inv = build_subject_inventory(dataset_dir=root)
    print(f"Loaded inventory for {len(inv)} subjects.")
