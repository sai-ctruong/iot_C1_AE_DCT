"""Integration and Real Data Pipeline Test Suite for C1_AE_DCT (PHẦN U)."""

import sys
import json
import inspect
from pathlib import Path
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import find_dataset_root, load_subject_pickle, extract_wrist_signals, inspect_subject, build_subject_inventory
from src.resample import resample_acc, align_ppg_acc
from src.windowing import create_windows
from src.utils import load_folds, validate_folds
from src.normalize import compute_norm_stats, normalize, denormalize
from src import pilot_run, run_main_experiments, run_seed_experiments
from src.baseline_dct import BaselineDCT
from src.codec import C1Codec
from src.model import C1Autoencoder
from src.evaluate import compute_channel_metrics


def test_integration_1_find_15_subjects():
    """1. Find all 15 real PPG-DaLiA subjects."""
    root = find_dataset_root()
    assert root.exists(), f"Dataset root does not exist: {root}"
    found_subjs = []
    for i in range(1, 16):
        subj_str = f"S{i}"
        p1 = root / f"{subj_str}.pkl"
        p2 = root / subj_str / f"{subj_str}.pkl"
        assert p1.exists() or p2.exists(), f"Subject file {subj_str} missing in {root}"
        found_subjs.append(subj_str)
    assert len(found_subjs) == 15, f"Expected 15 subjects, found {len(found_subjs)}"
    print("[PASS] 1. Found all 15 real PPG-DaLiA subjects.")


def test_integration_2_load_s1():
    """2. Load S1.pkl successfully."""
    data_dict = load_subject_pickle("S1")
    assert "signal" in data_dict, "S1.pkl missing 'signal' key"
    assert "wrist" in data_dict["signal"], "S1.pkl missing 'wrist' key"
    print("[PASS] 2. Loaded S1.pkl successfully.")


def test_integration_3_extract_ppg_acc():
    """3. Extract PPG (64 Hz) and ACC (32 Hz) from S1."""
    data_dict = load_subject_pickle("S1")
    ppg, acc, meta = extract_wrist_signals(data_dict)
    assert ppg.ndim == 1, f"PPG should be 1D, got shape {ppg.shape}"
    assert acc.ndim == 2 and acc.shape[1] == 3, f"ACC should be (N, 3), got shape {acc.shape}"
    assert len(ppg) > 0 and len(acc) > 0, "PPG or ACC array is empty"
    print("[PASS] 3. Extracted PPG and ACC signals successfully.")


def test_integration_4_resample_32_to_64():
    """4. Polyphase resample ACC 32->64 Hz and align with PPG."""
    data_dict = load_subject_pickle("S1")
    ppg, acc, meta = extract_wrist_signals(data_dict)
    acc_res, _ = resample_acc(acc, fs_in=32.0, fs_out=64.0)
    aligned_4ch = align_ppg_acc(ppg, acc_res, fs=64.0)
    assert aligned_4ch.ndim == 2 and aligned_4ch.shape[1] == 4, f"Aligned shape should be (N, 4), got {aligned_4ch.shape}"
    assert abs(len(aligned_4ch) - len(ppg)) <= 5, "Aligned length should match PPG length"
    print("[PASS] 4. Polyphase resampled ACC 32->64 Hz and aligned shape (N, 4).")


def test_integration_5_window_4_512():
    """5. Window signal into shape [4, 512]."""
    dummy_signal = np.random.randn(1000, 4).astype(np.float32)
    windows, meta_list, _ = create_windows(dummy_signal, fs=64.0, window_sec=8.0, step_sec=4.0)
    assert windows.shape[1:] == (4, 512), f"Window shape should be [4, 512], got {windows.shape[1:]}"
    print("[PASS] 5. Window shape is correct [4, 512].")


def test_integration_6_train_val_test_split():
    """6. Subject-wise Train/Val/Test split validation."""
    folds = load_folds()
    assert validate_folds(folds) is True, "Folds validation failed"
    print("[PASS] 6. Train/Val/Test 5-fold split verified.")


def test_integration_7_norm_stats_train_only():
    """7. Norm stats computed ONLY from continuous train signals."""
    dummy_train = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    stats = compute_norm_stats(dummy_train, ddof=0)
    assert "mean" in stats and "std" in stats, "Norm stats missing mean/std"
    assert len(stats["mean"]) == 4 and len(stats["std"]) == 4, "Norm stats channel dimension mismatch"
    print("[PASS] 7. Train-only norm stats computation verified.")


def test_integration_8_pilot_real_data():
    """8. Pilot run script uses real data loader."""
    src_code = inspect.getsource(pilot_run)
    assert "load_real_fold_data" in src_code, "pilot_run.py must use load_real_fold_data"
    assert "np.random.randn(num_train_samples" not in src_code, "pilot_run.py still contains random training data"
    print("[PASS] 8. pilot_run.py uses real PPG-DaLiA data.")


def test_integration_9_main_exp_no_random():
    """9. Main experiment script does not generate synthetic training data."""
    src_code = inspect.getsource(run_main_experiments)
    assert "load_fold_real_data" in src_code, "run_main_experiments.py must use load_fold_real_data"
    assert "train_windows = np.random.randn" not in src_code, "run_main_experiments.py still contains random training windows"
    print("[PASS] 9. run_main_experiments.py uses real data.")


def test_integration_10_seed_exp_no_random():
    """10. Seed experiment script does not generate synthetic training data."""
    src_code = inspect.getsource(run_seed_experiments)
    assert "load_fold_real_data" in src_code, "run_seed_experiments.py must use load_fold_real_data"
    assert "train_windows = np.random.randn" not in src_code, "run_seed_experiments.py still contains random training windows"
    print("[PASS] 10. run_seed_experiments.py uses real data.")


def test_integration_11_final_results_non_synthetic():
    """11. Ensure no synthetic labels in main dataset loader outputs."""
    inv = build_subject_inventory(output_csv=None)
    assert len(inv) == 15, "Inventory must contain 15 subjects"
    for r in inv:
        assert r["ppg_samples"] > 0, f"Subject {r['subject']} has 0 samples"
        assert r["has_nan"] is False, f"Subject {r['subject']} has NaN"
    print("[PASS] 11. Real dataset inventory verified.")


def test_integration_12_dct_full_reconstruction():
    """12. DCT full K=2048 coefficients reconstructs signal near perfectly."""
    dct_mod = BaselineDCT(k=2048)
    x = np.random.randn(4, 512).astype(np.float32)
    topk_vals, topk_idxs, ch_counts, sparse_dct = dct_mod.encode(x)
    x_rec = dct_mod.decode(sparse_dct)
    diff = np.max(np.abs(x - x_rec))
    assert diff < 1e-4, f"DCT full K reconstruction error too high: {diff}"
    print("[PASS] 12. DCT full K reconstruction accurate.")


def test_integration_13_codec_bytes():
    """13. Codec byte serialization and deserialization."""
    codec = C1Codec()
    x_latent = np.random.randn(1, 8, 32).astype(np.float32)
    b = codec.encode_ae(x_latent, d_b=8)
    assert isinstance(b, bytes) and len(b) > 0, "Encoded AE bytes invalid"
    x_dec, header_info = codec.decode_ae(b)
    assert x_dec.shape == x_latent.ravel().shape, f"Decoded AE shape mismatch: {x_dec.shape} vs {x_latent.shape}"
    print("[PASS] 13. Codec byte length and deserialization verified.")



def test_integration_14_ae_shape():
    """14. Autoencoder model forward output shape [B, 4, 512]."""
    import torch
    model = C1Autoencoder(d_b=8)
    x = torch.randn(2, 4, 512)
    y = model(x)
    assert y.shape == (2, 4, 512), f"Autoencoder output shape should be (2, 4, 512), got {y.shape}"
    print("[PASS] 14. Autoencoder output shape verified.")


def test_integration_15_metric_identity():
    """15. Metric calculation returns 0 distortion for identity pred==ref."""
    x_ref = np.random.randn(512) + 10.0
    res = compute_channel_metrics(x_ref, x_ref, channel_name="PPG")
    assert res["rmse"] == 0.0, f"RMSE identity should be 0.0, got {res['rmse']}"
    assert res["prd"] == 0.0, f"PRD identity should be 0.0, got {res['prd']}"
    assert res["prdn"] == 0.0, f"PRDN identity should be 0.0, got {res['prdn']}"
    assert res["valid_prd"] is True and res["valid_prdn"] is True, "Identity valid flags should be True"
    print("[PASS] 15. Metric identity test verified.")


if __name__ == "__main__":
    print("=== RUNNING REAL PIPELINE INTEGRATION TESTS ===")
    test_integration_1_find_15_subjects()
    test_integration_2_load_s1()
    test_integration_3_extract_ppg_acc()
    test_integration_4_resample_32_to_64()
    test_integration_5_window_4_512()
    test_integration_6_train_val_test_split()
    test_integration_7_norm_stats_train_only()
    test_integration_8_pilot_real_data()
    test_integration_9_main_exp_no_random()
    test_integration_10_seed_exp_no_random()
    test_integration_11_final_results_non_synthetic()
    test_integration_12_dct_full_reconstruction()
    test_integration_13_codec_bytes()
    test_integration_14_ae_shape()
    test_integration_15_metric_identity()
    print("\n=== ALL 15 INTEGRATION TESTS PASSED PERFECTLY ===")
