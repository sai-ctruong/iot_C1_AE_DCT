"""Automated Test Evaluation Pipeline across 5 Folds and 15 Test Subjects for PPG-DaLiA."""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, Any, List

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.normalize import load_norm_stats, denormalize
from src.baseline_dct import dct_encode_topk, dct_decode
from src.codec import encode_ae_bytes, decode_ae_bytes, encode_dct_bytes, decode_dct_bytes
from src.model import C1Autoencoder
from src.evaluate import compute_channel_metrics
from src.metrics import compute_equal_byte_budget
from src.results_schema import create_result_row, validate_results_schema

FOLDS = [1, 2, 3, 4, 5]
DB_FACTORS = [16, 8, 4, 2]
CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]


def evaluate_all_test_folds(
    output_csv_path: Path = PROJECT_ROOT / "results" / "results.csv"
) -> List[Dict[str, Any]]:
    """
    Evaluate all 20 AE checkpoints (5 Folds x 4 d_b) and paired DCT baselines
    on real PPG-DaLiA Test set windows (Subjects S1..S15 across Folds 1..5).
    Populates results/results.csv according to TASK 20 schema.
    """
    print("==========================================================================")
    print("      EVALUATING ALL 20 AE CHECKPOINTS AND DCT BASELINES ON TEST SET      ")
    print("==========================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    results_list = []
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for fold in FOLDS:
        proc_test_path = PROJECT_ROOT / "data" / "processed" / f"fold{fold}" / "test.npz"
        norm_stats_path = PROJECT_ROOT / "configs" / f"norm_stats_fold{fold}.json"

        if not proc_test_path.exists():
            raise FileNotFoundError(f"Processed test data missing for fold {fold} at {proc_test_path}")

        if not norm_stats_path.exists():
            raise FileNotFoundError(f"Norm stats missing for fold {fold} at {norm_stats_path}")

        npz_data = np.load(proc_test_path, allow_pickle=True)
        test_windows_norm = npz_data["windows"].astype(np.float32)  # Shape (N_win, 4, 512)
        test_meta_arr = npz_data["metadata"]
        test_meta_list = test_meta_arr.item() if test_meta_arr.ndim == 0 else list(test_meta_arr)

        norm_stats = load_norm_stats(norm_stats_path)
        num_windows = len(test_windows_norm)

        print(f"\n--- FOLD {fold}: Loaded {num_windows} Test Windows (Subjects: {sorted(list(set(m['subject'] for m in test_meta_list)))}) ---")

        for d_b in DB_FACTORS:
            # ----------------------------------------------------------------
            # 1. AUTOENCODER EVALUATION
            # ----------------------------------------------------------------
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"fold{fold:02d}_db{d_b:02d}_seed42.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing AE checkpoint at {ckpt_path}")

            ckpt_data = torch.load(ckpt_path, map_location=device)
            model = C1Autoencoder(d_b=d_b).to(device)
            model.load_state_dict(ckpt_data["model_state_dict"])
            model.eval()

            m_dim = 32 * d_b
            cr_dim_ae = 2048.0 / float(m_dim)

            # Pre-compute DCT equal-dim & equal-byte parameters
            k_eq_dim = m_dim
            cr_dim_dct_dim = 2048.0 / float(k_eq_dim)

            b_info = compute_equal_byte_budget(d_b)
            k_eq_byte = b_info["K_equal_byte"]
            pad_byte_val = b_info["padding"]
            cr_dim_dct_byte = 2048.0 / float(k_eq_byte)

            with torch.no_grad():
                for w_idx in range(num_windows):
                    w_norm = test_windows_norm[w_idx]  # Shape (4, 512)
                    w_meta = test_meta_list[w_idx]

                    # --------------------------------------------------------
                    # A. AE EVALUATION & DUAL LOGGING (equal_dim & equal_byte)
                    # --------------------------------------------------------
                    x_in = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)

                    latent_tensor = model.encode(x_in)
                    latent_np = latent_tensor.cpu().numpy()

                    ae_bytes = encode_ae_bytes(latent_np, d_b=d_b, profile_id=0)
                    nbytes_ae = len(ae_bytes)

                    latent_dec_np, _ = decode_ae_bytes(ae_bytes)
                    latent_dec_tensor = torch.from_numpy(latent_dec_np.reshape(1, d_b, 32)).float().to(device)

                    rec_norm_tensor = model.decode(latent_dec_tensor)
                    rec_norm_ae = rec_norm_tensor.squeeze(0).cpu().numpy()

                    # Denormalize to physical domain
                    ref_phys = denormalize(w_norm, norm_stats)
                    rec_phys_ae = denormalize(rec_norm_ae, norm_stats)

                    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                        sigma_c = float(norm_stats["std"][c_idx])
                        ch_m_ae = compute_channel_metrics(
                            ref_phys[c_idx], rec_phys_ae[c_idx], channel_name=ch_name, sigma_train=sigma_c
                        )

                        # Log AE row for comparison_type="equal_dim"
                        r_ae_dim = create_result_row(
                            fold=fold, subject=w_meta["subject"], seed=42, dataset="PPG-DaLiA", method="AE",
                            comparison_type="equal_dim", db=d_b, M=m_dim, K=0, representation_count=m_dim,
                            channel=ch_name, window_id=w_meta["window_id"], start_index=w_meta["start_index"],
                            nbytes=nbytes_ae, CR_dim=cr_dim_ae, CR_byte_64=8192.0 / float(nbytes_ae),
                            CR_byte_native=5120.0 / float(nbytes_ae), PRD=ch_m_ae["prd"], PRDN=ch_m_ae["prdn"],
                            RMSE=ch_m_ae["rmse"], valid_prd=ch_m_ae["valid_prd"], valid_prdn=ch_m_ae["valid_prdn"],
                            metric_valid=ch_m_ae["valid_prdn"], checkpoint=ckpt_path.name,
                            config_id=f"ae_f{fold}_db{d_b}_seed42", budget=d_b
                        )
                        results_list.append(r_ae_dim)

                        # Log AE row for comparison_type="equal_byte"
                        r_ae_byte = create_result_row(
                            fold=fold, subject=w_meta["subject"], seed=42, dataset="PPG-DaLiA", method="AE",
                            comparison_type="equal_byte", db=d_b, M=m_dim, K=0, representation_count=m_dim,
                            channel=ch_name, window_id=w_meta["window_id"], start_index=w_meta["start_index"],
                            nbytes=nbytes_ae, CR_dim=cr_dim_ae, CR_byte_64=8192.0 / float(nbytes_ae),
                            CR_byte_native=5120.0 / float(nbytes_ae), PRD=ch_m_ae["prd"], PRDN=ch_m_ae["prdn"],
                            RMSE=ch_m_ae["rmse"], valid_prd=ch_m_ae["valid_prd"], valid_prdn=ch_m_ae["valid_prdn"],
                            metric_valid=ch_m_ae["valid_prdn"], checkpoint=ckpt_path.name,
                            config_id=f"ae_f{fold}_db{d_b}_seed42", budget=d_b
                        )
                        results_list.append(r_ae_byte)

                    # --------------------------------------------------------
                    # B. DCT EQUAL-DIMENSION BASELINE (comparison_type="equal_dim", K=M)
                    # --------------------------------------------------------
                    topk_vals_dim, topk_idxs_dim, _, _ = dct_encode_topk(w_norm, k=k_eq_dim)
                    dct_bytes_dim = encode_dct_bytes(topk_vals_dim, topk_idxs_dim, d_b=d_b, profile_id=0, pad_bytes=0)
                    nbytes_dct_dim = len(dct_bytes_dim)

                    dec_vals_dim, dec_idxs_dim, _ = decode_dct_bytes(dct_bytes_dim)
                    sparse_dct_dec_dim = np.zeros((4, 512), dtype=np.float32)
                    np.put(sparse_dct_dec_dim, dec_idxs_dim, dec_vals_dim)

                    rec_norm_dct_dim = dct_decode(sparse_dct_dec_dim)
                    rec_phys_dct_dim = denormalize(rec_norm_dct_dim, norm_stats)

                    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                        sigma_c = float(norm_stats["std"][c_idx])
                        ch_m_dct_dim = compute_channel_metrics(
                            ref_phys[c_idx], rec_phys_dct_dim[c_idx], channel_name=ch_name, sigma_train=sigma_c
                        )

                        r_dct_dim = create_result_row(
                            fold=fold, subject=w_meta["subject"], seed=42, dataset="PPG-DaLiA", method="DCT",
                            comparison_type="equal_dim", db=d_b, M=m_dim, K=k_eq_dim, representation_count=k_eq_dim,
                            channel=ch_name, window_id=w_meta["window_id"], start_index=w_meta["start_index"],
                            nbytes=nbytes_dct_dim, CR_dim=cr_dim_dct_dim, CR_byte_64=8192.0 / float(nbytes_dct_dim),
                            CR_byte_native=5120.0 / float(nbytes_dct_dim), PRD=ch_m_dct_dim["prd"],
                            PRDN=ch_m_dct_dim["prdn"], RMSE=ch_m_dct_dim["rmse"], valid_prd=ch_m_dct_dim["valid_prd"],
                            valid_prdn=ch_m_dct_dim["valid_prdn"], metric_valid=ch_m_dct_dim["valid_prdn"],
                            checkpoint="N/A", config_id=f"dct_eqdim_db{d_b}_k{k_eq_dim}", budget=d_b
                        )
                        results_list.append(r_dct_dim)

                    # --------------------------------------------------------
                    # C. DCT EQUAL-BYTE BASELINE (comparison_type="equal_byte", K=floor(4M/6))
                    # --------------------------------------------------------
                    topk_vals_byte, topk_idxs_byte, _, _ = dct_encode_topk(w_norm, k=k_eq_byte)
                    dct_bytes_byte = encode_dct_bytes(topk_vals_byte, topk_idxs_byte, d_b=d_b, profile_id=0, pad_bytes=pad_byte_val)
                    nbytes_dct_byte = len(dct_bytes_byte)

                    dec_vals_byte, dec_idxs_byte, _ = decode_dct_bytes(dct_bytes_byte)
                    sparse_dct_dec_byte = np.zeros((4, 512), dtype=np.float32)
                    np.put(sparse_dct_dec_byte, dec_idxs_byte, dec_vals_byte)

                    rec_norm_dct_byte = dct_decode(sparse_dct_dec_byte)
                    rec_phys_dct_byte = denormalize(rec_norm_dct_byte, norm_stats)

                    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                        sigma_c = float(norm_stats["std"][c_idx])
                        ch_m_dct_byte = compute_channel_metrics(
                            ref_phys[c_idx], rec_phys_dct_byte[c_idx], channel_name=ch_name, sigma_train=sigma_c
                        )

                        r_dct_byte = create_result_row(
                            fold=fold, subject=w_meta["subject"], seed=42, dataset="PPG-DaLiA", method="DCT",
                            comparison_type="equal_byte", db=d_b, M=m_dim, K=k_eq_byte, representation_count=k_eq_byte,
                            channel=ch_name, window_id=w_meta["window_id"], start_index=w_meta["start_index"],
                            nbytes=nbytes_dct_byte, CR_dim=cr_dim_dct_byte, CR_byte_64=8192.0 / float(nbytes_dct_byte),
                            CR_byte_native=5120.0 / float(nbytes_dct_byte), PRD=ch_m_dct_byte["prd"],
                            PRDN=ch_m_dct_byte["prdn"], RMSE=ch_m_dct_byte["rmse"], valid_prd=ch_m_dct_byte["valid_prd"],
                            valid_prdn=ch_m_dct_byte["valid_prdn"], metric_valid=ch_m_dct_byte["valid_prdn"],
                            checkpoint="N/A", config_id=f"dct_eqbyte_db{d_b}_k{k_eq_byte}", budget=d_b
                        )
                        results_list.append(r_dct_byte)

            print(f"  [Fold {fold} d_b={d_b:02d}] Evaluated {num_windows} windows -> AE & DCT (equal_dim & equal_byte) records logged.")

    # ----------------------------------------------------------------
    # 3. VALIDATE SCHEMA & SAVE TO CSV
    # ----------------------------------------------------------------
    print("\nValidating results schema...")
    validate_results_schema(results_list)

    fieldnames = list(results_list[0].keys())
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    total_elapsed = time.time() - t_start
    print(f"\n[SUCCESS] Full test evaluation completed in {total_elapsed:.2f}s!")
    print(f"Total evaluated rows: {len(results_list)}")
    print(f"Results saved at: {output_csv_path}")

    return results_list


if __name__ == "__main__":
    evaluate_all_test_folds()
