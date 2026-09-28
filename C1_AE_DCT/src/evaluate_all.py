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
            cr_dim_val = 512.0 / float(m_dim)

            with torch.no_grad():
                for w_idx in range(num_windows):
                    w_norm = test_windows_norm[w_idx]  # Shape (4, 512)
                    w_meta = test_meta_list[w_idx]

                    x_in = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)

                    # Encode -> Codec Serialize -> Codec Deserialize -> Decode
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

                    # Calculate per-channel metrics
                    for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                        ch_m_ae = compute_channel_metrics(ref_phys[c_idx], rec_phys_ae[c_idx], channel_name=ch_name)

                        r_ae = create_result_row(
                            fold=fold,
                            subject=w_meta["subject"],
                            seed=42,
                            method="AE",
                            db=d_b,
                            K=0,
                            channel=ch_name,
                            window_id=w_meta["window_id"],
                            start_index=w_meta["start_index"],
                            nbytes=nbytes_ae,
                            CR_dim=cr_dim_val,
                            CR_byte_64=8192.0 / float(nbytes_ae),
                            CR_byte_native=5120.0 / float(nbytes_ae),
                            PRD=ch_m_ae["prd"],
                            PRDN=ch_m_ae["prdn"],
                            RMSE=ch_m_ae["rmse"],
                            metric_valid=ch_m_ae["valid_prdn"],
                            checkpoint=ckpt_path.name,
                            config_id=f"ae_f{fold}_db{d_b}_seed42",
                            budget=d_b
                        )
                        results_list.append(r_ae)

            # ----------------------------------------------------------------
            # 2. DCT BASELINE EVALUATION (Equal Byte Budget)
            # ----------------------------------------------------------------
            b_info = compute_equal_byte_budget(d_b)
            k_eq_byte = b_info["K_equal_byte"]
            pad_val = b_info["padding"]

            for w_idx in range(num_windows):
                w_norm = test_windows_norm[w_idx]
                w_meta = test_meta_list[w_idx]

                # DCT Top-K Encode -> Codec Serialize -> Codec Deserialize -> IDCT Decode
                topk_vals, topk_idxs, _, _ = dct_encode_topk(w_norm, k=k_eq_byte)
                dct_bytes = encode_dct_bytes(topk_vals, topk_idxs, d_b=d_b, profile_id=0, pad_bytes=pad_val)
                nbytes_dct = len(dct_bytes)

                dec_vals, dec_idxs, _ = decode_dct_bytes(dct_bytes)
                sparse_dct_dec = np.zeros((4, 512), dtype=np.float32)
                np.put(sparse_dct_dec, dec_idxs, dec_vals)

                rec_norm_dct = dct_decode(sparse_dct_dec)

                # Denormalize to physical domain
                ref_phys = denormalize(w_norm, norm_stats)
                rec_phys_dct = denormalize(rec_norm_dct, norm_stats)

                # Calculate per-channel metrics
                for c_idx, ch_name in enumerate(CHANNEL_NAMES):
                    ch_m_dct = compute_channel_metrics(ref_phys[c_idx], rec_phys_dct[c_idx], channel_name=ch_name)

                    r_dct = create_result_row(
                        fold=fold,
                        subject=w_meta["subject"],
                        seed=42,
                        method="DCT",
                        db=d_b,
                        K=k_eq_byte,
                        channel=ch_name,
                        window_id=w_meta["window_id"],
                        start_index=w_meta["start_index"],
                        nbytes=nbytes_dct,
                        CR_dim=cr_dim_val,
                        CR_byte_64=8192.0 / float(nbytes_dct),
                        CR_byte_native=5120.0 / float(nbytes_dct),
                        PRD=ch_m_dct["prd"],
                        PRDN=ch_m_dct["prdn"],
                        RMSE=ch_m_dct["rmse"],
                        metric_valid=ch_m_dct["valid_prdn"],
                        checkpoint="N/A",
                        config_id=f"dct_db{d_b}_k{k_eq_byte}",
                        budget=d_b
                    )
                    results_list.append(r_dct)

            print(f"  [Fold {fold} d_b={d_b:02d}] Evaluated {num_windows} windows -> AE & DCT records logged.")

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
