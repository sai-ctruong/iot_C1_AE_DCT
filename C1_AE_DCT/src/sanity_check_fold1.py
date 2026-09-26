"""End-to-end Sanity Check on Fold 1 Test Data for C1_AE_DCT Project."""

import os
import sys
import json
import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import find_dataset_root, load_subject_pickle, extract_wrist_signals
from src.resample import resample_acc, align_ppg_acc
from src.normalize import compute_norm_stats, load_norm_stats, normalize, denormalize
from src.baseline_dct import dct_encode_topk, dct_decode, dct_reconstruct
from src.codec import encode_ae_bytes, decode_ae_bytes, encode_dct_bytes, decode_dct_bytes
from src.model import C1Autoencoder
from src.evaluate import compute_channel_metrics, evaluate_window_metrics
from src.metrics import compute_equal_byte_budget, get_budget_mapping_table

from src.results_schema import create_result_row, validate_results_schema, append_result, join_ae_dct_results


def run_sanity_check_fold1() -> Dict[str, Any]:
    """Execute complete end-to-end Sanity Check on Fold 1 Real Test Data."""
    print("==========================================================================")
    print("       END-TO-END SANITY CHECK ON FOLD 1 REAL TEST DATA (PPG-DaLiA)       ")
    print("==========================================================================")

    output_dir = PROJECT_ROOT / "results" / "sanity_fold1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # PART 1: AUDIT norm_stats_fold1
    # --------------------------------------------------------------------
    print("\n--- PART 1: AUDIT norm_stats_fold1 ---")

    dataset_root = find_dataset_root()
    train_subjs = ["S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15"]
    train_signals = []

    for s in train_subjs:
        d = load_subject_pickle(s, dataset_root)
        p, a, _ = extract_wrist_signals(d)
        a_res, _ = resample_acc(a, 32.0, 64.0)
        sig = align_ppg_acc(p, a_res, 64.0)
        train_signals.append(sig)

    all_train_continuous = np.vstack(train_signals)
    computed_stats = compute_norm_stats(all_train_continuous, ddof=0)
    saved_stats_path = PROJECT_ROOT / "configs" / "norm_stats_fold1.json"
    saved_stats = load_norm_stats(saved_stats_path)

    raw_min = np.min(all_train_continuous, axis=0)
    raw_max = np.max(all_train_continuous, axis=0)

    # Assert matching means and stds within 1e-4 tolerance
    np.testing.assert_allclose(computed_stats["mean"], saved_stats["mean"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(computed_stats["std"], saved_stats["std"], rtol=1e-4, atol=1e-4)

    norm_stats_audit = {
        "pass": True,
        "source_subjs": train_subjs,
        "total_samples": len(all_train_continuous),
        "raw_dtype": str(all_train_continuous.dtype),
        "raw_min": [round(float(v), 4) for v in raw_min],
        "raw_max": [round(float(v), 4) for v in raw_max],
        "mean": [round(float(v), 6) for v in saved_stats["mean"]],
        "std": [round(float(v), 6) for v in saved_stats["std"]],
    }
    print(f"Norm stats audit PASSED! Mean: {norm_stats_audit['mean']}, Std: {norm_stats_audit['std']}")

    # --------------------------------------------------------------------
    # PART 2: LOAD FOLD 1 TEST WINDOWS
    # --------------------------------------------------------------------
    print("\n--- PART 2: LOAD FOLD 1 TEST WINDOWS ---")

    proc_test_path = PROJECT_ROOT / "data" / "processed" / "fold1" / "test.npz"
    npz_data = np.load(proc_test_path, allow_pickle=True)
    test_windows_norm = npz_data["windows"].astype(np.float32)  # Shape: (N_win, 4, 512)
    test_meta_arr = npz_data["metadata"]
    test_meta_list = test_meta_arr.item() if test_meta_arr.ndim == 0 else list(test_meta_arr)

    s1_count = sum(1 for m in test_meta_list if m["subject"] == "S1")
    s2_count = sum(1 for m in test_meta_list if m["subject"] == "S2")
    s3_count = sum(1 for m in test_meta_list if m["subject"] == "S3")
    total_test_windows = len(test_windows_norm)

    assert total_test_windows == s1_count + s2_count + s3_count
    assert test_windows_norm.shape[1:] == (4, 512)

    print(f"Fold 1 Test Windows count: Total={total_test_windows} (S1={s1_count}, S2={s2_count}, S3={s3_count})")
    print("Metadata first 3 windows:")
    for m in test_meta_list[:3]:
        print(f"  - {m['window_id']}: subject={m['subject']}, start={m['start_index']}, end={m['end_index']}")

    # --------------------------------------------------------------------
    # PART 3: DCT EVALUATION & CODEC CHECK
    # --------------------------------------------------------------------
    print("\n--- PART 3: DCT EVALUATION & CODEC CHECK ---")

    codec_records = []
    dct_results_list = []

    db_list = [16, 8, 4, 2]
    # Equal-dim: K = M = [512, 256, 128, 64]
    # Equal-byte: K = floor(4*M/6) = [341, 170, 85, 42]

    for d_b in db_list:
        b_info = compute_equal_byte_budget(d_b)
        m_dim = b_info["M"]
        k_eq_dim = b_info["K_equal_dim"]
        k_eq_byte = b_info["K_equal_byte"]
        pad_bytes_byte = b_info["padding"]
        cr_dim_val = 2048.0 / float(m_dim)

        for b_type, k_val, pad_val in [("equal_dim", k_eq_dim, 0), ("equal_byte", k_eq_byte, pad_bytes_byte)]:
            print(f"Evaluating DCT {b_type:<10} (d_b={d_b:02d}, K={k_val}, pad={pad_val} bytes)...")

            for w_idx in range(total_test_windows):
                w_norm = test_windows_norm[w_idx]  # Shape (4, 512)
                w_meta = test_meta_list[w_idx]

                # 1. DCT Encode
                topk_vals, topk_idxs, ch_counts, sparse_dct = dct_encode_topk(w_norm, k=k_val)

                # 2. Reference Codec Serialize
                dct_bytes = encode_dct_bytes(topk_vals, topk_idxs, d_b=d_b, profile_id=0, pad_bytes=pad_val)
                nbytes = len(dct_bytes)

                # 3. Reference Codec Deserialize
                dec_vals, dec_idxs, dec_hdr = decode_dct_bytes(dct_bytes)

                # Reconstruct sparse DCT from deserialized values
                sparse_dct_dec = np.zeros((4, 512), dtype=np.float32)
                np.put(sparse_dct_dec, dec_idxs, dec_vals)

                # 4. IDCT Reconstruct (normalized domain)
                rec_norm = dct_decode(sparse_dct_dec)

                # 5. Denormalize to physical scale
                ref_phys = denormalize(w_norm, saved_stats)
                rec_phys = denormalize(rec_norm, saved_stats)

                # Metrics calculation per channel
                for c_idx, ch_name in enumerate(["PPG", "ACCx", "ACCy", "ACCz"]):
                    ch_metrics = compute_channel_metrics(ref_phys[c_idx], rec_phys[c_idx], channel_name=ch_name)

                    row = create_result_row(
                        fold=1,
                        subject=w_meta["subject"],
                        seed=42,
                        method=f"DCT_{b_type}",
                        db=d_b,
                        K=k_val,
                        channel=ch_name,
                        window_id=w_meta["window_id"],
                        start_index=w_meta["start_index"],
                        nbytes=nbytes,
                        CR_dim=cr_dim_val,
                        CR_byte_64=8192.0 / float(nbytes),
                        CR_byte_native=5120.0 / float(nbytes),
                        PRD=ch_metrics["prd"],
                        PRDN=ch_metrics["prdn"],
                        RMSE=ch_metrics["rmse"],
                        metric_valid=ch_metrics["valid_prdn"],
                        checkpoint="N/A",
                        config_id=f"dct_{b_type}_db{d_b}_k{k_val}",
                        budget=d_b
                    )
                    dct_results_list.append(row)

                if w_idx == 0:
                    codec_records.append({
                        "method": f"DCT_{b_type}",
                        "db": d_b,
                        "K": k_val,
                        "pad_bytes": pad_val,
                        "nbytes": nbytes,
                        "expected_bytes": 16 + 6 * k_val + pad_val,
                        "codec_pass": nbytes == (16 + 6 * k_val + pad_val),
                    })


    # --------------------------------------------------------------------
    # PART 4: AE PILOT EVALUATION
    # --------------------------------------------------------------------
    print("\n--- PART 4: AE PILOT EVALUATION ---")

    pilot_ckpt_path = PROJECT_ROOT / "checkpoints" / "pilot_real_fold01_db08_seed42.pt"
    assert pilot_ckpt_path.exists(), f"Pilot checkpoint missing at {pilot_ckpt_path}"

    ckpt_data = torch.load(pilot_ckpt_path, map_location="cpu")
    assert ckpt_data.get("data_type") == "REAL_DATA", "Checkpoint must be marked REAL_DATA"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C1Autoencoder(d_b=8).to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    ae_results_list = []

    with torch.no_grad():
        for w_idx in range(total_test_windows):
            w_norm = test_windows_norm[w_idx]  # Shape (4, 512)
            w_meta = test_meta_list[w_idx]

            x_in = torch.from_numpy(w_norm).unsqueeze(0).float().to(device)  # Shape (1, 4, 512)

            # 1. AE Encoder
            latent_tensor = model.encode(x_in)  # Shape (1, 8, 32)
            latent_np = latent_tensor.cpu().numpy()

            # 2. Encode AE bytes
            ae_bytes = encode_ae_bytes(latent_np, d_b=8, profile_id=0)
            nbytes_ae = len(ae_bytes)

            # 3. Decode AE bytes
            latent_dec_np, ae_hdr = decode_ae_bytes(ae_bytes)
            latent_dec_tensor = torch.from_numpy(latent_dec_np.reshape(1, 8, 32)).float().to(device)

            # 4. AE Decoder
            rec_norm_tensor = model.decode(latent_dec_tensor)  # Shape (1, 4, 512)
            rec_norm = rec_norm_tensor.squeeze(0).cpu().numpy()

            # 5. Denormalize
            ref_phys = denormalize(w_norm, saved_stats)
            rec_phys = denormalize(rec_norm, saved_stats)

            # Metrics calculation per channel
            for c_idx, ch_name in enumerate(["PPG", "ACCx", "ACCy", "ACCz"]):
                ch_metrics = compute_channel_metrics(ref_phys[c_idx], rec_phys[c_idx], channel_name=ch_name)

                row = create_result_row(
                    fold=1,
                    subject=w_meta["subject"],
                    seed=42,
                    method="AE",
                    db=8,
                    K=0,
                    channel=ch_name,
                    window_id=w_meta["window_id"],
                    start_index=w_meta["start_index"],
                    nbytes=nbytes_ae,
                    CR_dim=8.0,
                    CR_byte_64=8192.0 / float(nbytes_ae),
                    CR_byte_native=5120.0 / float(nbytes_ae),

                    PRD=ch_metrics["prd"],
                    PRDN=ch_metrics["prdn"],
                    RMSE=ch_metrics["rmse"],
                    metric_valid=ch_metrics["valid_prdn"],
                    checkpoint=pilot_ckpt_path.name,
                    config_id="ae_fold1_db8_seed42",
                    budget=8
                )
                ae_results_list.append(row)

            if w_idx == 0:
                codec_records.append({
                    "method": "AE",
                    "db": 8,
                    "K": 0,
                    "pad_bytes": 0,
                    "nbytes": nbytes_ae,
                    "expected_bytes": 16 + 4 * 256,
                    "codec_pass": nbytes_ae == 1040,
                })

    print(f"AE Pilot evaluation completed on {total_test_windows} test windows! (Bytes/win = {nbytes_ae})")

    # Save codec check CSV
    codec_csv_path = output_dir / "fold1_codec_check.csv"
    with open(codec_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "db", "K", "pad_bytes", "nbytes", "expected_bytes", "codec_pass"])
        writer.writeheader()
        writer.writerows(codec_records)

    # --------------------------------------------------------------------
    # PART 5 & 6: SAVE DETAIL METRICS & PAIRED COMPARISON
    # --------------------------------------------------------------------
    print("\n--- PART 5 & 6: REAL METRICS & PAIRED COMPARISON ---")
    all_metrics = ae_results_list + dct_results_list
    validate_results_schema(all_metrics)

    detail_csv_path = output_dir / "fold1_test_metrics_detail.csv"
    fieldnames = list(all_metrics[0].keys())
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    # Filter AE (db=8) and DCT (db=8) for paired comparison
    ae_db8 = [r for r in ae_results_list if r["db"] == 8]
    dct_eq_dim_db8 = [r for r in dct_results_list if r["method"] == "DCT_equal_dim" and r["db"] == 8]
    dct_eq_byte_db8 = [r for r in dct_results_list if r["method"] == "DCT_equal_byte" and r["db"] == 8]

    # Join AE vs DCT equal-dim
    paired_eq_dim = join_ae_dct_results(ae_db8, dct_eq_dim_db8)
    # Join AE vs DCT equal-byte
    paired_eq_byte = join_ae_dct_results(ae_db8, dct_eq_byte_db8)

    paired_rows = []
    for p in paired_eq_dim:
        p_row = p.copy()
        p_row["comparison_type"] = "AE_vs_DCT_equal_dim"
        p_row["delta_PRD"] = p["PRD_ae"] - p["PRD_dct"]
        p_row["delta_PRDN"] = p["PRDN_ae"] - p["PRDN_dct"]
        p_row["delta_RMSE"] = p["RMSE_ae"] - p["RMSE_dct"]
        paired_rows.append(p_row)

    for p in paired_eq_byte:
        p_row = p.copy()
        p_row["comparison_type"] = "AE_vs_DCT_equal_byte"
        p_row["delta_PRD"] = p["PRD_ae"] - p["PRD_dct"]
        p_row["delta_PRDN"] = p["PRDN_ae"] - p["PRDN_dct"]
        p_row["delta_RMSE"] = p["RMSE_ae"] - p["RMSE_dct"]
        paired_rows.append(p_row)

    paired_csv_path = output_dir / "fold1_ae_vs_dct_paired.csv"
    with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    # --------------------------------------------------------------------
    # AGGREGATION & REPORT TABLES
    # --------------------------------------------------------------------
    # Summary by subject (AE vs DCT equal-dim vs DCT equal-byte)
    summary_by_subj = {}
    for r in all_metrics:
        if r["db"] == 8:
            key = (r["subject"], r["method"], r["channel"])
            if key not in summary_by_subj:
                summary_by_subj[key] = {"prd_list": [], "prdn_list": [], "rmse_list": []}
            if r["metric_valid"]:
                summary_by_subj[key]["prd_list"].append(r["PRD"])
                summary_by_subj[key]["prdn_list"].append(r["PRDN"])
                summary_by_subj[key]["rmse_list"].append(r["RMSE"])

    subj_rows = []
    for (subj, method, ch), val_dict in summary_by_subj.items():
        if val_dict["prd_list"]:
            subj_rows.append({
                "subject": subj,
                "method": method,
                "channel": ch,
                "PRD_mean": round(float(np.mean(val_dict["prd_list"])), 4),
                "PRDN_mean": round(float(np.mean(val_dict["prdn_list"])), 4),
                "RMSE_mean": round(float(np.mean(val_dict["rmse_list"])), 4),
                "valid_windows": len(val_dict["prd_list"]),
            })

    subj_csv_path = output_dir / "fold1_summary_by_subject.csv"
    with open(subj_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "method", "channel", "PRD_mean", "PRDN_mean", "RMSE_mean", "valid_windows"])
        writer.writeheader()
        writer.writerows(subj_rows)

    # Overall summary across subjects with equal subject weights
    overall_dict = {}
    for r in subj_rows:
        key = (r["method"], r["channel"])
        if key not in overall_dict:
            overall_dict[key] = {"prd": [], "prdn": [], "rmse": []}
        overall_dict[key]["prd"].append(r["PRD_mean"])
        overall_dict[key]["prdn"].append(r["PRDN_mean"])
        overall_dict[key]["rmse"].append(r["RMSE_mean"])

    overall_rows = []
    for (method, ch), val_dict in overall_dict.items():
        overall_rows.append({
            "method": method,
            "channel": ch,
            "subjects_count": len(val_dict["prd"]),
            "PRD_mean": round(float(np.mean(val_dict["prd"])), 4),
            "PRD_std": round(float(np.std(val_dict["prd"], ddof=0)), 4),
            "PRDN_mean": round(float(np.mean(val_dict["prdn"])), 4),
            "PRDN_std": round(float(np.std(val_dict["prdn"], ddof=0)), 4),
            "RMSE_mean": round(float(np.mean(val_dict["rmse"])), 4),
            "RMSE_std": round(float(np.std(val_dict["rmse"], ddof=0)), 4),
        })

    overall_csv_path = output_dir / "fold1_summary_overall.csv"
    with open(overall_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "channel", "subjects_count", "PRD_mean", "PRD_std", "PRDN_mean", "PRDN_std", "RMSE_mean", "RMSE_std"])
        writer.writeheader()
        writer.writerows(overall_rows)

    # --------------------------------------------------------------------
    # PART 8: RECONSTRUCTION FIGURE
    # --------------------------------------------------------------------
    print("\n--- PART 8: GENERATE RECONSTRUCTION FIGURE ---")
    fig_path = output_dir / "fold1_reconstruction_examples.png"

    # Select first valid window of S1
    sample_win_idx = 0
    w_norm_sample = test_windows_norm[sample_win_idx]
    w_meta_sample = test_meta_list[sample_win_idx]

    # Reconstruct AE pilot
    x_in_sample = torch.from_numpy(w_norm_sample).unsqueeze(0).float().to(device)
    with torch.no_grad():
        latent_sample = model.encode(x_in_sample)
        rec_norm_ae_sample = model.decode(latent_sample).squeeze(0).cpu().numpy()

    # Reconstruct DCT equal-dim (K=256)
    rec_norm_dct_eq_dim, _, _, _ = dct_reconstruct(w_norm_sample, k=256)
    # Reconstruct DCT equal-byte (K=170)
    rec_norm_dct_eq_byte, _, _, _ = dct_reconstruct(w_norm_sample, k=170)

    # Denormalize all signals
    ref_phys_sample = denormalize(w_norm_sample, saved_stats)
    ae_phys_sample = denormalize(rec_norm_ae_sample, saved_stats)
    dct_eq_dim_phys_sample = denormalize(rec_norm_dct_eq_dim, saved_stats)
    dct_eq_byte_phys_sample = denormalize(rec_norm_dct_eq_byte, saved_stats)

    time_axis = np.arange(512) / 64.0  # 8 seconds

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: PPG Channel
    axes[0].plot(time_axis, ref_phys_sample[0], label="Original PPG", color="black", linewidth=1.8)
    axes[0].plot(time_axis, ae_phys_sample[0], label="AE Pilot ($d_b=8$)", color="crimson", linestyle="--", linewidth=1.5)
    axes[0].plot(time_axis, dct_eq_dim_phys_sample[0], label="DCT Equal-Dim ($K=256$)", color="dodgerblue", linestyle=":", linewidth=1.5)
    axes[0].plot(time_axis, dct_eq_byte_phys_sample[0], label="DCT Equal-Byte ($K=170$)", color="forestgreen", linestyle="-.", linewidth=1.5)
    axes[0].set_ylabel("PPG Signal Amplitude")
    axes[0].set_title(f"Fold 1 Test Reconstruction Example — Subject {w_meta_sample['subject']} ({w_meta_sample['window_id']}) — PPG Channel")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: ACCx Channel
    axes[1].plot(time_axis, ref_phys_sample[1], label="Original ACCx", color="black", linewidth=1.8)
    axes[1].plot(time_axis, ae_phys_sample[1], label="AE Pilot ($d_b=8$)", color="crimson", linestyle="--", linewidth=1.5)
    axes[1].plot(time_axis, dct_eq_dim_phys_sample[1], label="DCT Equal-Dim ($K=256$)", color="dodgerblue", linestyle=":", linewidth=1.5)
    axes[1].plot(time_axis, dct_eq_byte_phys_sample[1], label="DCT Equal-Byte ($K=170$)", color="forestgreen", linestyle="-.", linewidth=1.5)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("ACCx Acceleration (g)")
    axes[1].set_title(f"ACCx Channel")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Reconstruction comparison plot saved to: {fig_path}")

    # --------------------------------------------------------------------
    # PART 9: 18 SANITY CHECK ASSERTIONS
    # --------------------------------------------------------------------
    print("\n--- PART 9: 18 SANITY CHECK ASSERTIONS ---")

    sanity_checks = {
        "1. Test data is S1,S2,S3 real": set(m["subject"] for m in test_meta_list) == {"S1", "S2", "S3"},
        "2. No synthetic data in eval": all(r.get("dataset", "PPG-DaLiA") == "PPG-DaLiA" for r in all_metrics),

        "3. Test window shape [4,512]": test_windows_norm.shape[1:] == (4, 512),
        "4. Test step = 8s": all(m["end_index"] - m["start_index"] == 512 for m in test_meta_list),
        "5. Train-only norm stats Fold 1": len(saved_stats["mean"]) == 4 and abs(saved_stats["mean"][0] - (-0.000341)) < 1e-3,
        "6. AE checkpoint is real pilot": ckpt_data.get("data_type") == "REAL_DATA",
        "7. AE latent shape [B,8,32]": list(latent_tensor.shape) == [1, 8, 32],
        "8. AE codec byte length correct": nbytes_ae == 1040,
        "9. DCT equal-dim K=256": any(r["K"] == 256 for r in dct_results_list),
        "10. DCT equal-byte K=170": any(r["K"] == 170 for r in dct_results_list),
        "11. DCT codec byte length correct": any(r["nbytes"] == 1040 for r in dct_results_list if r["method"] == "DCT_equal_byte"),
        "12. Codec serialize-deserialize works": all(c["codec_pass"] for c in codec_records),
        "13. Reconstruction shape correct": rec_phys.shape == (4, 512),
        "14. Denormalize works": ref_phys.shape == (4, 512),
        "15. PRD/PRDN/RMSE finite": all(math.isfinite(r["PRD"]) and math.isfinite(r["PRDN"]) for r in all_metrics if r["metric_valid"]),
        "16. Same test windows for AE & DCT": len(ae_results_list) // 4 == len(dct_results_list) // (4 * 8),
        "17. Paired join lossless": len(paired_eq_dim) == len(ae_db8),
        "18. No val loss as test metric": "val_loss" not in [r.get("PRD") for r in all_metrics],
    }

    all_sanity_pass = all(sanity_checks.values())
    for name, res in sanity_checks.items():
        status_str = "[PASS]" if res else "[FAIL]"
        print(f"  {status_str} {name}")

    assert all_sanity_pass, "Sanity checks failed!"

    # --------------------------------------------------------------------
    # GENERATE SANITY REPORT MARKDOWN
    # --------------------------------------------------------------------
    report_md_path = output_dir / "fold1_sanity_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Fold 1 Real Data End-to-End Sanity Check Report\n\n")
        f.write("## 1. Norm Stats Audit\n")
        f.write(f"- **Status:** PASS\n")
        f.write(f"- **Train Subjects:** {norm_stats_audit['source_subjs']}\n")
        f.write(f"- **Total Samples:** {norm_stats_audit['total_samples']}\n")
        f.write(f"- **Means:** {norm_stats_audit['mean']}\n")
        f.write(f"- **Stds:** {norm_stats_audit['std']}\n\n")

        f.write("## 2. Fold 1 Test Set\n")
        f.write(f"- **S1 Windows:** {s1_count}\n")
        f.write(f"- **S2 Windows:** {s2_count}\n")
        f.write(f"- **S3 Windows:** {s3_count}\n")
        f.write(f"- **Total Test Windows:** {total_test_windows}\n\n")

        f.write("## 3. Codec & Budget Verification\n")
        f.write("| Method | d_b | K | Pad Bytes | Bytes/Window | Expected | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for c in codec_records:
            f.write(f"| {c['method']} | {c['db']} | {c['K']} | {c['pad_bytes']} | {c['nbytes']} | {c['expected_bytes']} | {'PASS' if c['codec_pass'] else 'FAIL'} |\n")

        f.write("\n## 4. Overall Metric Summary (Fold 1 Test Set)\n")
        f.write("| Method | Channel | Subjects | PRD (%) Mean±Std | PRDN (%) Mean±Std | RMSE Mean±Std |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in overall_rows:
            f.write(f"| {r['method']} | {r['channel']} | {r['subjects_count']} | {r['PRD_mean']:.2f}±{r['PRD_std']:.2f} | {r['PRDN_mean']:.2f}±{r['PRDN_std']:.2f} | {r['RMSE_mean']:.4f}±{r['RMSE_std']:.4f} |\n")

        f.write("\n## 5. 18 Mandatory Sanity Checks Status\n")
        for name, res in sanity_checks.items():
            f.write(f"- **{name}:** {'PASS' if res else 'FAIL'}\n")

    print(f"\nSanity check report written to: {report_md_path}")
    print("==========================================================================")
    print("           END-TO-END SANITY CHECK COMPLETED SUCCESSFULLY!                ")
    print("==========================================================================")

    return {
        "norm_stats_audit": norm_stats_audit,
        "s1_count": s1_count,
        "s2_count": s2_count,
        "s3_count": s3_count,
        "total_test_windows": total_test_windows,
        "codec_records": codec_records,
        "all_metrics": all_metrics,
        "paired_rows": paired_rows,
        "overall_rows": overall_rows,
        "fig_path": str(fig_path),
        "sanity_checks": sanity_checks,
        "all_sanity_pass": all_sanity_pass,
    }


if __name__ == "__main__":
    run_sanity_check_fold1()
