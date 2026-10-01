# 📋 Final Validation Audit Report — C1_AE_DCT Real Data Experiment Pipeline

**Date:** 2026-10-01  
**Dataset:** PPG-DaLiA (Real Wearable Sensor Signals)  
**Execution Environment:** CPU Execution Mode (PyTorch 2.x, Windows 11 64-bit)  

---

## 🟢 AUDIT CONDITIONS SUMMARY (14/14 PASSED)

| ID | Mandatory Condition | Status | Empirical Evidence / Detail |
| :---: | :--- | :---: | :--- |
| **1** | **15/15 Subjects Present** | 🟢 **PASS** | $S1 \dots S15$ all evaluated on their respective test folds without omissions or NaNs in metadata. |
| **2** | **5/5 Folds Present** | 🟢 **PASS** | Subject-wise 5-fold cross-validation scheme strictly applied ($S1..S3 \to \text{F1}$, $S4..S6 \to \text{F2}$, $S7..S9 \to \text{F3}$, $S10..S12 \to \text{F4}$, $S13..S15 \to \text{F5}$). |
| **3** | **20/20 Main Checkpoints Present** | 🟢 **PASS** | 5 Folds $\times$ 4 $d_b \in \{16, 8, 4, 2\}$ trained with seed 42, saved under `checkpoints/fold{1..5}_db{16,8,4,2}_seed42.pt`. |
| **4** | **15/15 Seed Checkpoints ($d_b=8$) Present** | 🟢 **PASS** | 5 Folds $\times$ 3 Seeds (42, 123, 999) for $d_b=8$ trained and verified, saved under `checkpoints/seed_experiments/`. |
| **5** | **4 Compression Levels Present** | 🟢 **PASS** | $d_b = 16 \implies M=512$, $d_b = 8 \implies M=256$, $d_b = 4 \implies M=128$, $d_b = 2 \implies M=64$. |
| **6** | **`equal_dim` Comparison Complete** | 🟢 **PASS** | DCT Top-K evaluated at $K=M$ for all 20 main runs across 15 subjects and 4 channels. |
| **7** | **`equal_byte` Comparison Complete** | 🟢 **PASS** | DCT Top-K evaluated at $K = \lfloor 4M / 6 \rfloor$ matching AE byte length ($B_{\text{DCT}} = B_{\text{AE}}$ including 16-byte header). |
| **8** | **`results/results.csv` Non-Empty** | 🟢 **PASS** | Contains 1,035,584 real detail evaluation rows (compressed as `results/results.csv.gz` - 28.5 MB). |
| **9** | **Zero Synthetic Data Rows** | 🟢 **PASS** | Audited: 0 synthetic rows in `results.csv` and `seed_results_detail.csv`. Synthetic fallbacks removed from production code. |
| **10** | **Zero Synthetic Figures / Artifacts** | 🟢 **PASS** | All generated PNG plots (`cr_dim_prd.png`, `rmse_curves.png`, `reconstruction_examples.png`, `failure_cases.png`, etc.) generated 100% from real test set metrics. |
| **11** | **All AE/DCT Pairs Matched** | 🟢 **PASS** | Pairwise delta $\delta_s = \text{AE} - \text{DCT}$ evaluated on identical (subject, channel, window_id, fold, comparison_type) tuples. |
| **12** | **$CR_{\text{dim}} \in \{4, 8, 16, 32\}$ for AE** | 🟢 **PASS** | Calculated relative to $RAW\_64HZ\_VALUES = 2048$ ($4 \text{ channels} \times 512 \text{ samples}$). AE $CR_{\text{dim}} = 2048 / M$. |
| **13** | **Correct $CR_{\text{byte}}$ Computation** | 🟢 **PASS** | $CR_{\text{byte\_64}} = 8192 / nbytes$, $CR_{\text{byte\_native}} = 6144 / nbytes$ using actual bitstream sizes from `C1Codec`. |
| **14** | **README Status Table & Architecture Accurate** | 🟢 **PASS** | README.md updated with 78/78 tests passed, 20 main + 15 seed runs completed, system workflow diagram, and CPU execution details. |

---

## 📊 EMPIRICAL EXPERIMENT SUMMARY HIGHLIGHTS

### 1. Main Compression Results ($d_b=8$, $CR_{\text{dim}}=8\times$, PPG Channel)
- **AE ($d_b=8$):** PRD = $13.29 \pm 3.61\%$, PRDN = $13.31 \pm 3.61\%$, RMSE = $8.29 \pm 2.58$
- **DCT Equal-Dim ($K=256$):** PRD = $7.78 \pm 2.37\%$, PRDN = $7.79 \pm 2.38\%$, RMSE = $4.76 \pm 1.48$
- **DCT Equal-Byte ($K=170$):** PRD = $11.08 \pm 3.20\%$, PRDN = $11.09 \pm 3.21\%$ RMSE = $6.83 \pm 2.13$

### 2. Seed Stability Results ($d_b=8$, PPG Channel across 15 Subjects)
- **Seed 42:** PRD = $13.29 \pm 3.61\%$, PRDN = $13.31 \pm 3.61\%$, RMSE = $8.29 \pm 2.58$
- **Seed 123:** PRD = $12.06 \pm 2.99\%$, PRDN = $12.07 \pm 2.99\%$, RMSE = $7.49 \pm 1.84$
- **Seed 999:** PRD = $12.13 \pm 2.80\%$, PRDN = $12.15 \pm 2.81\%$, RMSE = $7.55 \pm 1.78$
- **Overall Seed Consistency:** Mean PRD across seeds = $12.49 \pm 0.69\%$ (very stable initialization behavior across distinct random seeds).

---

## 🏁 CONCLUSION

All 14 scientific audit conditions are **FULLY SATISFIED** with 100% real PPG-DaLiA data. The repository is ready for submission and archiving.
