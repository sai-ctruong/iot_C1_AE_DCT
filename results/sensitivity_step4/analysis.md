# Test Window Step Sensitivity Analysis Report

> **Methodology:** Optional Sensitivity Benchmark  
> **Main Test Setting (Official):** 8-second window, step = 8 seconds (0% overlap, non-overlapping)  
> **Sensitivity Test Setting:** 8-second window, step = 4 seconds (50% overlap)  
> **Dataset:** PPG-DaLiA (15 Test Subjects $S1 \dots S15$)  
> **Primary Artifact:** `results/sensitivity_step4/step_sensitivity.png`

---

## 1. Important Scientific & Methodological Note

> [!IMPORTANT]
> **Overlapping Test Window Scoping Constraint:**  
> The 4-second step Test windows contain 50% overlapping temporal segments. Consequently, individual step-4 windows are **NOT statistically independent observations**. This sensitivity analysis is conducted purely to verify whether temporal window alignment influences subject-level distortion metrics. **The official scientific benchmark results of this study remain strictly defined by the non-overlapping 8-second step Test configuration.**

---

## 2. Quantitative Comparison: Step 8s vs Step 4s (PPG Channel)

| Method | Mode | Bottleneck $d_b$ | Metric | Step 8s (Main) | Step 4s (Sensitivity) | Abs Diff | Rel Diff (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AE** | `equal_dim` | $d_b=16$ | PRD | 8.13% | 8.13% | -0.00% | -0.00% |
| **AE** | `equal_byte` | $d_b=16$ | PRD | 8.13% | 8.13% | -0.00% | -0.00% |
| **DCT** | `equal_dim` | $d_b=16$ | PRD | 3.25% | 3.26% | +0.00% | +0.09% |
| **DCT** | `equal_byte` | $d_b=16$ | PRD | 5.82% | 5.82% | -0.00% | -0.03% |
| **AE** | `equal_dim` | $d_b=8$ | PRD | 13.29% | 13.29% | -0.00% | -0.02% |
| **AE** | `equal_byte` | $d_b=8$ | PRD | 13.29% | 13.29% | -0.00% | -0.02% |
| **DCT** | `equal_dim` | $d_b=8$ | PRD | 7.97% | 7.96% | -0.01% | -0.08% |
| **DCT** | `equal_byte` | $d_b=8$ | PRD | 11.57% | 11.57% | -0.00% | -0.00% |
| **AE** | `equal_dim` | $d_b=4$ | PRD | 19.88% | 19.88% | +0.00% | +0.00% |
| **AE** | `equal_byte` | $d_b=4$ | PRD | 19.88% | 19.88% | +0.00% | +0.00% |
| **DCT** | `equal_dim` | $d_b=4$ | PRD | 14.51% | 14.51% | +0.01% | +0.05% |
| **DCT** | `equal_byte` | $d_b=4$ | PRD | 19.63% | 19.62% | -0.01% | -0.05% |
| **AE** | `equal_dim` | $d_b=2$ | PRD | 45.64% | 45.66% | +0.02% | +0.05% |
| **AE** | `equal_byte` | $d_b=2$ | PRD | 45.64% | 45.66% | +0.02% | +0.05% |
| **DCT** | `equal_dim` | $d_b=2$ | PRD | 24.00% | 24.00% | -0.00% | -0.01% |
| **DCT** | `equal_byte` | $d_b=2$ | PRD | 31.97% | 31.97% | -0.00% | -0.01% |

---

## 3. Key Findings & Research Questions

### A. Does overlap change absolute distortion metrics?
- **Observation:** Transitioning from non-overlapping 8-second steps to 50% overlapping 4-second steps produces **negligible changes in subject-level mean PRD distortion** (less than $\pm 0.5\%$ absolute variation across all compression levels).
- **Explanation:** Because metrics are aggregated subject-first (window $\to$ subject mean $\to$ 15-subject unweighted mean), 50% overlap increases the window sample density but preserves the stationary time-domain error distribution per subject.

### B. Does the AE vs DCT relative ranking change?
- **Observation:** **No.** The comparative ranking between the 1D-CNN Autoencoder and the DCT-II Top-K baseline remains 100% identical under the 4-second Test stride.
- **Empirical Evidence:** DCT Top-K continues to outperform AE across all compression levels ($d_b \in \{16, 8, 4, 2\}$) under both equal-dimension and equal-byte budget constraints.

### C. Are study conclusions robust to Test stride selection?
- **Observation:** Yes. The scientific conclusions of this study are **highly robust to Test window stride selection**. Evaluating on overlapping windows confirms that the performance superiority of frequency-domain sparse coding (DCT) on quasi-periodic wearable signals is an intrinsic signal property, not an artifact of window boundary placement.

---

## 4. Artifact References

- Detailed step-4 evaluation rows: `results/sensitivity_step4/results_step4.csv.gz`
- Direct step-4 vs step-8 comparison table: `results/sensitivity_step4/step4_vs_step8.csv`
- Comparative visualization figure: `results/sensitivity_step4/step_sensitivity.png`

---
*Report generated automatically by `src/run_sensitivity_step4.py`.*
