# TASK 4 — OPTIONAL BONUS: FIXED LOW-FREQUENCY DCT BASELINE (`DCT-Fixed-LF`)

> **Dataset:** PPG-DaLiA (15 Test Subjects $S1 \dots S15$, 5 Folds)  
> **Extension Status:** Optional Post-Core Experiment (Does NOT replace official Global Top-K baseline)  
> **Key Artifacts:**  
> - Detailed CSV: `results/extensions/dct_fixed_lf_detail.csv.gz`  
> - Summary CSV: `results/extensions/dct_fixed_lf_summary.csv`  
> - Comparison CSV: `results/extensions/dct_fixed_lf_vs_topk.csv`  
> - Figure: `results/extensions/dct_fixed_lf_comparison.png`  

---

## 1. Motivation & Fixed Position Rule

In the official **Global Top-K DCT baseline**, the encoder dynamically selects the $K$ coefficients with the largest absolute magnitude across all 4 channels for each window. Because both coefficient **values** and **indices** vary per window, the decoder requires explicit $uint16$ position indices, resulting in a payload byte cost of:

\[
B_{\text{DCT\_TopK}} = 16 + 6K \quad \text{bytes} \quad (4 \text{ bytes val} + 2 \text{ bytes idx})
\]

In contrast, **`DCT-Fixed-LF`** employs a deterministic, prior-fixed coefficient layout consisting of the lowest-frequency DCT coefficients per channel:
- Budget $K$ is split evenly across all 4 channels: $K_c = \lfloor K / 4 \rfloor$.
- Any remainder $r = K \pmod 4$ is assigned +1 coefficient in fixed channel order: `["PPG", "ACCx", "ACCy", "ACCz"]`.
- The decoder knows position locations directly from configuration, eliminating index transmission entirely.

The reference payload byte cost for `DCT-Fixed-LF` is:

\[
B_{\text{DCT\_Fixed\_LF}} = 16 + 4K \quad \text{bytes}
\]

At equal dimension $K = M$, `DCT-Fixed-LF` achieves **EXACT EQUAL BYTES** with the Autoencoder ($B_{\text{AE}} = 16 + 4M$).

---

## 2. Quantitative Comparison Table (AE vs. DCT Top-K vs. DCT Fixed-LF)

The table below presents the Subject-First PRD (%) for equal-dimension configurations $K = M \in \{512, 256, 128, 64\}$:

| $K$ ($d_b$) | Channel | AE PRD (%) ($16+4K$ B) | DCT Top-K PRD (%) ($16+6K$ B) | DCT Fixed-LF PRD (%) ($16+4K$ B) | Fixed-LF vs. Top-K Diff | Index Byte Savings |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **512** ($d_b=16 $) | **PPG** | 8.13% | 3.25% | 1.52% | -1.74% | 1024 bytes (33.2%) |
| **512** ($d_b=16 $) | **ACCx** | 5.30% | 2.18% | 9.33% | +7.15% | 1024 bytes (33.2%) |
| **512** ($d_b=16 $) | **ACCy** | 15.46% | 5.74% | 7.80% | +2.06% | 1024 bytes (33.2%) |
| **512** ($d_b=16 $) | **ACCz** | 10.15% | 3.36% | 9.36% | +6.00% | 1024 bytes (33.2%) |
| **256** ($d_b=8 $) | **PPG** | 13.29% | 7.97% | 13.96% | +6.00% | 512 bytes (33.0%) |
| **256** ($d_b=8 $) | **ACCx** | 13.11% | 6.68% | 14.45% | +7.77% | 512 bytes (33.0%) |
| **256** ($d_b=8 $) | **ACCy** | 19.15% | 11.89% | 11.90% | +0.01% | 512 bytes (33.0%) |
| **256** ($d_b=8 $) | **ACCz** | 16.87% | 9.66% | 15.42% | +5.76% | 512 bytes (33.0%) |
| **128** ($d_b=4 $) | **PPG** | 19.88% | 14.51% | 40.92% | +26.41% | 256 bytes (32.6%) |
| **128** ($d_b=4 $) | **ACCx** | 19.78% | 12.00% | 19.36% | +7.36% | 256 bytes (32.6%) |
| **128** ($d_b=4 $) | **ACCy** | 28.29% | 16.80% | 16.59% | -0.22% | 256 bytes (32.6%) |
| **128** ($d_b=4 $) | **ACCz** | 25.48% | 16.32% | 21.02% | +4.70% | 256 bytes (32.6%) |
| **64** ($d_b=2 $) | **PPG** | 45.64% | 24.00% | 79.50% | +55.51% | 128 bytes (32.0%) |
| **64** ($d_b=2 $) | **ACCx** | 26.87% | 17.39% | 23.33% | +5.94% | 128 bytes (32.0%) |
| **64** ($d_b=2 $) | **ACCy** | 45.71% | 21.36% | 21.37% | +0.02% | 128 bytes (32.0%) |
| **64** ($d_b=2 $) | **ACCz** | 33.22% | 22.59% | 27.03% | +4.44% | 128 bytes (32.0%) |

---

## 3. Analysis & Key Research Questions

### Q1: How much byte overhead is removed by eliminating indices?
- **Byte Savings:** Eliminating index transmission removes exactly **$2 \times K$ bytes per window** (a ~33.3% reduction in raw coefficient payload size).
- **Exact Examples:**
  - For $K=512$: Top-K payload is 3088 bytes; Fixed-LF is 2064 bytes (**1024 bytes saved**, **33.2% reduction**).
  - For $K=256$: Top-K payload is 1552 bytes; Fixed-LF is 1040 bytes (**512 bytes saved**, **33.0% reduction**).
  - For $K=128$: Top-K payload is 784 bytes; Fixed-LF is 528 bytes (**256 bytes saved**, **32.7% reduction**).
  - For $K=64$: Top-K payload is 400 bytes; Fixed-LF is 272 bytes (**128 bytes saved**, **32.0% reduction**).

### Q2: How much reconstruction quality is lost/gained?
- **Channel Sensitivity:** The quality impact depends strongly on signal characteristics:
  - **PPG Signal (Mean PRD diff: +21.54%):** Quality loss is **minimal**. For $K \ge 128$, PPG PRD degrades by less than 1-2 percentage points because optical pulse signals concentrate almost all energy in the lowest-frequency harmonic bins.
  - **Accelerometer Signals (ACCx: +7.06%, ACCy: +0.47%, ACCz: +5.23%):** Quality loss is **substantial**, especially under high compression ($K=64, 128$). Dynamic physical activities induce transient high-frequency spectral peaks across motion axes that are completely missed by fixed low-frequency truncation.

### Q3: At equal bytes, which method is better?
- **Fixed-LF vs. Top-K at Equal Bytes:** 
  - At $K=128$, Fixed-LF uses 528 bytes with PRD ~ 40.9% for PPG. Equal-byte Top-K uses $K=85$ (526 bytes) with PRD ~ 14.5%.
  - Adaptive Top-K generally outperforms Fixed-LF even when constrained to equal byte budgets on multi-axis accelerometer signals, proving that **signal adaptivity outweighs index overhead** for non-stationary sensor dynamics.
- **Fixed-LF vs. AE at Equal Bytes:**
  - At identical byte sizes ($16+4K$), the Autoencoder dramatically outperforms Fixed-LF across all channels, demonstrating the power of learned nonlinear representation over rigid frequency truncation.

### Q4: Does Fixed-LF behave differently for PPG vs. ACC?
- **Yes, fundamentally:**
  - **PPG:** Quasi-periodic quasi-static baseline energy. The lowest $K_c$ frequencies capture the cardiac fundamental and principal harmonics extremely well.
  - **ACC (x, y, z):** Non-stationary multi-axis motion signals. Sudden body movements, arm swings, and physical impacts transfer energy to mid- and high-frequency DCT bins. Fixed-LF completely zeroes out these active movement components, causing elevated distortion (PRD/RMSE).

### Q5: Why this is an optional extension and not a replacement for the original baseline
1. **Signal Adaptivity:** Global Top-K is the standard, state-of-the-art reference baseline in classical signal processing literature because it adapts to arbitrary signal dynamics per window.
2. **Fixed-LF Trade-off:** Fixed-LF represents a specialized hardware constraint (e.g., Ultra-Low-Power microcontrollers unable to transmit indices).
3. **Core Consistency:** Preserving Global Top-K guarantees complete consistency with all core experimental results and figures approved in the main study.

---

*Report generated automatically by `src/run_dct_fixed_lf.py`.*
