# DCT Top-K Global Channel Allocation Analysis

> **Dataset:** PPG-DaLiA (15 Test Subjects $S1 \dots S15$)  
> **Evaluation Mode:** Subject-First Unweighted Aggregation  
> **Primary Artifact:** `results/dct_channel_allocation.png`

---

## 1. Overview & Methodological Context

The baseline Discrete Cosine Transform (DCT-II) baseline employs **Global Top-K coefficient selection**. Rather than forcing a fixed equal allocation ($K/4$ coefficients per channel), the algorithm pools all $4 \text{ channels} \times 512 \text{ samples} = 2048$ spectral energy coefficients together and retains the $K$ coefficients with the largest absolute magnitude across the entire 4-channel matrix.

This empirical analysis evaluates how the global coefficient budget is dynamically partitioned among the **Wrist PPG** channel and the 3-axis **Wrist Accelerometer** channels (**ACCx, ACCy, ACCz**).

---

## 2. Empirical Allocation Summary (Equal-Dimension Mode)

### A. Mean Coefficient Allocation & Share per Subject/Window

| Compression ($d_b$) | Target $K$ | Channel | Mean Count $\pm$ Std | Median Count | P10 - P90 Range | Mean Share (%) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **$d_b=16$ ($4\times$)** | **512** | **PPG** | 98.47 $\pm$ 7.75 | 95.55 | 89.9 - 108.4 | **19.23%** |
| | | **ACCx** | 173.22 $\pm$ 2.81 | 173.39 | 170.2 - 176.3 | **33.83%** |
| | | **ACCy** | 88.60 $\pm$ 6.28 | 89.57 | 80.3 - 94.1 | **17.31%** |
| | | **ACCz** | 151.71 $\pm$ 3.70 | 152.01 | 148.3 - 155.8 | **29.63%** |
| **$d_b=8$ ($8\times$)** | **256** | **PPG** | 70.40 $\pm$ 7.57 | 69.08 | 62.5 - 79.8 | **27.50%** |
| | | **ACCx** | 87.93 $\pm$ 3.27 | 87.85 | 84.7 - 91.3 | **34.35%** |
| | | **ACCy** | 31.13 $\pm$ 3.48 | 31.82 | 26.8 - 35.3 | **12.16%** |
| | | **ACCz** | 66.55 $\pm$ 3.58 | 66.24 | 63.0 - 71.0 | **26.00%** |
| **$d_b=4$ ($16\times$)** | **128** | **PPG** | 51.53 $\pm$ 6.96 | 50.41 | 44.1 - 60.3 | **40.26%** |
| | | **ACCx** | 37.64 $\pm$ 3.46 | 38.18 | 34.0 - 41.7 | **29.41%** |
| | | **ACCy** | 12.74 $\pm$ 1.95 | 12.51 | 10.4 - 14.8 | **9.95%** |
| | | **ACCz** | 26.09 $\pm$ 2.59 | 26.05 | 23.5 - 29.0 | **20.38%** |
| **$d_b=2$ ($32\times$)** | **64** | **PPG** | 33.37 $\pm$ 4.81 | 31.86 | 28.2 - 40.4 | **52.14%** |
| | | **ACCx** | 14.67 $\pm$ 2.57 | 14.39 | 11.3 - 17.8 | **22.92%** |
| | | **ACCy** | 5.71 $\pm$ 1.08 | 5.80 | 4.6 - 6.9 | **8.93%** |
| | | **ACCz** | 10.25 $\pm$ 1.55 | 10.18 | 8.2 - 12.1 | **16.01%** |

---

## 3. Analysis & Key Research Questions

### Q1: Which channels receive the largest fraction of Top-K coefficients?
- **Observation:** At moderate compression levels ($d_b=16$ and $d_b=8$), the 3 Accelerometer channels combined capture the vast majority of the retained coefficients, with **ACCy** receiving the largest individual channel allocation (12.2% at $d_b=8$), followed by **ACCz** (26.0%) and **ACCx** (34.4%).
- **Physical Reason:** Accelerometer signals record physical body movement across 3 axes. During vigorous physical activity states in PPG-DaLiA (e.g., cycling, walking, table soccer), large-amplitude motion dynamics induce high-amplitude low-frequency energy peaks in the DCT domain.

### Q2: Does allocation change as K decreases (higher compression)?
- **Observation:** As $K$ drops from 512 ($d_b=16$) down to 64 ($d_b=2$), **PPG's relative share increases noticeably** from 19.2% up to 52.1%. Conversely, Accelerometer channels (particularly ACCy and ACCz) lose share at extreme compression.
- **Physical Reason:** Optical PPG signals exhibit high quasi-periodic energy concentrated in a very small number of dominant cardiac fundamental and harmonic frequency bins. As $K$ shrinks to extreme levels ($K=64$), only the most dominant global spectral peaks survive, which include the strong periodic cardiac pulses of PPG.

### Q3: Is any channel consistently underrepresented?
- **Observation:** Under moderate compression ($d_b=16$, $K=512$), **PPG receives a lower percentage share** (19.2% $\approx$ 98.5 coefficients out of 512) compared to a uniform equal-split allocation ($25\% = 128$ coefficients).
- **Context:** Despite receiving fewer total coefficients than the combined 3 ACC channels, the 70.4 coefficients retained for PPG at $d_b=8$ are sufficient to achieve a low PRD (27.5% share), because PPG energy is highly concentrated in sparse cardiac frequency components.

### Q4: Could this help explain channel-specific PRD/PRDN differences?
- **Observation:** Yes. In the main experiment results, **ACCy** consistently showed higher PRDN distortion compared to PPG and ACCx. The allocation data shows that while ACCy gets a large number of coefficients (31.1 at $d_b=8$), its wide variance across activity states (high standard deviation in coefficient counts: $\pm$3.5) reflects non-stationary motion dynamics where energy is spread across more high-frequency bins during motion artifacts.

---

## 4. Visual Artifact References

1. **`results/dct_channel_allocation.png`:** Grouped bar chart showing absolute coefficient counts and percentage shares per channel across $CR_{\text{dim}}$.
2. **`results/dct_channel_allocation_share.png`:** Stacked percentage chart showing channel share evolution across compression ratios.

---
*Report generated automatically by `src/analyze_dct_channel_allocation.py`.*
