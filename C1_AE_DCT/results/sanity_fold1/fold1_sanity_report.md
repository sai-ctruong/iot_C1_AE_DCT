# Fold 1 Real Data End-to-End Sanity Check Report

## 1. Norm Stats Audit
- **Status:** PASS
- **Train Subjects:** ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15']
- **Total Samples:** 5432064
- **Means:** [-0.000341, -0.543489, 0.09613, 0.311879]
- **Stds:** [86.878957, 0.360015, 0.645617, 0.403271]

## 2. Fold 1 Test Set
- **S1 Windows:** 1151
- **S2 Windows:** 1025
- **S3 Windows:** 1092
- **Total Test Windows:** 3268

## 3. Codec & Budget Verification
| Method | d_b | K | Pad Bytes | Bytes/Window | Expected | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DCT_equal_dim | 16 | 512 | 0 | 3088 | 3088 | PASS |
| DCT_equal_byte | 16 | 341 | 2 | 2064 | 2064 | PASS |
| DCT_equal_dim | 8 | 256 | 0 | 1552 | 1552 | PASS |
| DCT_equal_byte | 8 | 170 | 4 | 1040 | 1040 | PASS |
| DCT_equal_dim | 4 | 128 | 0 | 784 | 784 | PASS |
| DCT_equal_byte | 4 | 85 | 2 | 528 | 528 | PASS |
| DCT_equal_dim | 2 | 64 | 0 | 400 | 400 | PASS |
| DCT_equal_byte | 2 | 42 | 4 | 272 | 272 | PASS |
| AE | 8 | 0 | 0 | 1040 | 1040 | PASS |

## 4. Overall Metric Summary (Fold 1 Test Set)
| Method | Channel | Subjects | PRD (%) Mean±Std | PRDN (%) Mean±Std | RMSE Mean±Std |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AE | PPG | 3 | 17.09±4.67 | 17.11±4.68 | 8.3835±1.4226 |
| AE | ACCx | 3 | 15.52±1.38 | 120.02±18.21 | 0.0712±0.0020 |
| AE | ACCy | 3 | 27.14±2.25 | 2086.58±1644.76 | 0.0896±0.0032 |
| AE | ACCz | 3 | 14.68±0.95 | 159.16±69.64 | 0.0689±0.0015 |
| DCT_equal_dim | PPG | 3 | 8.91±3.36 | 8.92±3.37 | 4.1416±0.2211 |
| DCT_equal_dim | ACCx | 3 | 6.34±0.73 | 22.95±0.91 | 0.0296±0.0013 |
| DCT_equal_dim | ACCy | 3 | 12.11±0.52 | 41.93±1.10 | 0.0459±0.0015 |
| DCT_equal_dim | ACCz | 3 | 7.30±0.41 | 27.70±1.12 | 0.0334±0.0013 |
| DCT_equal_byte | PPG | 3 | 12.76±4.72 | 12.78±4.72 | 6.0145±0.4062 |
| DCT_equal_byte | ACCx | 3 | 9.21±0.93 | 33.20±1.70 | 0.0431±0.0021 |
| DCT_equal_byte | ACCy | 3 | 15.12±0.54 | 50.63±1.23 | 0.0593±0.0023 |
| DCT_equal_byte | ACCz | 3 | 10.33±0.57 | 38.32±1.53 | 0.0470±0.0022 |

## 5. 18 Mandatory Sanity Checks Status
- **1. Test data is S1,S2,S3 real:** PASS
- **2. No synthetic data in eval:** PASS
- **3. Test window shape [4,512]:** PASS
- **4. Test step = 8s:** PASS
- **5. Train-only norm stats Fold 1:** PASS
- **6. AE checkpoint is real pilot:** PASS
- **7. AE latent shape [B,8,32]:** PASS
- **8. AE codec byte length correct:** PASS
- **9. DCT equal-dim K=256:** PASS
- **10. DCT equal-byte K=170:** PASS
- **11. DCT codec byte length correct:** PASS
- **12. Codec serialize-deserialize works:** PASS
- **13. Reconstruction shape correct:** PASS
- **14. Denormalize works:** PASS
- **15. PRD/PRDN/RMSE finite:** PASS
- **16. Same test windows for AE & DCT:** PASS
- **17. Paired join lossless:** PASS
- **18. No val loss as test metric:** PASS
