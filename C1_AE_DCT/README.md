# C1_AE_DCT — Signal Compression with Autoencoder & DCT Baseline

Project nghiên cứu và triển khai giải pháp n nén tín hiệu sinh học 4 kênh (Wrist PPG & Tri-axial Wrist ACC) từ tập dữ liệu thực nghiệm **PPG-DaLiA**. Dự án so sánh đối đầu giữa mô hình **1D-CNN Autoencoder (AE)** và thuật toán cơ sở **Discrete Cosine Transform (DCT)** theo quy trình 5-fold cross-validation độc lập theo đối tượng (Subject-Wise Split, $S1 \dots S15$).

---

## 📋 MỤC LỤC (TABLE OF CONTENTS)
1. [Trạng thái Thực nghiệm & Phân định Kết quả (Experiment Status)](#1-trạng-thái-thực-nghiệm--phân-định-kết-quả-experiment-status)
2. [Mục tiêu Đề tài (Project Objectives)](#2-mục-tiêu-đề-tài-project-objectives)
3. [Nguồn tải PPG-DaLiA (Dataset Download Source)](#3-nguồn-tải-ppg-dalia-dataset-download-source)
4. [Cấu trúc Thư mục (Directory Structure)](#4-cấu-trúc-thư-mục-directory-structure)
5. [Thiết lập Môi trường (Environment Setup)](#5-thiết-lập-môi-trường-environment-setup)
6. [Quy trình Preprocessing Thực nghiệm (Data Preprocessing Pipeline)](#6-quy-trình-preprocessing-thực-nghiệm-data-preprocessing-pipeline)
7. [Chạy Real Pilot Run (Real Data Verification)](#7-chạy-real-pilot-run-real-data-verification)
8. [Chạy 20 Main Runs (Main Experiments)](#8-chạy-20-main-runs-main-experiments)
9. [Chạy Seed Experiments (Seed Stability)](#9-chạy-seed-experiments-seed-stability)
10. [Đánh giá chỉ số Méo dạng (Evaluation & Metrics)](#10-đánh-giá-chỉ-số-méo-dạng-evaluation--metrics)
11. [Vị trí Artifacts & Phân loại Synthetic vs Real Data](#11-vị-trí-artifacts--phân-loại-synthetic-vs-real-data)

---

## 1. TRẠNG THÁI THỰC NGHIỆM & PHÂN ĐỊNH KẾT QUẢ (EXPERIMENT STATUS)

| Hạng mục | Trạng thái | Ghi chú / Chi tiết |
| :--- | :--- | :--- |
| **1. Unit Test / Code Logic** | ✅ **100% PASS** | 100% code logic, architecture, codec, metrics, loss & integration test suite pass. |
| **2. Preprocessing Data Pipeline** | ✅ **100% READY** | Đã đọc đủ 15 subject $S1 \dots S15$, polyphase resample ACC 32->64Hz, train-only Z-score norm stats 5 folds. |
| **3. Real Pilot Run** | 🔄 **TESTED / PENDING FINAL** | Đã verified trên Fold 1, $d_b=8$, seed 42 với dữ liệu thật PPG-DaLiA. |
| **4. Real Main Runs (20 runs)** | ⏸️ **READY FOR CONFIRMATION** | Pipeline nén 20 runs (5 Folds $\times 4 d_b$) đã khóa spec, sẵn sàng kích hoạt sau khi nghiệm thu Pilot. |
| **5. Seed Runs (10 runs)** | ⏸️ **READY FOR CONFIRMATION** | 10 runs bổ sung ($d_b=8$, seeds 42, 123, 999 trên cả 5 folds) sẵn sàng thực thi. |

> [!IMPORTANT]
> **Phân định Artifacts Synthetic vs Real Data:**
> Các file sinh ra từ thử nghiệm synthetic/smoke-test trước đây đã được phân loại và cô lập trong thư mục `synthetic_smoke_test/`. Các artifact synthetic này **chỉ dùng để test kỹ thuật**, KHÔNG phải kết quả khoa học. Kết quả khoa học chính thức sẽ sinh ra 100% từ tập dữ liệu thực nghiệm PPG-DaLiA.

---

## 2. MỤC TIÊU ĐỀ TÀI (PROJECT OBJECTIVES)

- **Bài toán:** Nén tín hiệu 4 kênh bao gồm 1 kênh Quang thể tích đồ Wrist PPG (BVP @ 64Hz) và 3 kênh Gia tốc kế Wrist ACC (ACCx, ACCy, ACCz @ 32Hz) từ thiết bị đeo thu thập trong các hoạt động phức tạp.
- **Kiến trúc Mô hình:**
  1. **1D-CNN Autoencoder (AE):** Conv1d (4->16->32->64) + Bottleneck Linear ($64 \to d_b$) + ConvTranspose1d ($d_b \to 64 \to 32 \to 16 \to 4$) với $d_b \in \{16, 8, 4, 2\}$.
  2. **Discrete Cosine Transform (DCT):** DCT-II Top-K baseline với ngân sách byte $B_{\text{DCT}} = B_{\text{AE}}$.
- **Đặc tả 5-Fold Subject-Wise:**
  - Fold 1: Train S6-S15, Val S4-S5, Test S1-S3
  - Fold 2: Train S1-S3, S9-S15, Val S7-S8, Test S4-S6
  - Fold 3: Train S1-S6, S12-S15, Val S10-S11, Test S7-S9
  - Fold 4: Train S1-S9, S15, Val S13-S14, Test S10-S12
  - Fold 5: Train S3-S12, Val S1-S2, Test S13-S15
- **Tiêu chuẩn Đánh giá:**
  - Đánh giá độc lập trên từng kênh (`PPG`, `ACCx`, `ACCy`, `ACCz`).
  - Mẫu số PRDN dùng năng lượng mẫu trừ mean; denominator $\le 1e-12$ được đánh dấu undefined (valid_prdn=False), không cộng epsilon tùy tiện.

---

## 3. NGUỒN TẢI PPG-DALIA (DATASET DOWNLOAD SOURCE)

Tập dữ liệu **PPG-DaLiA** công khai trên UCI Machine Learning Repository:
- **Link tải:** [UCI Machine Learning Repository — PPG-DaLiA Dataset](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
- **Định dạng:** 15 file pickle `S1.pkl` đến `S15.pkl`.
- Thư mục chứa dữ liệu trong project: `ppg+dalia/data/PPG_FieldStudy/` hoặc `C1_AE_DCT/data/raw/`.

---

## 4. CẤU TRÚC THƯ MỤC (DIRECTORY STRUCTURE)

```text
C1_AE_DCT/
├── configs/
│   ├── config.json             # File cấu hình chính (batch_size=128, max_epoch=100, lr=1e-3, patience=10)
│   ├── config.yaml             # File cấu hình dạng YAML (đồng bộ với config.json)
│   ├── folds.json              # File định nghĩa 5-fold subject split (S1-S15)
│   └── norm_stats_fold1..5.json# Thống kê Z-score tính CHỈ từ tập Train từng fold
├── data/
│   ├── raw/                    # Đường dẫn chứa S1.pkl .. S15.pkl
│   └── processed/              # Chứa dữ liệu đã resample & windowing (.npz) theo fold
├── src/
│   ├── dataset.py              # Loader & PyTorch Dataset class (C1Dataset) & Data Inventory
│   ├── resample.py             # Polyphase resample ACC 32Hz -> 64Hz (scipy.signal.resample_poly)
│   ├── windowing.py            # Cắt cửa sổ T=512 (8s), stride S=256 (Train/Val) & S=512 (Test)
│   ├── normalize.py            # Z-score normalization (Train-only statistics, ddof=0)
│   ├── model.py                # Kiến trúc 1D-CNN Autoencoder (C1Autoencoder)
│   ├── loss.py                 # Reconstruction MSE Loss
│   ├── baseline_dct.py         # Baseline DCT-II Top-K & equal byte/dim budget
│   ├── codec.py                # Quantization & Byte Bitstream Codec (C1Codec)
│   ├── metrics.py              # Tính CR_dim, CR_byte và ngân sách byte
│   ├── evaluate.py             # Tính PRD, PRDN, RMSE độc lập từng kênh & valid-mask
│   ├── results_schema.py       # Cấu trúc bảng kết quả & join AE/DCT
│   ├── aggregation.py          # Aggregate theo subject & tính delta_s = AE - DCT
│   ├── reporting.py            # Xuất đồ thị và bảng biểu tổng hợp
│   ├── reconstruction_visualization.py # Đồ thị dạng sóng & failure cases
│   ├── prepare_data.py         # End-to-end preprocessing pipeline
│   ├── train.py                # Core training loop với early stopping
│   ├── pilot_run.py            # Real Pilot Run (Fold 1, d_b=8, seed 42)
│   ├── run_main_experiments.py # Launcher 20 Main Runs trên data thật
│   ├── run_seed_experiments.py # Launcher Seed Stability (d_b=8, seeds 42, 123, 999 trên 5 folds)
│   ├── make_report.py          # Script tự động tạo báo cáo tổng hợp
│   └── utils.py                # Helper functions & kiểm tra phần cứng
├── tests/                      # Suite chứa unit & integration tests
│   ├── test_real_pipeline.py   # Suite 15 integration tests trên data thật
│   └── ...
├── checkpoints/                # Checkpoints thật sinh từ PPG-DaLiA
│   └── synthetic_smoke_test/   # Lưu trữ artifacts synthetic cũ để cô lập
├── logs/                       # Logs huấn luyện thật
│   └── synthetic_smoke_test/   # Logs synthetic cũ
├── results/                    # Kết quả CSV/PNG chính thức
│   ├── data_inventory.csv      # Báo cáo kiểm kê 15 subject
│   └── synthetic_smoke_test/   # Kết quả synthetic cũ
├── requirements.txt            # Thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 5. THIẾT LẬP MÔI TRƯỜNG (ENVIRONMENT SETUP)

```bash
# Di chuyển vào thư mục C1_AE_DCT
cd C1_AE_DCT

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 6. QUY TRÌNH PREPROCESSING THỰC NGHIỆM (DATA PREPROCESSING PIPELINE)

Thực thi kịch bản tiền xử lý dữ liệu thật từ PPG-DaLiA:

```bash
python -m src.prepare_data
```

**Các bước thực hiện:**
1. Đọc và kiểm định đủ 15 file `S1.pkl` .. `S15.pkl`.
2. Chiết xuất Wrist PPG (64 Hz) và Wrist ACC (32 Hz, 3 trục).
3. Resample ACC từ 32 Hz lên 64 Hz sử dụng `scipy.signal.resample_poly(acc, up=2, down=1, axis=0, window=("kaiser", 5.0), padtype="line")`.
4. Căn chỉnh PPG và ACC thành ma trận tín hiệu liên tục 4 kênh $[PPG, ACCx, ACCy, ACCz]$ shape $(N, 4)$.
5. Với mỗi fold $1 \dots 5$, tính toán thống kê Z-score (mean, std) **chỉ từ tập Train** và lưu vào `configs/norm_stats_fold{1..5}.json`.
6. Chuẩn hóa Z-score tín hiệu liên tục từng subject theo norm stats của fold tương ứng.
7. Cắt cửa sổ 8 giây ($T=512$ mẫu): step 4s ($S=256$) cho Train/Val, step 8s ($S=512$, không overlap) cho Test.
8. Lưu kết quả nén dạng `.npz` vào `data/processed/fold{1..5}/{train|val|test}.npz`.

---

## 7. CHẠY REAL PILOT RUN (REAL DATA VERIFICATION)

Chạy Pilot Run trên dữ liệu PPG-DaLiA thật (Fold 1, $d_b=8$, Seed 42, 10 epochs):

```bash
python -m src.pilot_run
```

Save checkpoint: `checkpoints/pilot_real_fold01_db08_seed42.pt` với cờ metadata `data_type = "REAL_DATA"`.

---

## 8. CHẠY 20 MAIN RUNS (MAIN EXPERIMENTS)

Khi đã được xác nhận nghiệm thu Pilot, chạy 20 thử nghiệm chính (5 Folds $\times 4 d_b \in \{16, 8, 4, 2\}$, Seed 42):

```bash
# Chạy chính thức (100 epochs, early stopping patience=10, batch_size=128):
python -m src.run_main_experiments

# Hoặc chạy thử nhanh smoke-test 5 epoch:
python -m src.run_main_experiments --smoke-test
```

---

## 9. CHẠY SEED EXPERIMENTS (SEED STABILITY)

Đánh giá độ ổn định mô hình trên cấu hình $d_b = 8$ với 3 seeds (42, 123, 999) trên cả 5 folds (tổng cộng 15 runs, trong đó seed 42 kế thừa từ Main Experiment):

```bash
python -m src.run_seed_experiments
```

*Lưu ý:* Seed stability được tổng hợp từ các chỉ số méo dạng thật theo subject trên Test set, không giả định 15 subjects $\times$ 3 seeds là 45 subjects độc lập.

---

## 10. ĐÁNH GIÁ CHỈ SỐ MÉO DẠNG (EVALUATION & METRICS)

Đánh giá các chỉ số méo dạng trên tín hiệu vật lý đã denormalize:

```bash
python -m src.evaluate
```

Vẽ đồ thị và tạo báo cáo tự động:
```bash
python -m src.make_report
```

---

## 11. VỊ TRÍ ARTIFACTS & PHÂN LOẠI SYNTHETIC VS REAL DATA

- **Dữ liệu kiểm kê (Real Data Inventory):** `results/data_inventory.csv`
- **Checkpoints thật:** `checkpoints/fold{F}_db{DB}_seed{S}.pt`
- **Thư mục cô lập dữ liệu thử nghiệm synthetic cũ:**
  - `checkpoints/synthetic_smoke_test/`
  - `logs/synthetic_smoke_test/`
  - `results/synthetic_smoke_test/`

---

## 🧪 KIỂM THỬ TỰ ĐỘNG (INTEGRATION & UNIT TESTS)

Chạy toàn bộ unit test và integration test suite:

```bash
python -m pytest tests/
```
