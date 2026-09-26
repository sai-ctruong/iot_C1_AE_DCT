# C1_AE_DCT — Signal Compression with Autoencoder & DCT Baseline
## Comprehensive Final Reproducibility Package & Execution Guide

Project mã nguồn mở nghiên cứu và triển khai giải pháp nén tín hiệu sinh học nhiều kênh (Multichannel PPG & Tri-axial ACC) trên thiết bị đeo IoT. Dự án so sánh đối đầu giữa mô hình học sâu **1D-CNN Autoencoder (AE)** và thuật toán cơ sở **Discrete Cosine Transform (DCT)** dựa trên tập dữ liệu PPG-DaLiA.

---

## 📋 MỤC LỤC (TABLE OF CONTENTS)
1. [Mục tiêu đề tài (Project Objectives)](#1-mục-tiêu-đề-tài-project-objectives)
2. [Nguồn tải PPG-DaLiA (Dataset Download Source)](#2-nguồn-tải-ppg-dalia-dataset-download-source)
3. [Cấu trúc thư mục (Directory Structure)](#3-cấu-trúc-thư-mục-directory-structure)
4. [Tạo Environment & Cài đặt (Setup Environment)](#4-tạo-environment--cài-đặt-setup-environment)
5. [Chuẩn bị Dataset (Dataset Preparation)](#5-chuẩn-bị-dataset-dataset-preparation)
6. [Tạo Folds (Subject-Wise 5-Fold Split)](#6-tạo-folds-subject-wise-5-fold-split)
7. [Chạy Preprocessing (Data Pipeline)](#7-chạy-preprocessing-data-pipeline)
8. [Chạy DCT Baseline Experiments](#8-chạy-dct-baseline-experiments)
9. [Chạy một AE Experiment (Pilot Run)](#9-chạy-một-ae-experiment-pilot-run)
10. [Chạy toàn bộ 20 Main Runs](#10-chạy-toàn-bộ-20-main-runs)
11. [Chạy Seed Experiments (Seed Robustness)](#11-chạy-seed-experiments-seed-robustness)
12. [Đánh giá chỉ số Méo dạng (Evaluate Metrics)](#12-đánh-giá-chỉ-số-méo-dạng-evaluate-metrics)
13. [Tạo Bảng biểu & Đồ thị (Generate Plots & Tables)](#13-tạo-bảng-biểu--đồ-thị-generate-plots--tables)
14. [Vị trí Checkpoint / Log / Result (Outputs Directory)](#14-vị-trí-checkpoint--log--result-outputs-directory)
15. [Cấu hình Phần cứng / Phần mềm (System Specs)](#15-cấu-hình-phần-cứng--phần-mềm-system-specs)
16. [Những Experiment chưa hoàn thành / Hướng phát triển (Roadmap)](#16-những-experiment-chưa-hoàn-thành--hướng-phát-triển-roadmap)

---

## 1. MỤC TIÊU ĐỀ TÀI (PROJECT OBJECTIVES)

- **Bài toán:** Nén tín hiệu 4 kênh bao gồm 1 kênh Quang thể tích đồ (PPG @ 64Hz) và 3 kênh Gia tốc kế (ACCx, ACCy, ACCz @ 32Hz) từ thiết bị đeo (Empatica E4) thu thập trong chuỗi hoạt động thể chất phức tạp.
- **Phương pháp:**
  1. **1D-CNN Autoencoder (AE):** Nén tín hiệu vật lý $4 \times 512$ thành không gian ẩn $M = 32 \times d_b$ ($d_b \in \{16, 8, 4, 2\}$ tương ứng $\text{CR}_{\text{dim}} \in \{4, 8, 16, 32\}$).
  2. **Discrete Cosine Transform (DCT):** Thuật toán nén chuẩn baseline chọn $K$ hệ số lớn nhất với ngân sách byte khớp chính xác $B_{\text{DCT}} = B_{\text{AE}}$.
- **Tiêu chuẩn đánh giá:**
  - Đánh giá độc lập trên từng kênh (`PPG`, `ACCx`, `ACCy`, `ACCz`).
  - Sử dụng các metric méo dạng trên tín hiệu vật lý: **PRD**, **PRDN** (dùng năng lượng mẫu trừ mean), và **RMSE**.
  - Kiểm thử 5-fold cross-validation theo đối tượng (Subject-Wise Split, 15 đối tượng $S1 \dots S15$).

---

## 2. NGUỒN TẢI PPG-DALIA (DATASET DOWNLOAD SOURCE)

Tập dữ liệu **PPG-DaLiA** công khai trên UCI Machine Learning Repository:
- **Link tải trực tiếp:** [UCI Machine Learning Repository — PPG-DaLiA Dataset](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
- **Trích dẫn khoa học:**
  > Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). *Deep PPG: Large-Scale Heart Rate Estimation from Photoplethysmography Using Convolutional Neural Networks*. Sensors, 19(14), 3079.
- **Định dạng file thô:** 15 file pickle từ `S1.pkl` đến `S15.pkl`.

---

## 3. CẤU TRÚC THƯ MỤC (DIRECTORY STRUCTURE)

```text
project_Cuoiki_iot/
└── C1_AE_DCT/
    ├── configs/
    │   ├── config.json             # File cấu hình chung của dự án
    │   ├── config.yaml             # File cấu hình dạng YAML
    │   └── folds.json              # File định nghĩa 5-fold subject split (S1-S15)
    ├── data/
    │   ├── raw/                    # Chứa 15 file dữ liệu thô S1.pkl .. S15.pkl
    │   └── processed/              # Chứa dữ liệu đã resample & windowing (.npz)
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py              # Loader & PyTorch Dataset class (C1Dataset)
    │   ├── resample.py             # Resample ACC 32Hz -> 64Hz bằng Polyphase filter
    │   ├── windowing.py            # Cắt cửa sổ T=512 (8s), stride S=256 (4s)
    │   ├── normalize.py            # Z-score normalization (Fold-wise Train statistics)
    │   ├── model.py                # Kiến trúc 1D-CNN Autoencoder (C1Autoencoder)
    │   ├── loss.py                 # Hàm tổn thất Reconstruction MSE Loss
    │   ├── baseline_dct.py         # Thuật toán nén DCT baseline (Top-K & equal byte)
    │   ├── codec.py                # Quantization & Byte Bitstream Codec (C1Codec)
    │   ├── metrics.py              # Tính CR_dim, CR_byte và ngân sách byte (B_AE = B_DCT)
    │   ├── evaluate.py             # Tính PRD, PRDN, RMSE độc lập từng kênh & valid-mask
    │   ├── results_schema.py       # Cấu trúc bảng kết quả 19 trường & join AE/DCT
    │   ├── aggregation.py          # Aggregate theo subject & tính delta_s = AE - DCT
    │   ├── reporting.py            # Tự động xuất 5 đồ thị (PNG+CSV) và 4 bảng biểu
    │   ├── reconstruction_visualization.py # Đồ thị so sánh dạng sóng & failure cases
    │   ├── train.py                # Pipeline huấn luyện Autoencoder
    │   ├── pilot_run.py            # Pilot run (Fold 1, d_b=8, seed 42)
    │   ├── run_main_experiments.py # Chạy toàn bộ 20 main runs (5F x 4db)
    │   ├── run_seed_experiments.py # Chạy 10 seed robustness runs
    │   ├── make_report.py          # Script tổng hợp báo cáo tự động (TASK 24)
    │   └── utils.py                # Helper functions & kiểm tra phần cứng
    ├── tests/                      # Suite chứa 60 Unit Tests kiểm thử tự động
    ├── checkpoints/                # Lưu file weights mô hình (.pt)
    ├── logs/                       # Nhật ký huấn luyện (Tensorboard & logs)
    ├── results/                    # Kết quả xuất tự động (CSVs, PNGs, experiment_summary.md)
    ├── requirements.txt            # Thư viện phụ thuộc
    └── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 4. TẠO ENVIRONMENT & CÀI ĐẶT (SETUP ENVIRONMENT)

### Cách 1: Sử dụng `venv` (Phổ biến)

```bash
# 1. Di chuyển vào thư mục dự án
cd C1_AE_DCT

# 2. Tạo môi trường ảo
python -m venv .venv

# 3. Kích hoạt môi trường ảo
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Nâng cấp pip & cài đặt dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Cách 2: Sử dụng Anaconda / Miniconda

```bash
conda create -n c1_ae_dct python=3.10 -y
conda activate c1_ae_dct
pip install -r requirements.txt
```

---

## 5. CHUẨN BỊ DATASET (DATASET PREPARATION)

1. Tải 15 file `S1.pkl` đến `S15.pkl` từ nguồn PPG-DaLiA.
2. Đặt tất cả 15 file vào thư mục:
   `C1_AE_DCT/data/raw/`
3. Kiểm tra sự tồn tại của file:
   ```bash
   python -c "from pathlib import Path; print(len(list(Path('C1_AE_DCT/data/raw').glob('*.pkl'))))"
   # Trả về 15 là thành công.
   ```

---

## 6. TẠO FOLDS (SUBJECT-WISE 5-FOLD SPLIT)

Khởi tạo và xác minh phân chia 5-fold theo đối tượng (mỗi fold gồm 10 Train, 2 Val, 3 Test subjects, không trùng lặp):

```bash
python C1_AE_DCT/src/utils.py
```

Hoặc chạy test kiểm thử fold:
```bash
python -m pytest C1_AE_DCT/tests/test_folds.py
```

---

## 7. CHẠY PREPROCESSING (DATA PIPELINE)

Quy trình tiền xử lý tín hiệu thực thi đồng thời:
1. **Resample:** Đưa kênh ACC từ 32Hz lên 64Hz bằng bộ lọc Polyphase chống méo tần số.
2. **Alignment:** Căn chỉnh thời gian khớp 4 kênh $[PPG, ACCx, ACCy, ACCz]$ tại tần số 64Hz.
3. **Windowing:** Cắt cửa sổ $T=512$ mẫu (8 giây), độ dịch $S=256$ mẫu (chồng lấp 50%).
4. **Normalization:** Chuẩn hóa Z-score với thống kê (mean, std) tính **chỉ trên tập Train** của từng fold.

Thực thi kịch bản preprocessing:
```bash
python C1_AE_DCT/src/dataset.py
```

---

## 8. CHẠY DCT BASELINE EXPERIMENTS

Chạy thử nghiệm nén cơ sở với biến đổi cosin rời rạc (DCT):

```bash
python C1_AE_DCT/src/baseline_dct.py
```

Hoặc thực thi suite kiểm thử DCT:
```bash
python -m pytest C1_AE_DCT/tests/test_dct_suite.py
```

---

## 9. CHẠY MỘT AE EXPERIMENT (PILOT RUN)

Thực thi Pilot Run đại diện cho **Fold 1, $d_b = 8$, Seed 42** để kiểm tra bộ nhớ RAM/VRAM, tốc độ huấn luyện và hình dạng latent vector:

```bash
python C1_AE_DCT/src/pilot_run.py
```

**Kết quả dự kiến:**
- Thời gian huấn luyện: ~0.46 giây / epoch.
- Chiều Latent vector: $[B, 8, 32] \implies M = 256$ giá trị/mẫu.
- Kích thước Checkpoint file: ~124 KB.

---

## 10. CHẠY TOÀN BỘ 20 MAIN RUNS

Chạy toàn bộ 20 thử nghiệm chính (5 Folds $\times$ 4 Mức ngân sách $d_b \in \{16, 8, 4, 2\}$):

```bash
python C1_AE_DCT/src/run_main_experiments.py
```

*Lưu ý: Kết quả chi tiết của từng cửa sổ sẽ tự động ghi vào `results/results.csv`.*

---

## 11. CHẠY SEED EXPERIMENTS (SEED ROBUSTNESS)

Đánh giá độ ổn định của Autoencoder qua 10 seed ngẫu nhiên khác nhau trên Fold 1, $d_b = 8$:

```bash
python C1_AE_DCT/src/run_seed_experiments.py
```

---

## 12. ĐÁNH GIÁ CHỈ SỐ MÉO DẠNG (EVALUATE METRICS)

Đánh giá các chỉ số méo dạng **PRD**, **PRDN** và **RMSE** riêng cho từng kênh tín hiệu (`PPG`, `ACCx`, `ACCy`, `ACCz`) sau khi denormalize:

```bash
python C1_AE_DCT/src/evaluate.py
```

Chạy unit tests cho module evaluate (5 test cases của TASK 19):
```bash
python -m pytest C1_AE_DCT/tests/test_metrics_eval.py
```

---

## 13. TẠO BẢNG BIỂU & ĐỒ THỊ (GENERATE PLOTS & TABLES)

Thực thi tự động quy trình tổng hợp cấp subject, vẽ 5 đồ thị chất lượng cao và xuất 4 bảng biểu khoa học:

```bash
python C1_AE_DCT/src/make_report.py
```

**Các file kết quả sinh tự động trong `results/`:**
- `summary_by_subject.csv`
- `paired_comparison.csv`
- `overall_summary.csv`
- `cr_dim_prd.png` & `cr_dim_prd_data.csv`
- `cr_byte_prd.png` & `cr_byte_prd_data.csv`
- `prdn_curves.png` & `prdn_curves_data.csv`
- `rmse_curves.png` & `rmse_curves_data.csv`
- `reconstruction_examples.png`
- `failure_cases.png` & `failure_cases.csv`
- `experiment_summary.md`

---

## 14. VỊ TRÍ CHECKPOINT / LOG / RESULT (OUTPUTS DIRECTORY)

- **Checkpoints mô hình:** `C1_AE_DCT/checkpoints/`
  - `pilot_model_fold1_db8.pt`
  - `model_fold{F}_db{DB}.pt`
- **Nhật ký huấn luyện:** `C1_AE_DCT/logs/`
  - Nhật ký văn bản và file Tensorboard.
- **Báo cáo & Đồ thị xuất ra:** `C1_AE_DCT/results/`
  - Chứa toàn bộ các file CSVs, PNGs và file markdown `experiment_summary.md`.

---

## 15. CẤU HÌNH PHẦN CỨNG / PHẦN MỀM (SYSTEM SPECS)

Thông số môi trường được thử nghiệm và nghiệm thu:

| Thông số | Chi tiết cấu hình |
| :--- | :--- |
| **Hệ điều hành** | Windows 11 Home/Pro 64-bit |
| **Bộ xử lý (CPU)** | AMD Ryzen Series (AMD64 Architecture) |
| **Bộ nhớ RAM** | 16 GB System Memory |
| **Card đồ họa (GPU)** | NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM) |
| **Python** | Python `3.14.5` (hoặc Python `3.10+`) |
| **Thư viện chính** | PyTorch `2.x`, NumPy `2.x`, SciPy `1.x`, Matplotlib `3.x`, PyTest `9.x` |

---

## 16. NHỮNG EXPERIMENT CHƯA HOÀN THÀNH / HƯỚNG PHÁT TRIỂN (ROADMAP)

- **Tình trạng thử nghiệm hiện tại:**
  - Tất cả các module core, preprocessing, DCT baseline, Autoencoder architecture, loss, metrics, evaluate, results schema, aggregation, reporting và 60 unit tests đã hoàn thành 100%.
- **Hướng phát triển tiếp theo:**
  - Tích hợp thêm module lượng hóa số nguyên Bit-level Entropy Coding (HUFFMAN / ANS) để so sánh chỉ số $CR_{\text{bit}}$ thực tế truyền qua sóng vô tuyến LoRaWAN / BLE.
  - Mở rộng thử nghiệm trên các tập dữ liệu bổ sung (IEEE SPC, WESAD).

---

## 🧪 KIỂM THỬ TOÀN BỘ SUITE (RUNNING ALL UNIT TESTS)

Để xác minh toàn bộ hệ thống dự án hoạt động chính xác từ đầu đến cuối, chạy lệnh:

```bash
python -m pytest C1_AE_DCT/tests/
```

**Kỳ vọng:** `60 passed in ~12.5s` (100% Passed).
