# 🫀 C1_AE_DCT — Wearable IoT Signal Compression Pipeline
> **Multichannel PPG & Tri-axial ACC Signal Compression using 1D-CNN Autoencoder and Discrete Cosine Transform (DCT-II) Baseline on PPG-DaLiA Dataset**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![SciPy](https://img.shields.io/badge/SciPy-Signal_Processing-005493?style=for-the-badge&logo=scipy)
![Tests](https://img.shields.io/badge/Tests-75%2F75%20PASSED-brightgreen?style=for-the-badge&logo=pytest)
![Dataset](https://img.shields.io/badge/Dataset-PPG--DaLiA-orange?style=for-the-badge)

Dự án triển khai giải pháp nén tín hiệu sinh học 4 kênh (Wrist PPG & Tri-axial Wrist ACC) từ tập dữ liệu thực nghiệm **PPG-DaLiA** trên thiết bị đeo IoT. Dự án so sánh đối đầu giữa mô hình **1D-CNN Autoencoder (AE)** và thuật toán cơ sở **Discrete Cosine Transform (DCT-II)** theo quy trình 5-fold cross-validation độc lập theo đối tượng (Subject-Wise Split, $S1 \dots S15$).

👉 **Xem Hướng dẫn Chi tiết & Mã nguồn Dự án tại:** [`C1_AE_DCT/README.md`](file:///d:/UTE/AIForIOT/project_Cuoiki_iot/C1_AE_DCT/README.md)

---

## ⚡ QUICK START (THỰC THI QUY TRÌNH DỮ LIỆU THẬT)

```bash
# 1. Di chuyển vào thư mục dự án
cd C1_AE_DCT

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Tiền xử lý dữ liệu thật PPG-DaLiA (Resample ACC 32->64Hz, Train norm stats, Windowing)
python -m src.prepare_data

# 4. Real Pilot Run (Fold 1, d_b=8, Seed 42)
python -m src.pilot_run

# 5. Thực thi End-to-End Sanity Check (Fold 1 Test Set)
python -m src.sanity_check_fold1

# 6. Kiểm thử Suite Tests Tự động (75/75 Passed)
python -m pytest tests/
```
