# C1_AE_DCT — Signal Compression with Autoencoder & DCT Baseline

Project nghiên cứu và triển khai giải pháp nén tín hiệu sinh học 4 kênh (Wrist PPG & Tri-axial Wrist ACC) từ tập dữ liệu thực nghiệm **PPG-DaLiA**. Dự án so sánh đối đầu giữa mô hình **1D-CNN Autoencoder (AE)** và thuật toán cơ sở **Discrete Cosine Transform (DCT)** theo quy trình 5-fold cross-validation độc lập theo đối tượng (Subject-Wise Split, $S1 \dots S15$).

Xem tài liệu hướng dẫn chi tiết và mã nguồn tại: [`C1_AE_DCT/README.md`](file:///d:/UTE/AIForIOT/project_Cuoiki_iot/C1_AE_DCT/README.md).

## Quick Start (Thực thi quy trình dữ liệu thật)

1. Preprocessing dữ liệu thật PPG-DaLiA:
   ```bash
   python -m src.prepare_data
   ```

2. Real Pilot Run (Fold 1, $d_b=8$, Seed 42):
   ```bash
   python -m src.pilot_run
   ```

3. Running Integration & Unit Tests:
   ```bash
   python -m pytest tests/
   ```
