Bài tập lớn môn các vấn đề hiện đại của Kỹ thuật máy tính
# 🧠 EEG Emotion Recognition

Dự án phân loại cảm xúc **Valence / Arousal** từ tín hiệu điện não đồ (EEG) sử dụng bộ dữ liệu DEAP.
Hệ thống tích hợp pipeline **MRMR** (Minimum Redundancy Maximum Relevance) để tối ưu hóa việc chọn kênh tín hiệu và sử dụng mạng **BiLSTM** để huấn luyện mô hình.

---

## 📁 Cấu trúc thư mục cơ bản

Emotion-Recognition/
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml
├── app/
│   ├── config.py
│   ├── data_io.py
│   ├── data_normalization.py
│   ├── data_processing.py
│   ├── main.py
│   ├── model_utils.py
│   ├── pages.py
│   ├── state_management.py
│   ├── ui_components.py
│   └── ui_helpers.py
├── src/
│   ├── models.py
│   └── mrmr_selection.py
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README-app.md
├── README.md
└── requirements.txt

---

## 🚀 Hướng dẫn cài đặt và khởi chạy

### 0. Chuẩn bị dữ liệu DEAP (Bắt buộc)
Dù chạy bằng phương pháp nào, bạn cũng cần chuẩn bị dữ liệu gốc:
* Tải DEAP dataset (phiên bản preprocessed) từ [Trang chủ DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
* Đặt các file từ `s01.dat` đến `s32.dat` vào thư mục `data/raw/` (tạo thư mục nếu chưa có).
* *(Lưu ý: Tuyệt đối không push các file `.dat` này lên GitHub).*

Bạn có thể khởi chạy dự án theo 1 trong 2 cách dưới đây:

### Cách 1: Chạy bằng Docker (Khuyến nghị - One-command run)
Cách này giúp tự động hóa 100% quá trình cài đặt, không lo xung đột thư viện hay khác biệt hệ điều hành. Rất phù hợp để test nhanh hoặc chấm điểm.

**Yêu cầu:** Máy tính cần cài đặt sẵn [Docker Desktop](https://www.docker.com/products/docker-desktop/).

1. Mở Terminal tại thư mục gốc của dự án.

2. Khởi chạy hệ thống bằng 1 câu lệnh duy nhất:

   docker-compose up

Mở trình duyệt và truy cập Web Dashboard tại: http://localhost:8501
(Lưu ý cho Developer: Khi phát triển, hãy mở dự án bằng VS Code và chọn Reopen in Container để bật tính năng tự động cập nhật code (hot-reload)).

### Cách 2: Chạy Local với Virtual Environment (Cách truyền thống)
Dành cho những ai muốn cài đặt và quản lý gói trực tiếp trên máy tính cá nhân.

1. Tạo và kích hoạt môi trường ảo:

Tạo môi trường: python -m venv venv

Kích hoạt (Windows): venv\Scripts\activate

Kích hoạt (Mac/Linux): source venv/bin/activate

2. Cài đặt thư viện:

pip install -r requirements.txt

3. Khởi chạy Web Dashboard:

streamlit run app/main.py

(Trình duyệt sẽ tự động mở tại địa chỉ: http://localhost:8501)

**Workflow trên Dashboard:**
1. **📤 Load Data:** Nạp file `.dat` từ DEAP.
2. **⚡ Preprocess:** Cắt cửa sổ trượt (sliding window) và trích xuất đặc trưng FFT 5 dải tần.
3. **🎓 Train Model:** Tự động chạy thuật toán MRMR chọn top kênh tối ưu và huấn luyện mô hình.
4. **🔮 Predict:** Suy luận song song trên dữ liệu mới và trả về độ chính xác theo từng Trial.

### 3. Huấn luyện qua Command Line (CLI)
Nếu bạn muốn chạy ngầm hoặc huấn luyện trên server, sử dụng các lệnh sau cho từng nhãn cảm xúc:

**Huấn luyện mô hình cho nhãn Arousal:**
`python -m src.train --target arousal --feat mrmr --data-dir data/raw --epochs 200`

**Huấn luyện mô hình cho nhãn Valence:**
`python -m src.train --target valence --feat mrmr --data-dir data/raw --epochs 200`

---

## 🧪 Kiến trúc mô hình

Mô hình học sâu chính được sử dụng là **BiLSTM 5 tầng**, được xây dựng bằng PyTorch để thay thế cho phiên bản TensorFlow/Keras cũ.

| Model | Mô tả | Dùng với Command |
|---|---|---|
| `EEGMRMRLSTMNet` | **BiLSTM 5 tầng** – khớp kiến trúc DEAP gốc | `--feat mrmr` |

**Chi tiết luồng mạng MRMR BiLSTM:**
* **Input:** `(batch, K×5, 1)` ← K kênh (lọc bởi MRMR) × 5 dải tần (FFT)
* **Layer 1:** `BiLSTM(128)` → `Dropout(0.5)`
* **Layer 2:** `LSTM(256)` → `Dropout(0.5)`
* **Layer 3:** `LSTM(64)` → `Dropout(0.5)`
* **Layer 4:** `LSTM(64)` → `Dropout(0.5)`
* **Layer 5:** `LSTM(32)` → `Dropout(0.35)`
* **Layer 6:** `Dense(16)` → `ReLU`
* **Output:** `Dense(2)` → Phân loại `[Low, High]`

---

## 🔄 Nhật ký chuyển đổi (Migration Log)
So với mã nguồn gốc của bài báo `DEAP-Emotion-Recognition`, dự án này đã có những cải tiến kỹ thuật cốt lõi:

| Thành phần gốc | Phiên bản hiện tại | Cải tiến đáng chú ý |
|---|---|---|
| `pyeeg.bin_power` | `src/mrmr_selection.bin_power_fft` | Tối ưu bằng Numpy FFT, không dùng thư viện pyeeg cũ. |
| `FeatureExtraction/MRMR.py` | `src/mrmr_selection.py` | Tương thích hoàn toàn với Python ≥ 3.10. Chạy đa luồng (multi-core). |
| `LSTMModel/Model.py` | `src/models.EEGMRMRLSTMNet` | **Chuyển đổi từ TensorFlow/Keras sang hoàn toàn bằng PyTorch.** |
| `LSTMModel/PrepareDataset.py`| `src/mrmr_selection.prepare_for_lstm` | **Sửa lỗi Data-leakage** (Phân chia Train/Test bằng Modulo thay vì Random). |

---

## 📄 Tài liệu tham khảo
* Koelstra, S. et al. *DEAP: A Database for Emotion Analysis Using Physiological Signals.* IEEE TAC, 2012.
* [MNE-Python Documentation](https://mne.tools/stable/index.html)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [mrmr-py (Feature Selection Library)](https://github.com/smazzanti/mrmr)