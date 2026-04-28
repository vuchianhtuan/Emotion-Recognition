Bài tập lớn — EEG Emotion Recognition

Phân loại cảm xúc Valence / Arousal từ tín hiệu EEG sử dụng bộ dữ liệu DEAP. Dự án tích hợp pipeline MRMR (Minimum Redundancy Maximum Relevance) để chọn kênh quan trọng và mô hình BiLSTM để huấn luyện.

---

**Nội dung chính**
- Tải và chuẩn bị dữ liệu DEAP
- Pipeline tiền xử lý (FFT 5-band, PSD, DE)
- Lựa chọn kênh bằng MRMR
- Mô hình BiLSTM MRMR để phân lớp cảm xúc
- Giao diện web bằng Streamlit để chạy thử nghiệm và demo

---

**Cấu trúc thư mục (chính)**

```text
Emotion-Recognition/
├── .devcontainer/
├── .streamlit/
├── app/                # Streamlit app + UI helpers
├── src/                # Core model + MRMR selection
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Bước 0 — Chuẩn bị dữ liệu DEAP (bắt buộc)

- Tải DEAP dataset (preprocessed) từ: https://drive.google.com/drive/folders/15O-qPO1ewUWPFb9kt8gTwmGv4cyr_FG5
- Đặt các file `s01.dat` … `s32.dat` vào `data/raw/` (tạo thư mục nếu chưa có).
- Không đẩy các file `.dat` này lên GitHub.

---

## Chạy dự án

Có hai cách chính để chạy:

1) Chạy bằng Docker (khuyến nghị cho người dùng)

- Yêu cầu: Docker Desktop
- Từ thư mục gốc, chạy:

```powershell
docker-compose up
```

- Mở trình duyệt: http://localhost:8501

2) Chạy local với virtual environment (developer)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

Lưu ý: trong VS Code bạn có thể dùng Dev Container (Reopen in Container) để phát triển trong môi trường Linux chứa sẵn dependencies; sau đó chạy `streamlit run app/main.py`.

---

## Workflow trong giao diện Streamlit

1. Load Data — upload hoặc chỉ đường dẫn đến file `.dat`
2. Preprocess — trích xuất FFT 5-band / PSD / DE
3. MRMR Selection — chọn top-K kênh EEG
4. Train Model — huấn luyện BiLSTM MRMR
5. Predict — dự đoán trên dữ liệu mới

---

## Chạy huấn luyện

Hiện tại project không có một script CLI chuẩn sẵn (ví dụ `src/train.py`).

- Để huấn luyện dễ nhất: dùng trang **Train Model** trong giao diện Streamlit (`app/main.py`) — chọn cấu hình, MRMR, và nhấn `Train`.
- Nếu bạn muốn huấn luyện không cần giao diện (headless), tôi có thể thêm một script CLI `src/train.py` theo mẫu sau:

```powershell
# ví dụ (tùy chỉnh khi script được thêm):
python -m src.train --target arousal --feat mrmr --data-dir data/raw --epochs 200
```

Hoặc hiện có thể sử dụng các helper trong `app/` (ví dụ `app/data_processing.py`, `app/model_utils.py`) để tự viết pipeline training.

---

## Chế độ chạy (Run modes)

Project hỗ trợ hai chế độ chạy chính — chọn theo nhu cầu:

- **Docker (recommended for users / demo):**
	- Yêu cầu: `Docker Desktop`.
	- Khởi chạy toàn bộ stack (web dashboard + môi trường đã cấu hình):

```powershell
docker-compose up
```

	- Mở trình duyệt: `http://localhost:8501`.

- **Local / CLI (recommended for development & training):**
	- Tạo và kích hoạt virtual environment, cài dependencies, và chạy Streamlit app:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

	- Huấn luyện/đánh giá mô hình qua CLI (ví dụ MRMR BiLSTM):

```powershell
python -m src.train --target arousal --feat mrmr --data-dir data/raw --epochs 200
python -m src.train --target valence --feat mrmr --data-dir data/raw --epochs 200
```

	- Lưu ý: đảm bảo đã chuẩn bị dữ liệu DEAP trong `data/raw/` (tập `s01.dat`…`s32.dat`).

---

## Preprocess & định dạng dữ liệu

- Pipeline preprocess (MNE-compatible) chuyển `data/raw/*.dat` → `data/processed/v2_mne/*.npz`.
- Mỗi `.npz` chứa:
	- `features`: `(n_windows, 32, 5)`
	- `labels`: `(n_windows, 2)`
	- `config_json`: cấu hình preprocess để reproducible

Core preprocess steps: band-pass, notch, baseline removal, windowing, FFT 5-band feature extraction.

---

## Kiến trúc mô hình (MRMR BiLSTM)

- Model chính: `EEGMRMRLSTMNet` — BiLSTM 5 tầng (PyTorch)

Kiến trúc tóm tắt:

Input: `(batch, K×5, 1)`  (K = số kênh sau MRMR × 5 dải tần)

- BiLSTM(128) → Dropout(0.5)
- LSTM(256) → Dropout(0.5)
- LSTM(64)  → Dropout(0.5)
- LSTM(64)  → Dropout(0.5)
- LSTM(32)  → Dropout(0.35)
- Dense(16) → ReLU
- Dense(2)  → Softmax/Logits (phân lớp Low/High)

---

## Scripts & modules chính

- `app/` — Streamlit app và helper UI (xem `app/main.py`)
- `app/data_processing.py` — chuẩn hóa / đóng gói dữ liệu cho model
- `app/data_io.py` — I/O cho dữ liệu DEAP / processed arrays
- `app/model_utils.py` — tập hợp hàm huấn luyện / checkpoint (dùng bởi app)
- `src/models.py` — định nghĩa mô hình (PyTorch)
- `src/mrmr_selection.py` — MRMR channel selection
- `app/main.py` — entry point Streamlit

---

## Tài liệu tham khảo

- Koelstra, S. et al. DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE TAC, 2012.
- MNE-Python: https://mne.tools/
- PyTorch: https://pytorch.org/
- mrmr-py: https://github.com/smazzanti/mrmr

---
