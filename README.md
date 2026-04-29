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

## 🚀 Hướng dẫn khởi chạy (Run Modes)

Dự án hỗ trợ 2 chế độ chạy tùy theo mục đích sử dụng của bạn:

### Cách 1: Chạy bằng Docker (Khuyến nghị cho Người dùng / Demo / Chấm điểm)
Cách này giúp tự động hóa 100% quá trình setup, không lo xung đột môi trường.

1. **Bật phần mềm Docker Desktop** trên máy tính (đảm bảo hệ thống báo *Engine running*).
2. Mở Terminal (PowerShell/CMD) tại thư mục gốc của dự án và chạy lệnh:
   ```bash
   docker-compose up
   ```
3. Mở trình duyệt và truy cập Web Dashboard tại: `http://localhost:8501`

*(💡 Mẹo: Để tắt web, nhấp vào Terminal và bấm `Ctrl + C`).*

---

### Cách 2: Chạy Local với Virtual Environment (Khuyến nghị cho Developer)
Dành cho việc tùy biến code, huấn luyện ngầm trên server hoặc chạy trực tiếp qua CLI.

1. Tạo, kích hoạt môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # (Dùng `source venv/bin/activate` nếu ở Mac/Linux)
   pip install -r requirements.txt
   ```
2. Khởi chạy Web Dashboard:
   ```bash
   streamlit run app/main.py
   ```

*(🛠️ Lưu ý khi thao tác trên VS Code: Bạn có thể dùng tính năng Dev Containers (Reopen in Container) để VS Code tự động chui vào môi trường Linux đã cấu hình sẵn. Khi đó, chỉ cần mở Terminal gõ `streamlit run app/main.py` mà không cần làm bước 1).*

---

## Workflow trong giao diện Streamlit

1. Load Data — upload hoặc chỉ đường dẫn đến file `.dat`
2. Preprocess — trích xuất FFT 5-band / PSD / DE
3. MRMR Selection — chọn top-K kênh EEG
4. Train Model — huấn luyện BiLSTM MRMR
5. Predict — dự đoán trên dữ liệu mới


## 💻 Huấn luyện qua Command Line (CLI)

Nếu bạn không muốn dùng giao diện Web mà muốn huấn luyện ngầm/đánh giá mô hình qua CLI (ví dụ MRMR BiLSTM):

```powershell
python -m src.train --target arousal --feat mrmr --data-dir data/raw --epochs 200
python -m src.train --target valence --feat mrmr --data-dir data/raw --epochs 200
```
*Lưu ý: đảm bảo đã chuẩn bị dữ liệu DEAP trong `data/raw/` (tập `s01.dat`…`s32.dat`).*


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
