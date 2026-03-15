# 🧠 EEG Emotion Recognition

Phân loại cảm xúc **Valence / Arousal** từ tín hiệu EEG sử dụng DEAP Dataset.  
Dự án môn học – Nhóm 4 người.

---

## 📁 Cấu trúc thư mục

```
eeg-emotion-recognition/
├── data/
│   ├── raw/          ← File .dat gốc từ DEAP (không push GitHub)
│   └── processed/    ← Đặc trưng PSD/DE đã trích xuất (.npy)
├── models/           ← Checkpoint mô hình (.pth) – không push GitHub
├── notebooks/
│   ├── 1.0-eda-eeg-signals.ipynb          ← Khám phá tín hiệu EEG
│   └── 2.0-feature-extraction-testing.ipynb ← Kiểm tra PSD vs DE
├── src/
│   ├── preprocess.py  ← Lọc, baseline removal, PSD, DE
│   ├── dataset.py     ← PyTorch Dataset cho DEAP
│   ├── models.py      ← CNN / LSTM / Transformer
│   ├── train.py       ← Script huấn luyện + lưu checkpoint
│   └── utils.py       ← Seed, checkpoint I/O, visualization
├── app/
│   └── main.py        ← Streamlit Dashboard
├── reports/
│   └── figures/       ← Biểu đồ kết quả huấn luyện
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt môi trường

```bash
# Tạo và kích hoạt virtualenv
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn chạy

### 1. Chuẩn bị dữ liệu
Tải DEAP dataset từ [trang chủ](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) và đặt các file `s01.dat … s32.dat` vào `data/raw/`.

### 2. Khám phá dữ liệu (EDA)
Mở notebook trên Kaggle / Colab hoặc chạy local:
```bash
jupyter lab notebooks/1.0-eda-eeg-signals.ipynb
```

### 3. Huấn luyện mô hình
```bash
# Valence với LSTM + PSD features – 50 epoch
python -m src.train --target valence --arch lstm --feat psd --epochs 50

# Arousal với Transformer + DE features
python -m src.train --target arousal --arch transformer --feat de --epochs 50
```
Checkpoint được lưu tự động vào `models/`.

### 4. Chạy Web Dashboard
```bash
streamlit run app/main.py
```
Mở trình duyệt tại `http://localhost:8501`.

---

## 🧪 Kiến trúc mô hình

| Model | Mô tả |
|---|---|
| `EEGConvNet` | 1-D CNN trên trục kênh EEG |
| `EEGLSTM` | Bidirectional LSTM, coi dải tần là time-step |
| `EEGTransformer` | Transformer Encoder, coi kênh EEG là token |

---

## 📊 Phân công nhiệm vụ

| Thành viên | Nhiệm vụ |
|---|---|
| Thành viên 1 | Tiền xử lý tín hiệu (`preprocess.py`) và EDA notebook |
| Thành viên 2 | Xây dựng Dataset + Kiến trúc mô hình (`dataset.py`, `models.py`) |
| Thành viên 3 | Script huấn luyện + đánh giá (`train.py`, `utils.py`) |
| Thành viên 4 | Web Dashboard Streamlit (`app/main.py`) + Báo cáo |

---

## 📄 Tài liệu tham khảo

- Koelstra, S. et al. *DEAP: A Database for Emotion Analysis Using Physiological Signals.* IEEE Transactions on Affective Computing, 2012.
- [MNE-Python Documentation](https://mne.tools/stable/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
