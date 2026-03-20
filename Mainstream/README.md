# 🧠 EEG Emotion Recognition

Phân loại cảm xúc **Valence / Arousal** từ tín hiệu EEG sử dụng DEAP Dataset.  
Tích hợp pipeline **MRMR** (Minimum Redundancy Maximum Relevance) từ `DEAP-Emotion-Recognition`.

---

## 📁 Cấu trúc thư mục

```
Mainstream/
├── data/
│   ├── raw/          ← File .dat gốc từ DEAP (không push GitHub)
│   └── processed/    ← Đặc trưng đã trích xuất (.npy)
├── models/           ← Checkpoint mô hình (.pth) – không push GitHub
├── notebooks/
│   ├── 1.0-eda-eeg-signals.ipynb
│   └── 2.0-feature-extraction-testing.ipynb
├── src/
│   ├── preprocess.py      ← Lọc, baseline removal, PSD, DE, FFT 5-band
│   ├── mrmr_selection.py  ← MRMR channel selection (numpy FFT thay pyeeg)
│   ├── dataset.py         ← PyTorch Dataset cho DEAP (PSD/DE)
│   ├── models.py          ← CNN / LSTM / Transformer / MRMR BiLSTM
│   ├── train.py           ← Script huấn luyện (PSD/DE + MRMR pipeline)
│   └── utils.py           ← Seed, checkpoint I/O, visualization
├── app/
│   └── main.py            ← Streamlit Dashboard đa trang
├── reports/
│   └── figures/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn chạy

### 1. Chuẩn bị dữ liệu
Tải DEAP dataset từ [trang chủ](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)  
Đặt `s01.dat … s32.dat` vào `data/raw/`.

### 2. Chạy Web Dashboard (khuyến nghị)
```bash
cd Mainstream
streamlit run app/main.py
```
Mở trình duyệt tại `http://localhost:8501`.

Workflow trong giao diện:
1. **📤 Load Data** – Upload file `.dat`
2. **⚡ Preprocess** – Trích xuất FFT 5 dải tần
3. **🔬 MRMR Selection** – Chọn top-K kênh EEG
4. **🎓 Train Model** – Huấn luyện BiLSTM MRMR
5. **🔮 Predict** – Dự đoán cảm xúc từ dữ liệu mới

### 3. Huấn luyện qua CLI

#### Pipeline MRMR (BiLSTM – khớp kiến trúc DEAP gốc)
```bash
python -m src.train --target arousal --feat mrmr --data-dir data/raw --epochs 200
python -m src.train --target valence --feat mrmr --data-dir data/raw --epochs 200
```

#### Pipeline PSD/DE (CNN/LSTM/Transformer)
```bash
python -m src.train --target valence --arch lstm   --feat psd --epochs 50
python -m src.train --target arousal --arch transformer --feat de --epochs 50
```

### 4. Preprocess chuẩn DEAP (MNE) ra `data/processed/`

Pipeline preprocess đã chuẩn hóa theo schema trung gian:

- `raw` (`.dat`) → `clean` (band-pass + notch + baseline + artifact clamp)
- `window` (sliding window)
- `feature` (FFT 5-band / channel)
- `label` (`[valence_bin, arousal_bin]`)

Chạy end-to-end:

```bash
python -m src.preprocess --data-dir data/raw --output-dir data/processed --version v2_mne
```

Output được version hóa tại:

```text
data/processed/v2_mne/s01.npz
data/processed/v2_mne/s02.npz
...
```

Mỗi `.npz` gồm:
- `features`: `(n_windows, 32, 5)`
- `labels`: `(n_windows, 2)`
- `config_json`: cấu hình preprocess để reproducible

`src.mrmr_selection.preprocess_subject_fft` và pipeline train/app MRMR dùng cùng core preprocess này để bảo đảm thống nhất train và inference.

---

## 🧪 Kiến trúc mô hình

| Model | Mô tả | Dùng với |
|---|---|---|
| `EEGConvNet` | 1-D CNN trên trục kênh EEG | `--feat psd/de` |
| `EEGLSTM` | Bidirectional LSTM, coi dải tần là time-step | `--feat psd/de` |
| `EEGTransformer` | Transformer Encoder, coi kênh EEG là token | `--feat psd/de` |
| `EEGMRMRLSTMNet` | **BiLSTM 5 tầng** – khớp kiến trúc DEAP gốc | `--feat mrmr` |

### Kiến trúc MRMR BiLSTM
```
Input: (batch, K×5, 1)   ← K kênh MRMR × 5 dải tần
  BiLSTM(128) → Dropout(0.5)
  LSTM(256)   → Dropout(0.5)
  LSTM(64)    → Dropout(0.5)
  LSTM(64)    → Dropout(0.5)
  LSTM(32)    → Dropout(0.35)
  Dense(16) → ReLU
  Dense(2)  → [Low, High]
```

---

## 🔄 Chuyển đổi từ DEAP-Emotion-Recognition

| Thành phần gốc | Phiên bản Mainstream | Thay đổi |
|---|---|---|
| `pyeeg.bin_power` | `src/mrmr_selection.bin_power_fft` | numpy FFT thay pyeeg |
| `FeatureExtraction/MRMR.py` | `src/mrmr_selection.py` | Tương thích Python ≥ 3.10 |
| `LSTMModel/Model.py` (TF/Keras) | `src/models.EEGMRMRLSTMNet` (PyTorch) | PyTorch thay TensorFlow |
| `LSTMModel/PrepareDataset.py` | `src/mrmr_selection.prepare_for_lstm` | Sửa data-leakage bug |
| `PreProcessing/FFT.py` | `src/mrmr_selection.preprocess_subject_fft` | numpy/scipy thay pyeeg |

---

## 📄 Tài liệu tham khảo

- Koelstra, S. et al. *DEAP: A Database for Emotion Analysis Using Physiological Signals.* IEEE TAC, 2012.
- [MNE-Python Documentation](https://mne.tools/stable/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [mrmr-py](https://github.com/smazzanti/mrmr)
