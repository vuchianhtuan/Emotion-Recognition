"""
app/main.py
-----------
Streamlit Web Dashboard – EEG Emotion Recognition (DEAP + MRMR)

Giao diện đa trang tích hợp pipeline MRMR từ DEAP-Emotion-Recognition:
  🏠 Home            – Giới thiệu & hướng dẫn
  📤 Load Data       – Upload file .dat DEAP, xem trước tín hiệu
  ⚡ Preprocess      – Trích xuất đặc trưng FFT 5 dải
  🔬 MRMR Selection  – Chọn kênh MRMR, xem kết quả
  🎓 Train Model     – Huấn luyện BiLSTM với đặc trưng MRMR
  🔮 Predict         – Upload file .npy để dự đoán cảm xúc

Cách chạy:
    cd Mainstream
    streamlit run app/main.py
"""

import io
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import streamlit as st

# Thêm thư mục gốc vào sys.path để import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_model
from src.mrmr_selection import (
    preprocess_subject_fft,
    run_mrmr_selection,
    build_mrmr_dataset,
    prepare_for_lstm,
    BANDS,
    N_CHANNELS,
    N_FREQUENCIES,
    MRMR_COMPONENTS,
)
from src.preprocess import MRMR_BAND_NAMES
from src.utils import save_checkpoint, load_checkpoint, plot_history, set_seed

# ── Page config ──────────────────────────────────────────────────── #
st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BATCH_SIZE = 256
LR = 1e-3
DEAP_ELECTRODES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]


# ── Session-state helpers ─────────────────────────────────────────── #
def _init_state():
    defaults = {
        "page":                  "Home",
        "subjects_raw":          [],   # list of (filename, subject_dict)
        "subjects_preprocessed": [],   # list of np.ndarray from preprocess_subject_fft
        "selected_channels":     [],   # list of channel lists per subject
        "x_train": None, "y_train": None,
        "x_test":  None, "y_test":  None,
        "train_history":         None,
        "trained_model":         None,
        "trained_seq_len":       None,
        "classify_type":         "arousal",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def goto(page: str):
    st.session_state.page = page


# ── Sidebar navigation ────────────────────────────────────────────── #
with st.sidebar:
    st.title("🧠 EEG Emotion")
    st.caption("Recognition Dashboard")
    st.markdown("---")

    pages = {
        "Home":           "🏠 Home",
        "Load Data":      "📤 Load DEAP Data",
        "Preprocess":     "⚡ Preprocess (FFT)",
        "MRMR Selection": "🔬 MRMR Channel Selection",
        "Train Model":    "🎓 Train Model",
        "Predict":        "🔮 Predict",
    }
    for key, label in pages.items():
        is_active = st.session_state.page == key
        st.button(label, on_click=goto, args=(key,),
                  use_container_width=True,
                  type="primary" if is_active else "secondary")

    st.markdown("---")
    st.caption("DEAP Dataset · MRMR · BiLSTM")
    st.caption("Mainstream – Emotion Recognition")


# ═══════════════════════════════════════════════════════════════════ #
#  PAGE: Home
# ═══════════════════════════════════════════════════════════════════ #
def page_home():
    st.title("🧠 EEG Emotion Recognition")
    st.subheader("Nhận diện cảm xúc từ tín hiệu EEG – DEAP Dataset")

    st.markdown("""
    <div style="background:#eaf5ff;padding:18px;border-radius:10px;border:1px solid #cce0ff">
    <p>Ứng dụng này cung cấp pipeline đầy đủ để phân loại cảm xúc <b>Arousal / Valence</b>
    từ tín hiệu EEG sử dụng thuật toán lựa chọn kênh <b>MRMR</b> (Minimum Redundancy
    Maximum Relevance) và mô hình <b>Bidirectional LSTM</b>.</p>
    <p>Pipeline chuyển đổi từ <em>DEAP-Emotion-Recognition</em> sang thư viện Python hiện đại
    (numpy FFT thay thế pyeeg, PyTorch thay thế TensorFlow/Keras).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Quy trình sử dụng")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1️⃣ Load Data**\nUpload file `.dat` từ DEAP dataset (s01.dat … s32.dat)")
        st.info("**4️⃣ MRMR Selection**\nChọn top-K kênh EEG ít dư thừa, nhiều thông tin nhất")
    with col2:
        st.success("**2️⃣ Preprocess**\nTrích xuất đặc trưng FFT 5 dải tần: Theta / Alpha / LowerBeta / UpperBeta / Gamma")
        st.success("**5️⃣ Train Model**\nHuấn luyện BiLSTM với đặc trưng MRMR, theo dõi loss/accuracy")
    with col3:
        st.warning("**3️⃣ Explore**\nXem trước tín hiệu EEG và phân bố nhãn")
        st.warning("**6️⃣ Predict**\nDự đoán cảm xúc từ file đặc trưng `.npy` mới")

    st.markdown("---")
    st.subheader("🏗️ Kiến trúc mô hình MRMR LSTM")
    arch_df = pd.DataFrame({
        "Layer": ["BiLSTM(128)", "LSTM(256)", "LSTM(64)", "LSTM(64)", "LSTM(32)", "Dense(16)", "Dense(2)"],
        "Output size": ["(B, T, 256)", "(B, T, 256)", "(B, T, 64)", "(B, T, 64)", "(B, T, 32)", "(B, 16)", "(B, 2)"],
        "Dropout": ["0.5", "0.5", "0.5", "0.5", "0.35", "–", "–"],
    })
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("📚 Tài liệu: Koelstra et al. *DEAP: A Database for Emotion Analysis Using Physiological Signals.* IEEE TAC 2012.")


# ═══════════════════════════════════════════════════════════════════ #
#  PAGE: Load Data
# ═══════════════════════════════════════════════════════════════════ #
def page_load_data():
    st.title("📤 Load DEAP Data")
    st.markdown("Upload một hoặc nhiều file `.dat` từ DEAP dataset.")

    classify_type = st.selectbox("Nhãn phân loại", ["arousal", "valence"], key="cls_type_load")
    st.session_state.classify_type = classify_type

    uploaded = st.file_uploader(
        "Chọn file .dat (có thể upload nhiều file)",
        type=["dat"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("🔄 Load & Preview", type="primary"):
            subjects = []
            prog = st.progress(0)
            for i, f in enumerate(uploaded):
                prog.progress((i + 1) / len(uploaded))
                raw = pickle.load(io.BytesIO(f.read()), encoding="latin1")
                subjects.append((f.name, raw))
                st.toast(f"✅ Đã load: {f.name}")
            st.session_state.subjects_raw = subjects
            st.success(f"Đã load {len(subjects)} subject(s).")

    if st.session_state.subjects_raw:
        st.markdown("---")
        st.subheader("📊 Preview dữ liệu")

        fname, subj = st.session_state.subjects_raw[0]
        st.write(f"**File**: `{fname}` | Trials: {subj['data'].shape[0]} | "
                 f"Channels: {subj['data'].shape[1]} | Samples: {subj['data'].shape[2]}")

        trial_idx = st.slider("Chọn Trial", 0, subj["data"].shape[0] - 1, 0)
        ch_idx    = st.slider("Chọn Kênh EEG", 0, N_CHANNELS - 1, 0)

        raw_signal = subj["data"][trial_idx, ch_idx, :]
        label_val  = subj["labels"][trial_idx]

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 3))
            t = np.arange(len(raw_signal)) / 128.0
            ax.plot(t, raw_signal, linewidth=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (μV)")
            ax.set_title(f"EEG – {DEAP_ELECTRODES[ch_idx]} (Ch {ch_idx}) | Trial {trial_idx}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            arousal_val = label_val[1]
            valence_val = label_val[0]
            st.write(f"**Valence**: {valence_val:.1f} ({'High' if valence_val >= 5 else 'Low'})")
            st.write(f"**Arousal**: {arousal_val:.1f} ({'High' if arousal_val >= 5 else 'Low'})")
            st.metric("Dominance", f"{label_val[2]:.1f}")
            st.metric("Liking",    f"{label_val[3]:.1f}")

        st.markdown("---")
        # Label distribution
        st.subheader("Phân bố nhãn")
        all_labels = subj["labels"][:, :2]
        col_a, col_b = st.columns(2)
        for col_w, name, idx in [(col_a, "Arousal", 1), (col_b, "Valence", 0)]:
            with col_w:
                counts = pd.Series(
                    (all_labels[:, idx] >= 5).astype(int)
                ).value_counts().rename({0: "Low", 1: "High"})
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(counts, labels=counts.index, autopct="%1.0f%%",
                       colors=["#5b8dd9", "#e07b39"])
                ax.set_title(name)
                col_w.pyplot(fig)
                plt.close(fig)

        st.button("Tiếp tục → Preprocess", on_click=goto, args=("Preprocess",))


# ═══════════════════════════════════════════════════════════════════ #
#  PAGE: Preprocess (FFT)
# ═══════════════════════════════════════════════════════════════════ #
def page_preprocess():
    st.title("⚡ Preprocess – FFT Feature Extraction")

    if not st.session_state.subjects_raw:
        st.warning("⚠️ Chưa có dữ liệu. Hãy Load Data trước.")
        st.button("← Load Data", on_click=goto, args=("Load Data",))
        return

    st.markdown(f"""
    Trích xuất đặc trưng **FFT 5 dải tần** (sliding window) cho **{len(st.session_state.subjects_raw)}** subject(s).

    | Dải tần | Phạm vi (Hz) |
    |---------|-------------|
    | Theta       | 4 – 8   |
    | Alpha       | 8 – 12  |
    | Lower Beta  | 12 – 16 |
    | Upper Beta  | 16 – 25 |
    | Gamma       | 25 – 45 |
    """)

    col1, col2 = st.columns(2)
    window_size = col1.number_input("Window size (samples)", value=256, min_value=64, max_value=512, step=64)
    step_size   = col2.number_input("Step size (samples)", value=16, min_value=4, max_value=128, step=4)

    if st.button("🚀 Bắt đầu Preprocess", type="primary"):
        results = []
        bar = st.progress(0)
        status = st.empty()
        for i, (fname, subj) in enumerate(st.session_state.subjects_raw):
            status.text(f"⏳ Xử lý {fname} ({i+1}/{len(st.session_state.subjects_raw)})…")
            preprocessed = preprocess_subject_fft(
                subj, window_size=int(window_size), step_size=int(step_size)
            )
            results.append(preprocessed)
            bar.progress((i + 1) / len(st.session_state.subjects_raw))

        st.session_state.subjects_preprocessed = results
        status.text("✅ Hoàn thành!")
        st.success(f"Đã xử lý {len(results)} subject(s).")

    if st.session_state.subjects_preprocessed:
        st.markdown("---")
        st.subheader("📊 Kết quả Preprocess")
        preprocessed = st.session_state.subjects_preprocessed[0]
        n_windows = preprocessed.shape[0]
        st.write(f"**Subject 1** – số cửa sổ: `{n_windows}` | "
                 f"Features shape: `{preprocessed[0][0].shape}` (channels × bands)")

        # Show FFT feature heatmap for first window
        sample_features = preprocessed[0][0]  # (32, 5)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(sample_features, aspect="auto", cmap="viridis")
        ax.set_yticks(range(N_CHANNELS))
        ax.set_yticklabels(DEAP_ELECTRODES, fontsize=6)
        ax.set_xticks(range(N_FREQUENCIES))
        ax.set_xticklabels(MRMR_BAND_NAMES, fontsize=9)
        ax.set_title("FFT Feature Map – Window 0, Subject 1")
        plt.colorbar(im, ax=ax, label="Power (μV²/Hz)")
        st.pyplot(fig)
        plt.close(fig)

        st.button("Tiếp tục → MRMR", on_click=goto, args=("MRMR Selection",))


# ═══════════════════════════════════════════════════════════════════ #
#  PAGE: MRMR Channel Selection
# ═══════════════════════════════════════════════════════════════════ #
def page_mrmr():
    st.title("🔬 MRMR Channel Selection")

    if not st.session_state.subjects_preprocessed:
        st.warning("⚠️ Chưa có dữ liệu FFT. Hãy thực hiện Preprocess trước.")
        st.button("← Preprocess", on_click=goto, args=("Preprocess",))
        return

    classify_type = st.session_state.classify_type
    st.markdown(f"Nhãn phân loại: **{classify_type.capitalize()}**")

    K = st.slider("Số kênh MRMR (K)", min_value=5, max_value=32, value=MRMR_COMPONENTS, step=1)

    if st.button("🔬 Chạy MRMR", type="primary"):
        all_selected = []
        bar = st.progress(0)
        status = st.empty()
        all_subjects_pre = st.session_state.subjects_preprocessed

        for i, preprocessed in enumerate(all_subjects_pre):
            fname = st.session_state.subjects_raw[i][0] if i < len(st.session_state.subjects_raw) else f"Subject {i+1}"
            status.text(f"⏳ MRMR trên {fname} ({i+1}/{len(all_subjects_pre)})…")
            selected = run_mrmr_selection(preprocessed, classify_type=classify_type, K=K)
            all_selected.append(selected)
            bar.progress((i + 1) / len(all_subjects_pre))

        st.session_state.selected_channels = all_selected
        status.text("✅ Hoàn thành!")
        st.success(f"Đã chọn {K} kênh cho {len(all_selected)} subject(s).")

    if st.session_state.selected_channels:
        st.markdown("---")
        st.subheader("📊 Kênh được chọn")

        rows = []
        for i, ch_list in enumerate(st.session_state.selected_channels):
            fname = (st.session_state.subjects_raw[i][0]
                     if i < len(st.session_state.subjects_raw) else f"Sub {i+1}")
            ch_names = [DEAP_ELECTRODES[c] for c in ch_list if c < len(DEAP_ELECTRODES)]
            rows.append({"Subject": fname, "Selected channels": ", ".join(ch_names),
                         "Count": len(ch_list)})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Frequency of channel selection across subjects
        from collections import Counter
        all_flat = [ch for chl in st.session_state.selected_channels for ch in chl]
        counts = Counter(all_flat)
        ch_freq = pd.DataFrame(
            [(DEAP_ELECTRODES[k], v) for k, v in sorted(counts.items(), key=lambda x: -x[1])],
            columns=["Channel", "Frequency"],
        )
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(ch_freq["Channel"], ch_freq["Frequency"], color="#5b8dd9")
        ax.set_xlabel("EEG Channel")
        ax.set_ylabel("Số lần được chọn")
        ax.set_title("Tần suất lựa chọn kênh qua các subject")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        st.button("Tiếp tục → Train Model", on_click=goto, args=("Train Model",))


# ═══════════════════════════════════════════════════════════════════ #
#  PAGE: Train Model
# ═══════════════════════════════════════════════════════════════════ #
def page_train():
    st.title("🎓 Train Model")

    if not st.session_state.selected_channels:
        st.warning("⚠️ Chưa có kết quả MRMR. Hãy chạy MRMR Selection trước.")
        st.button("← MRMR Selection", on_click=goto, args=("MRMR Selection",))
        return

    classify_type = st.session_state.classify_type
    st.markdown(f"Nhãn phân loại: **{classify_type.capitalize()}**")

    col1, col2, col3 = st.columns(3)
    epochs    = col1.number_input("Số epochs", value=50, min_value=5, max_value=500, step=5)
    lr        = col2.number_input("Learning rate", value=0.001, min_value=1e-5, max_value=0.1,
                                   format="%.5f")
    batch_sz  = col3.number_input("Batch size", value=256, min_value=32, max_value=1024, step=32)
    dropout   = st.slider("Dropout rate", 0.0, 0.8, 0.5, 0.05)

    if st.button("🚀 Bắt đầu Training", type="primary"):
        with st.spinner("Đang chuẩn bị dataset…"):
            x_train_raw, y_train_raw, x_test_raw, y_test_raw = build_mrmr_dataset(
                st.session_state.subjects_preprocessed,
                st.session_state.selected_channels,
            )
            x_train, y_train_bin, x_test, y_test_bin = prepare_for_lstm(
                x_train_raw, x_test_raw, y_train_raw, y_test_raw,
                classify_type=classify_type,
            )

        st.write(f"Train: `{x_train.shape}` | Test: `{x_test.shape}`")

        train_ds = TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train_bin, dtype=torch.long)
        )
        test_ds = TensorDataset(
            torch.tensor(x_test), torch.tensor(y_test_bin, dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=int(batch_sz), shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=int(batch_sz), shuffle=False)

        seq_len = x_train.shape[1]
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model   = build_model("mrmr_lstm", seq_len=seq_len, dropout=dropout).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = 0.0

        # Progress UI
        epoch_bar   = st.progress(0)
        metric_placeholder = st.empty()
        chart_placeholder  = st.empty()

        for epoch in range(1, int(epochs) + 1):
            # Train
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                tr_loss    += loss.item() * len(yb)
                tr_correct += (logits.argmax(1) == yb).sum().item()
                tr_total   += len(yb)
            scheduler.step()

            # Eval
            model.eval()
            va_loss, va_correct, va_total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits  = model(xb)
                    loss    = criterion(logits, yb)
                    va_loss    += loss.item() * len(yb)
                    va_correct += (logits.argmax(1) == yb).sum().item()
                    va_total   += len(yb)

            tr_loss_ep = tr_loss / tr_total
            tr_acc_ep  = tr_correct / tr_total
            va_loss_ep = va_loss / va_total
            va_acc_ep  = va_correct / va_total

            history["train_loss"].append(tr_loss_ep)
            history["val_loss"].append(va_loss_ep)
            history["train_acc"].append(tr_acc_ep)
            history["val_acc"].append(va_acc_ep)

            epoch_bar.progress(epoch / int(epochs))
            metric_placeholder.markdown(
                f"**Epoch {epoch}/{int(epochs)}** &nbsp;|&nbsp; "
                f"Train Loss: `{tr_loss_ep:.4f}` Acc: `{tr_acc_ep:.3f}` &nbsp;|&nbsp; "
                f"Val Loss: `{va_loss_ep:.4f}` Acc: `{va_acc_ep:.3f}`"
            )

            if va_acc_ep > best_val_acc:
                best_val_acc = va_acc_ep
                st.session_state.trained_model   = model
                st.session_state.trained_seq_len = seq_len

        st.session_state.train_history = history
        st.session_state.x_train = x_train
        st.session_state.y_train = y_train_bin
        st.session_state.x_test  = x_test
        st.session_state.y_test  = y_test_bin

        st.success(f"✅ Hoàn thành! Best val accuracy: **{best_val_acc:.4f}**")

    # Show history chart
    if st.session_state.train_history:
        hist = st.session_state.train_history
        st.markdown("---")
        st.subheader("📈 Training History")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist["train_loss"], label="Train Loss")
        axes[0].plot(hist["val_loss"],   label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Loss")

        axes[1].plot(hist["train_acc"], label="Train Acc")
        axes[1].plot(hist["val_acc"],   label="Val Acc")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Accuracy")

        fig.suptitle(f"{st.session_state.classify_type.capitalize()} – MRMR BiLSTM")
        st.pyplot(fig)
        plt.close(fig)

        # Download checkpoint
        if st.session_state.trained_model is not None:
            buf = io.BytesIO()
            torch.save({
                "model":    st.session_state.trained_model.state_dict(),
                "seq_len":  st.session_state.trained_seq_len,
                "target":   st.session_state.classify_type,
                "history":  st.session_state.train_history,
            }, buf)
            buf.seek(0)
            st.download_button(
                "💾 Download model checkpoint (.pth)",
                data=buf,
                file_name=f"{st.session_state.classify_type}_mrmr_lstm.pth",
                mime="application/octet-stream",
            )

        st.button("Tiếp tục → Predict", on_click=goto, args=("Predict",))


# ═══════════════════════════════════════════════════════════════════ #
#  PAGE: Predict
# ═══════════════════════════════════════════════════════════════════ #
def page_predict():
    st.title("🔮 Predict")
    st.markdown("Dự đoán cảm xúc từ dữ liệu EEG mới.")

    # ── Model source ─────────────────────────────────────────────── #
    model_source = st.radio(
        "Nguồn mô hình",
        ["Mô hình vừa train", "Upload checkpoint (.pth)"],
        horizontal=True,
    )

    model = None
    classify_type = st.session_state.classify_type

    if model_source == "Mô hình vừa train":
        if st.session_state.trained_model is not None:
            model = st.session_state.trained_model
            st.success(f"✅ Đang dùng mô hình đã train ({classify_type})")
        else:
            st.warning("Chưa có mô hình – hãy train trước.")
    else:
        ckpt_file = st.file_uploader("Upload checkpoint (.pth)", type=["pth"])
        classify_type = st.selectbox("Nhãn của mô hình", ["arousal", "valence"])
        if ckpt_file:
            buf = io.BytesIO(ckpt_file.read())
            ckpt = torch.load(buf, map_location="cpu")
            seq_len = ckpt.get("seq_len", 100)
            model = build_model("mrmr_lstm", seq_len=seq_len)
            model.load_state_dict(ckpt["model"])
            st.success("✅ Checkpoint loaded.")

    st.markdown("---")
    st.subheader("📤 Upload dữ liệu để dự đoán")

    input_mode = st.radio(
        "Kiểu dữ liệu đầu vào",
        ["File đặc trưng MRMR (.npy)", "File .dat DEAP (xử lý trực tiếp)"],
        horizontal=True,
    )

    if input_mode == "File đặc trưng MRMR (.npy)":
        npy_file = st.file_uploader("Upload file .npy (shape: [n_samples, seq_len, 1])", type=["npy"])
        if npy_file and model is not None:
            features = np.load(io.BytesIO(npy_file.read()))
            st.write(f"Shape: `{features.shape}`")
            _run_prediction(model, features, classify_type)

    else:
        dat_file = st.file_uploader("Upload file .dat", type=["dat"])
        if dat_file and model is not None and st.session_state.selected_channels:
            subj = pickle.load(io.BytesIO(dat_file.read()), encoding="latin1")
            with st.spinner("Preprocessing…"):
                preprocessed = preprocess_subject_fft(subj)
                # Use first subject's selected channels as reference
                selected = st.session_state.selected_channels[0]
                x_r, y_r, x_t, y_t = build_mrmr_dataset([preprocessed], [selected])
                all_x = np.concatenate([x_r, x_t], axis=0)
                all_y = np.concatenate([y_r, y_t], axis=0)
                # For inference: fit scaler on full data (no train/test split needed)
                from sklearn.preprocessing import normalize as _normalize, StandardScaler as _SS
                all_x_norm = _normalize(all_x).astype(np.float32)
                scaler = _SS()
                all_x_scaled = scaler.fit_transform(all_x_norm).astype(np.float32)
                x_norm = all_x_scaled.reshape(all_x_scaled.shape[0], all_x_scaled.shape[1], 1)
                col = 0 if classify_type.lower() == "arousal" else 1
                y_bin = all_y[:, col]

            st.write(f"Shape sau xử lý: `{x_norm.shape}`")
            _run_prediction(model, x_norm, classify_type, true_labels=y_bin)

        elif dat_file and model is not None and not st.session_state.selected_channels:
            st.warning("⚠️ Chưa có kết quả MRMR channel selection. Hãy chạy MRMR Selection trước.")


def _run_prediction(model, x: np.ndarray, classify_type: str, true_labels=None):
    """Helper: run model on x and display results."""
    model.eval()
    device = next(model.parameters()).device

    # Ensure correct shape
    if x.ndim == 2:
        x = x[:, :, np.newaxis]  # (n, seq, 1)

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=1)

    labels_map = {0: "Low", 1: "High"}
    st.markdown("---")
    st.subheader("🔮 Kết quả dự đoán")

    col1, col2 = st.columns(2)
    counts = pd.Series(preds).value_counts().rename(labels_map)
    col1.metric(f"{classify_type.capitalize()} – Predicted (High)", int((preds == 1).sum()))
    col1.metric(f"{classify_type.capitalize()} – Predicted (Low)",  int((preds == 0).sum()))
    col2.bar_chart(counts)

    if true_labels is not None:
        correct = (preds == true_labels).mean()
        st.metric("Accuracy", f"{correct:.3%}")

    # Show first few probabilities
    df_pred = pd.DataFrame({
        "Sample":     range(len(preds)),
        "Prediction": [labels_map[p] for p in preds],
        "P(Low)":     probs[:, 0].round(4),
        "P(High)":    probs[:, 1].round(4),
    })
    if true_labels is not None:
        df_pred["True Label"] = [labels_map.get(int(l), str(l)) for l in true_labels]

    st.dataframe(df_pred.head(50), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════ #
#  Router
# ═══════════════════════════════════════════════════════════════════ #
PAGE_FUNCS = {
    "Home":           page_home,
    "Load Data":      page_load_data,
    "Preprocess":     page_preprocess,
    "MRMR Selection": page_mrmr,
    "Train Model":    page_train,
    "Predict":        page_predict,
}

PAGE_FUNCS[st.session_state.page]()

