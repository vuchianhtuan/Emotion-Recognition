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
import tempfile
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
    run_mrmr_global_selection,
    build_mrmr_dataset,
    prepare_for_lstm,
    BANDS,
    N_CHANNELS,
    N_FREQUENCIES,
    MRMR_COMPONENTS,
    TEST_SPLIT_MODULO,
)    
from src.preprocess import MRMR_BAND_NAMES, PreprocessConfig
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
        "subjects_raw":          [],
        "subjects_preprocessed": [],
        "selected_channels":     [],
        "x_train": None, "y_train": None,
        "x_test":  None, "y_test":  None,
        "train_history":         None,
        "trained_model":         None,
        "trained_seq_len":       None,
        "classify_type":         "arousal",
        
        # --- THÊM 4 DÒNG NÀY VÀO ---
        "pred_npy_results":      None,
        "pred_dat_results":      None,
        "pred_dat_n_trials":     None,
        "pred_dat_acc":          None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def goto(page: str):
    st.session_state.page = page
# ── Sidebar navigation ───────────────────────────────────`─────────── #
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
        with st.spinner("Đang chạy MRMR Global trên tất cả subjects..."):
            all_subjects_pre = st.session_state.subjects_preprocessed
            global_selected = run_mrmr_global_selection(all_subjects_pre, classify_type=classify_type, K=K)
            
        # Lưu cùng channels cho tất cả subjects
        st.session_state.selected_channels = [global_selected] * len(all_subjects_pre)
        st.success(f"✅ Đã chọn {K} kênh Global cho tất cả {len(all_subjects_pre)} subjects.")
        st.info(f"📋 Global channels: {', '.join([DEAP_ELECTRODES[c] for c in global_selected])}")

    if st.session_state.selected_channels:
        st.markdown("---")
        st.subheader("📊 Kênh được chọn (Global)")

        # Tất cả subjects có cùng channels
        global_channels = st.session_state.selected_channels[0]
        ch_names = [DEAP_ELECTRODES[c] for c in global_channels if c < len(DEAP_ELECTRODES)]
        
        st.markdown(f"**Selected {len(global_channels)} channels for all subjects:**")
        st.markdown(f"**Names:** {', '.join(ch_names)}")
        st.markdown(f"**Indices:** {global_channels}")

        # Hiển thị bảng cho tất cả subjects
        rows = []
        for i in range(len(st.session_state.subjects_preprocessed)):
            fname = (st.session_state.subjects_raw[i][0]
                     if i < len(st.session_state.subjects_raw) else f"Subject {i+1}")
            rows.append({"Subject": fname, "Selected channels": ", ".join(ch_names),
                         "Count": len(global_channels)})

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ─── Download channels file ─────────────────────────────── #
        st.markdown("---")
        st.subheader("💾 Xuất file kênh được chọn")
        
        channels_df = pd.DataFrame({
            "channels": global_channels,
            "channel_names": ch_names,
        })
        
        csv_buffer = channels_df.to_csv(index=False)
        st.download_button(
            label="📥 Tải file kênh (CSV)",
            data=csv_buffer,
            file_name=f"mrmr_global_channels_{classify_type}.csv",
            mime="text/csv",
            key="download_channels",
        )
        
        st.info(f"📋 Global channels ({len(global_channels)}): {', '.join(ch_names)}")

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
        model   = build_model("mrmr_lstm", seq_len=seq_len, dropout=dropout, input_size=1).to(device)

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
            try:
                checkpoint = torch.load(io.BytesIO(ckpt_file.read()), map_location="cpu")
                model = build_model("mrmr_lstm", input_size=1)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                st.success("✅ Checkpoint `.pth` loaded.")
                classify_type = checkpoint.get("target", classify_type)
                if "target" in checkpoint:
                    st.info(f"Model target loaded: {classify_type}")
            except Exception as exc:
                st.error(f"Không thể load checkpoint: {exc}")

    st.markdown("---")
    st.subheader("📤 Upload dữ liệu để dự đoán")

    # ─── Upload channels file (mới) ──────────────────────────────── #
    st.markdown("**📋 File kênh được chọn (Optional)**")
    st.info("Tải file kênh từ MRMR Selection để áp dụng cùng kênh được dùng khi training")
    
    channels_file = st.file_uploader("Upload file kênh CSV (mrmr_global_channels_*.csv)", type=["csv"], key="upload_channels")
    selected_channels_from_file = None
    
    if channels_file:
        channels_df = pd.read_csv(channels_file)
        if "channels" in channels_df.columns:
            selected_channels_from_file = channels_df["channels"].tolist()
            st.success(f"✅ Loaded {len(selected_channels_from_file)} channels: {', '.join([DEAP_ELECTRODES[int(c)] for c in selected_channels_from_file if int(c) < len(DEAP_ELECTRODES)])}")
        else:
            st.warning("⚠️ File không đúng format. Cần có cột 'channels'")

    st.markdown("---")

    input_mode = st.radio(
        "Kiểu dữ liệu đầu vào",
        ["File đặc trưng MRMR (.npy)", "File .dat DEAP (xử lý trực tiếp)"],
        horizontal=True,
    )

    if input_mode == "File đặc trưng MRMR (.npy)":
        npy_file = st.file_uploader("Upload file .npy (shape: [n_samples, seq_len, 1])", type=["npy"])
        if npy_file and model is not None:
            # 1. NÚT BẤM DỰ ĐOÁN CHO FILE NPY
            if st.button("🚀 Bắt đầu dự đoán .npy", type="primary"):
                with st.spinner("Đang dự đoán..."):
                    features = np.load(io.BytesIO(npy_file.read()))
                    probs, preds = _predict_model(model, features)
                    # Lưu vào session_state
                    st.session_state.pred_npy_results = {
                        "features_shape": features.shape,
                        "probs": probs,
                        "preds": preds
                    }
                        
            # 2. HIỂN THỊ KẾT QUẢ CHO FILE NPY
            if st.session_state.get("pred_npy_results") is not None:
                res = st.session_state.pred_npy_results
                st.write(f"Shape: `{res['features_shape']}`")
                st.metric("Samples", res['features_shape'][0])
                
                # Việc kéo slider ở đây sẽ không làm code phía trên chạy lại
                sample_idx = st.slider("Chọn sample index", 1, res['features_shape'][0], 1)
                selected = sample_idx - 1
                
                st.markdown("---")
                st.subheader("🔎 Sample prediction")
                st.write({
                    "Sample index": selected,
                    "Prediction": _label_name(res['preds'][selected]),
                    "P(Low)": float(res['probs'][selected, 0]),
                    "P(High)": float(res['probs'][selected, 1]),
                })
                st.markdown("---")
                st.write("Lưu ý: file .npy không chứa nhãn thực tế, chỉ hiển thị dự đoán từng sample.")

    else:
        dat_file = st.file_uploader("Upload file .dat", type=["dat"], key="upload_dat")
        if dat_file and model is not None:
            # Xác định channels để sử dụng
            if selected_channels_from_file:
                selected = selected_channels_from_file
                st.info(f"📌 Sử dụng {len(selected)} kênh từ file upload")
            elif st.session_state.selected_channels:
                selected = st.session_state.selected_channels[0]
                st.info(f"📌 Sử dụng {len(selected)} kênh từ MRMR Selection hiện tại")
            else:
                st.warning("⚠️ Chưa có thông tin kênh. Hãy tải file kênh hoặc chạy MRMR Selection trước.")
                return
            
            # 1. NÚT BẤM TIỀN XỬ LÝ & DỰ ĐOÁN CHO FILE DAT
            if st.button("🚀 Tiền xử lý & Dự đoán file .dat", type="primary"):
                with st.spinner("Đang Preprocessing & Predicting (Có thể mất thời gian)…"):
                    subj = pickle.load(io.BytesIO(dat_file.read()), encoding="latin1")
                    preprocessed = preprocess_subject_fft(subj)
                    x_r, y_r, x_t, y_t = build_mrmr_dataset([preprocessed], [selected])
                    all_x = np.concatenate([x_r, x_t], axis=0)
                    all_y = np.concatenate([y_r, y_t], axis=0)
                    
                    from sklearn.preprocessing import normalize as _normalize, StandardScaler as _SS
                    all_x_norm = _normalize(all_x).astype(np.float32)
                    scaler = _SS()
                    all_x_scaled = scaler.fit_transform(all_x_norm).astype(np.float32)
                    x_norm = all_x_scaled.reshape(all_x_scaled.shape[0], all_x_scaled.shape[1], 1)
                    col = 0 if classify_type.lower() == "arousal" else 1
                    y_bin = all_y[:, col]

                    trial_ids, window_ids, start_samples = _build_window_metadata(subj)
                    train_mask = np.array([i % TEST_SPLIT_MODULO != 0 for i in range(len(trial_ids))])
                    trial_ids = np.concatenate([trial_ids[train_mask], trial_ids[~train_mask]])
                    window_ids = np.concatenate([window_ids[train_mask], window_ids[~train_mask]])
                    start_samples = np.concatenate([start_samples[train_mask], start_samples[~train_mask]])

                    probs, preds = _predict_model(model, x_norm)
                    total_acc = (preds == y_bin).mean()

                    # Build results dataframe
                    df_results = pd.DataFrame({
                        "trial": trial_ids,
                        "window": window_ids,
                        "start_sample": start_samples,
                        "prediction": [ _label_name(p) for p in preds ],
                        "true_label": [ _label_name(y) for y in y_bin ],
                        "p_low": probs[:, 0],
                        "p_high": probs[:, 1],
                    })

                    # Lưu vào session_state để tái sử dụng
                    st.session_state.pred_dat_results = df_results
                    st.session_state.pred_dat_n_trials = int(subj["data"].shape[0])
                    st.session_state.pred_dat_acc = total_acc

            # 2. HIỂN THỊ KẾT QUẢ TỪ SESSION_STATE VÀ KHỞI TẠO SLIDERS CHO FILE DAT
            if st.session_state.get("pred_dat_results") is not None:
                df_results = st.session_state.pred_dat_results
                total_acc = st.session_state.pred_dat_acc
                n_trials = st.session_state.pred_dat_n_trials

                st.write(f"Số mẫu sau xử lý: `{len(df_results)}`")
                st.metric("Overall Accuracy (Tất cả trials)", f"{total_acc:.3%}")
                
                st.markdown("---")
                st.subheader("📌 Dự đoán theo Trial")

                # Từ đây kéo slider thì Streamlit chỉ chạy lại từ đây trở xuống
                trial_choice = st.slider("Chọn trial để xem chi tiết", 1, n_trials, 1)
                trial_mask = df_results["trial"] == (trial_choice - 1)
                trial_df = df_results[trial_mask].reset_index(drop=True)

                if not trial_df.empty:
                    # ─── TÍNH TOÁN KẾT QUẢ TỔNG QUAN CỦA TRIAL ───
                    true_label = trial_df["true_label"].iloc[0] # Nhãn thực tế của cả trial
                    total_samples = len(trial_df)
                    
                    # Đếm số lượng dự đoán High / Low
                    preds_counts = trial_df["prediction"].value_counts()
                    pred_high = preds_counts.get("High", 0)
                    pred_low = preds_counts.get("Low", 0)
                    
                    # Dự đoán của Trial (Majority Vote)
                    trial_prediction = "High" if pred_high >= pred_low else "Low"
                    is_correct = (trial_prediction == true_label)
                    
                    # Đếm chính xác số lượng sample đoán đúng
                    correct_samples = (trial_df["prediction"] == trial_df["true_label"]).sum()
                    
                    # Tỉ lệ sample dự đoán đúng trong Trial này
                    trial_acc = correct_samples / total_samples if total_samples > 0 else 0.0

                    # ─── HIỂN THỊ KẾT QUẢ TỔNG QUAN BẰNG COLUMNS ───
                    st.markdown(f"**📊 Kết quả tổng hợp Trial {trial_choice}**")
                    
                    # Đặt màu sắc thông báo
                    if is_correct:
                        st.success(f"✅ Dự đoán Trial **CHÍNH XÁC**! (Dự đoán: {trial_prediction} | Thực tế: {true_label})")
                    else:
                        st.error(f"❌ Dự đoán Trial **SAI**! (Dự đoán: {trial_prediction} | Thực tế: {true_label})")

                    # Chia làm 4 cột để hiển thị rõ Số sample đúng / Tổng số
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nhãn thực tế", true_label)
                    col2.metric("Dự đoán (Đa số)", trial_prediction)
                    col3.metric("Sample dự đoán đúng", f"{correct_samples} / {total_samples}")
                    col4.metric("Tỉ lệ (Accuracy)", f"{trial_acc:.2%}")

                    st.write(f"*Chi tiết bầu chọn:* Có **{pred_high}** mẫu đoán `High` và **{pred_low}** mẫu đoán `Low`.")

                    # ─── CHI TIẾT TỪNG SAMPLE (Dành cho việc phân tích sâu) ───
                    st.markdown("---")
                    st.markdown("**🔎 Xem chi tiết từng Sample (Window) trong Trial**")
                    sample_choice = st.slider("Chọn sample", 1, len(trial_df), 1)
                    selected_row = trial_df.iloc[sample_choice - 1]
                    
                    st.write({
                        "Trial": int(selected_row["trial"]) + 1,
                        "Window index": int(selected_row["window"]),
                        "Start sample": int(selected_row["start_sample"]),
                        "Prediction": selected_row["prediction"],
                        "True label": selected_row["true_label"],
                        "P(Low)": float(selected_row["p_low"]),
                        "P(High)": float(selected_row["p_high"]),
                    })

                    with st.expander(f"Hiển thị bảng dữ liệu của trial {trial_choice}"):
                        st.dataframe(trial_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Không có window nào cho trial đã chọn.")
def _run_prediction(model, x: np.ndarray, classify_type: str, true_labels=None):
    """Helper: run model on x and display results."""
    # Ensure correct shape for models expecting (n, seq, 1)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]

    model.eval()
    x_tensor = torch.from_numpy(x.astype(np.float32))
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    x_tensor = x_tensor.to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)

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


def _predict_model(model, x: np.ndarray):
    """Run model inference and return probabilities and class predictions."""
    if x.ndim == 2:
        x = x[:, :, np.newaxis]

    model.eval()
    x_tensor = torch.from_numpy(x.astype(np.float32))
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    x_tensor = x_tensor.to(device)
    
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)
    return probs, preds

def _build_window_metadata(subject):
    """Build trial and window metadata for DEAP .dat data."""
    cfg = PreprocessConfig()
    trial_ids = []
    window_ids = []
    start_samples = []

    for trial_idx in range(subject["data"].shape[0]):
        trial = np.asarray(subject["data"][trial_idx], dtype=np.float32)[: cfg.n_eeg_channels]
        window_idx = 0
        start = 0
        while start + cfg.window_size <= trial.shape[1]:
            trial_ids.append(trial_idx)
            window_ids.append(window_idx)
            start_samples.append(start)
            window_idx += 1
            start += cfg.step_size

    return (
        np.array(trial_ids, dtype=np.int32),
        np.array(window_ids, dtype=np.int32),
        np.array(start_samples, dtype=np.int32),
    )


def _label_name(value):
    return "High" if int(value) == 1 else "Low"


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

