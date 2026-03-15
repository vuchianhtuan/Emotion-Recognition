"""
app/main.py
-----------
Streamlit Web Dashboard – EEG Emotion Recognition

Cách chạy:
    streamlit run app/main.py
"""

import os
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Thêm thư mục gốc project vào sys.path để import src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models  import build_model
from src.utils   import load_checkpoint


# ------------------------------------------------------------------ #
#  Cấu hình trang
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide",
)


# ------------------------------------------------------------------ #
#  Sidebar – Tuỳ chọn
# ------------------------------------------------------------------ #
st.sidebar.title("⚙️ Cài đặt")

target = st.sidebar.selectbox("Nhãn dự đoán", ["valence", "arousal"])
arch   = st.sidebar.selectbox("Kiến trúc",   ["lstm", "cnn", "transformer"])

ckpt_path = st.sidebar.text_input(
    "Đường dẫn checkpoint (.pth)",
    value=f"models/{target}_{arch}_v1.pth",
)


# ------------------------------------------------------------------ #
#  Load mô hình
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model(architecture: str, checkpoint: str):
    model = build_model(architecture)
    if os.path.exists(checkpoint):
        epoch = load_checkpoint(model, checkpoint)
        st.sidebar.success(f"✅ Checkpoint epoch {epoch}")
    else:
        st.sidebar.warning("⚠️ Chưa có checkpoint – dùng trọng số ngẫu nhiên.")
    model.eval()
    return model


model = load_model(arch, ckpt_path)


# ------------------------------------------------------------------ #
#  Giao diện chính
# ------------------------------------------------------------------ #
st.title("🧠 EEG Emotion Recognition Dashboard")
st.markdown("""
Dashboard dự đoán **Valence / Arousal** từ tín hiệu EEG (DEAP dataset).
Upload file đặc trưng `.npy` (shape `32 × 4`) để nhận kết quả phân loại.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload đặc trưng EEG")
    uploaded = st.file_uploader("File numpy (.npy) shape (32, 4)", type=["npy"])

    if uploaded:
        features = np.load(uploaded)               # (32, 4)
        st.write(f"Shape: `{features.shape}`")

        fig, ax = plt.subplots(figsize=(6, 3))
        im = ax.imshow(features, aspect="auto", cmap="viridis")
        ax.set_xlabel("Dải tần (θ α β γ)")
        ax.set_ylabel("Channel EEG")
        ax.set_title("Feature Map")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

with col2:
    st.subheader("🔮 Kết quả dự đoán")
    if uploaded:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 32, 4)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=-1).squeeze().numpy()

        label_names = ["Low", "High"]
        predicted   = label_names[probs.argmax()]

        st.metric(
            label=f"{target.capitalize()} dự đoán",
            value=predicted,
            delta=f"Confidence: {probs.max():.1%}",
        )

        st.bar_chart({"Low": probs[0], "High": probs[1]})
    else:
        st.info("← Upload file đặc trưng để dự đoán.")
