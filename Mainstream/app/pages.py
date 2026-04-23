"""Page functions for the Streamlit application."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import threading
import io
import os
import pickle
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.mrmr_selection import (
    MRMR_BAND_NAMES,
    N_FREQUENCIES,
    N_CHANNELS,
    TEST_SPLIT_MODULO,
    extract_subject_fft_windows,
)

from app.config import BATCH_SIZE, LR, DEBUG_MODE, TEST_DATA_DIR, DEAP_ELECTRODES, TARGETS
from app.state_management import goto, trigger_scroll_if_needed, on_file_link_click, snapshot_file_manager
from app.ui_helpers import target_label, channel_name
from app.data_io import (
    store_raw_data,
    store_processed_data,
    store_mrmr_result,
    store_model_result,
    selection_to_download_bytes,
)
from app.data_normalization import get_raw_records, get_processed_records, load_selected_channels
from app.ui_components import layout_with_manager
from app.model_utils import (
    run_mrmr_task,
    fit_model_for_target,
    predict_target,
)
from app.data_io import normalize_raw_record


def page_home() -> None:
    """Home page."""
    st.title("🧠 EEG Emotion Recognition")
    st.subheader("Nhận diện cảm xúc từ tín hiệu EEG – DEAP Dataset")
    st.markdown(
        """
        <div style="background:#eaf5ff;padding:18px;border-radius:10px;border:1px solid #cce0ff">
        <p>Ứng dụng này hỗ trợ xử lý song song cho <b>Arousal</b> và <b>Valence</b>,
        đồng thời quản lý toàn bộ dữ liệu qua <b>Virtual File Manager</b> ở bên phải.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("1. Load Data\nUpload file .dat từ DEAP dataset")
        st.info("4. Train Model\nTự chạy MRMR global và train theo nhãn")
    with col2:
        st.success("2. Preprocess\nTrích xuất FFT 5 dải tần")
        st.success("5. Download Model\nLưu checkpoint kèm channels + scaler")
    with col3:
        st.warning("3. Explore\nXem trước tín hiệu EEG")
        st.warning("6. Predict\nChạy suy luận song song")


def page_load_data() -> None:
    """Load Data page."""
    def render_main() -> None:
        st.title("📤 Load DEAP Data")
        st.markdown("Upload một hoặc nhiều file .dat từ DEAP dataset.")

        uploaded = st.file_uploader(
            "Chọn file .dat",
            type=["dat"],
            accept_multiple_files=True,
            key="page_load_data_uploader",
        )

        if st.button("🔄 Load Data", type="primary"):
            if not uploaded:
                st.warning("Hãy chọn ít nhất một file .dat.")
            else:
                with st.status("Đang nạp dữ liệu...", expanded=True) as status:
                    store_raw_data(uploaded)
                    status.update(label=f"Đã load {len(uploaded)} file .dat", state="complete")

        if DEBUG_MODE:
            with st.expander("🛠 Chế độ Kiểm thử (Load siêu tốc trực tiếp từ Server)", expanded=True):
                st.caption(f"Hãy tạo thư mục `{TEST_DATA_DIR}` cùng cấp với code và ném các file .dat vào đó.")
                if os.path.exists(TEST_DATA_DIR):
                    test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.dat')]
                    if test_files:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            selected_test_file = st.selectbox("Chọn file test:", test_files, label_visibility="collapsed")
                        with col_b:
                            if st.button("⚡ Nạp file này", use_container_width=True):
                                class MockFile:
                                    def __init__(self, filepath, filename):
                                        self.name = filename
                                        self.filepath = filepath
                                    def read(self):
                                        with open(self.filepath, "rb") as f:
                                            return f.read()
                                
                                mock_file = MockFile(os.path.join(TEST_DATA_DIR, selected_test_file), selected_test_file)
                                with st.status(f"Đang đọc {selected_test_file} từ ổ cứng server...", expanded=True) as status:
                                    store_raw_data([mock_file])
                                    status.update(label=f"Đã load {selected_test_file} siêu tốc!", state="complete")
                                st.rerun() 
                    else:
                        st.info(f"Thư mục `{TEST_DATA_DIR}` đang trống.")
                else:
                    st.warning(f"Chưa tìm thấy thư mục `{TEST_DATA_DIR}` trên server. Hãy tạo nó!")

        raw_records = get_raw_records()
        if raw_records:
            preview_name_key = "dashboard_load_data_selected_name"
            record_names = [record["name"] for record in raw_records]
            options = ["None"] + record_names 
            current_name = st.session_state.get(preview_name_key, "None")
            if current_name not in options:
                current_name = "None"
                st.session_state[preview_name_key] = current_name

            st.markdown("<div id='preview-section'></div>", unsafe_allow_html=True)
            st.markdown("---")
            
            header_col, action_col, _ = st.columns([2.2, 1.5, 6.3], gap="small")
            with header_col:
                st.markdown("<h3 style='margin-top:-8px; white-space: nowrap;'>Preview dữ liệu:</h3>", unsafe_allow_html=True)
            with action_col:
                with st.popover(current_name):
                    st.radio(
                        "File .dat",
                        options, 
                        index=options.index(current_name),
                        key=preview_name_key,
                        label_visibility="collapsed",
                    )

            if current_name != "None":
                current = next(record for record in raw_records if record["name"] == current_name)
                subject = current["subject"]
                st.write(
                    f"**File**: {current['name']} | Trials: {subject['data'].shape[0]} | Channels: {subject['data'].shape[1]} | Samples: {subject['data'].shape[2]}"
                )
                trial_idx = st.slider("Chọn Trial", 0, subject["data"].shape[0] - 1, 0)
                ch_idx = st.slider("Chọn kênh EEG", 0, N_CHANNELS - 1, 0)
                raw_signal = subject["data"][trial_idx, ch_idx, :]
                label_val = subject["labels"][trial_idx]

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    t = np.arange(len(raw_signal)) / 128.0
                    ax.plot(t, raw_signal, linewidth=0.7)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude (μV)")
                    ax.set_title(f"EEG – {channel_name(ch_idx)} | Trial {trial_idx}")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
                with col_b:
                    st.write(f"**Valence**: {label_val[0]:.1f}")
                    st.write(f"**Arousal**: {label_val[1]:.1f}")
                    st.metric("Dominance", f"{label_val[2]:.1f}")
                    st.metric("Liking", f"{label_val[3]:.1f}")

                st.markdown("---")
                st.subheader("Phân bố nhãn")
                label_frame = subject["labels"][:, :2]
                c1, c2 = st.columns(2)
                for container, name, idx in ((c1, "Valence", 0), (c2, "Arousal", 1)):
                    with container:
                        counts = pd.Series((label_frame[:, idx] >= 5).astype(int)).value_counts().rename({0: "Low", 1: "High"})
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.pie(counts, labels=counts.index, autopct="%1.0f%%", colors=["#5b8dd9", "#e07b39"])
                        ax.set_title(name)
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                st.info("Vui lòng chọn một file phía trên để xem trước dữ liệu.")

        st.button("Tiếp tục → Preprocess", on_click=goto, args=("Preprocess",))

    layout_with_manager(render_main)
    trigger_scroll_if_needed()


def page_preprocess() -> None:
    """Preprocess page."""
    def render_main() -> None:
        st.title("⚡ Preprocess – FFT Feature Extraction")
        raw_records = get_raw_records()
        if not raw_records:
            st.warning("Chưa có dữ liệu. Hãy load Data trước.")
            st.button("← Load Data", on_click=goto, args=("Load Data",))
            return

        st.markdown(
            f"Trích xuất đặc trưng FFT 5 dải tần cho **{len(raw_records)}** subject(s)."
        )
        st.table(pd.DataFrame({"Band": MRMR_BAND_NAMES, "Range (Hz)": ["4-8", "8-12", "12-16", "16-25", "25-45"]}))

        col_a, col_b = st.columns(2)
        window_size = col_a.number_input("Window size (samples)", value=256, min_value=64, max_value=512, step=64)
        step_size = col_b.number_input("Step size (samples)", value=16, min_value=4, max_value=128, step=4)

        if st.button("🚀 Bắt đầu Preprocess", type="primary"):
            processed = [None] * len(raw_records)
            progress = st.progress(0)
            status_text = st.empty()
            
            status_text.info(f"Đang phân bổ tác vụ FFT lên toàn bộ CPU cores (Đa tiến trình)...")
            
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_idx = {
                    executor.submit(
                        extract_subject_fft_windows, 
                        record["subject"], 
                        window_size=int(window_size), 
                        step_size=int(step_size)
                    ): (idx, record) 
                    for idx, record in enumerate(raw_records)
                }
                
                completed = 0
                for future in as_completed(future_to_idx):
                    idx, record = future_to_idx[future]
                    try:
                        preprocessed = future.result()
                        from app.data_normalization import normalize_processed_record
                        processed[idx] = normalize_processed_record(
                            record["name"], preprocessed, 
                            {"window_size": int(window_size), "step_size": int(step_size)}
                        )
                    except Exception as exc:
                        st.error(f"Lỗi khi xử lý {record['name']}: {exc}")
                        
                    completed += 1
                    status_text.info(f"Đang xử lý song song... Hoàn thành {completed}/{len(raw_records)} file.")
                    progress.progress(completed / len(raw_records))
            
            from app.state_management import file_manager
            file_manager()["processed_data"] = [p for p in processed if p is not None]
            status_text.success("Preprocess hoàn tát.")
            st.success(f"Đã trích xuất FFT cho {len(file_manager()['processed_data'])} subject(s).")

        processed_records = get_processed_records()
        if processed_records:
            preview_name_key = "dashboard_preprocess_selected_name"
            record_names = [record["name"] for record in processed_records]
            options = ["None"] + record_names
            current_name = st.session_state.get(preview_name_key, "None")
            
            if current_name not in options:
                current_name = "None"
                st.session_state[preview_name_key] = current_name

            st.markdown("<div id='preview-section'></div>", unsafe_allow_html=True)
            st.markdown("---")
            
            header_col, action_col, _ = st.columns([2.2, 1.5, 6.3], gap="small")
            with header_col:
                st.markdown("<h3 style='margin-top:-8px; white-space: nowrap;'>Preview kết quả:</h3>", unsafe_allow_html=True)
            with action_col:
                with st.popover(current_name):
                    st.radio(
                        "File .dat",
                        options,
                        index=options.index(current_name),
                        key=preview_name_key,
                        label_visibility="collapsed",
                    )

            if current_name != "None":
                current_record = next(record for record in processed_records if record["name"] == current_name)
                current_data = current_record["data"]
                num_windows = current_data.shape[0]

                st.write(
                    f"**File**: {current_record['name']} | Tổng số Windows: `{num_windows}` | Feature shape: `{current_data[0][0].shape}`"
                )
                
                st.write("") 
                col_lbl, col_inp, col_cap = st.columns([2.5, 1.5, 4.5], gap="small")
                
                with col_lbl:
                    st.markdown("<p style='margin-top: 6px; font-weight: 500;'>Chọn Window (Cửa sổ trượt):</p>", unsafe_allow_html=True)
                
                with col_inp:
                    window_idx = st.number_input(
                        "Chọn Window", 
                        min_value=0, 
                        max_value=num_windows - 1, 
                        value=0, 
                        step=1, 
                        label_visibility="collapsed" 
                    )
                
                with col_cap:
                    st.markdown(f"<p style='margin-top: 8px; font-size: 0.85em; color: #666;'>(Nhập từ 0 đến {num_windows - 1})</p>", unsafe_allow_html=True)
                st.write("") 
                
                sample_features = current_data[window_idx][0]
                label_bin = current_data[window_idx][1]

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    im = ax.imshow(sample_features, aspect="auto", cmap="viridis")
                    ax.set_yticks(range(sample_features.shape[0]))
                    ax.set_yticklabels([channel_name(i) for i in range(sample_features.shape[0])], fontsize=6)
                    ax.set_xticks(range(sample_features.shape[1]))
                    ax.set_xticklabels(MRMR_BAND_NAMES, fontsize=9)
                    ax.set_title(f"FFT Feature Map – Window {window_idx}")
                    plt.colorbar(im, ax=ax, label="Power")
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col_b:
                    st.markdown("**Nhãn nhị phân (Binary Labels)**")
                    st.write(f"**Valence**: {'High (1)' if label_bin[0] == 1 else 'Low (0)'}")
                    st.write(f"**Arousal**: {'High (1)' if label_bin[1] == 1 else 'Low (0)'}")
            else:
                st.info("Vui lòng chọn một file phía trên để xem trước biểu đồ FFT.")

        st.button("Tiếp tục → Train Model", on_click=goto, args=("Train Model",))

    layout_with_manager(render_main)
    trigger_scroll_if_needed()


def page_train() -> None:
    """Train Model page."""
    def render_main() -> None:
        st.title("🎓 Train Model")
        processed_records = get_processed_records()
        if not processed_records:
            st.warning("Chưa có dữ liệu FFT. Hãy preprocess trước.")
            st.button("← Preprocess", on_click=goto, args=("Preprocess",))
            return

        target_choices = {target: st.checkbox(target_label(target), value=True, key=f"train_choice_{target}") for target in TARGETS}
        selected_targets = [target for target, enabled in target_choices.items() if enabled]
        if not selected_targets:
            st.warning("Hãy chọn ít nhất một nhãn để train.")
            return

        with st.expander("⚙️ Advanced Hyperparameters (Epochs, LR, Batch Size, Dropout)", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            epochs = int(col_a.number_input("Epochs", value=50, min_value=1, max_value=500, step=5))
            lr = float(col_b.number_input("Learning rate", value=LR, min_value=1e-5, max_value=0.1, format="%.5f"))
            batch_size = int(col_c.number_input("Batch size", value=BATCH_SIZE, min_value=32, max_value=1024, step=32))
            dropout = float(st.slider("Dropout", 0.0, 0.8, 0.5, 0.05))
        
        from src.mrmr_selection import MRMR_COMPONENTS
        k_value = int(st.slider("Số kênh MRMR (K)", min_value=5, max_value=32, value=MRMR_COMPONENTS, step=1))

        if "training_progress_state" not in st.session_state.runtime:
            st.session_state.runtime["training_progress_state"] = {}

        start_train = st.button("🚀 Bắt đầu Training", type="primary")

        mrmr_container = st.container()
        
        st.markdown("<br>", unsafe_allow_html=True)
        prog_col_arousal, prog_col_valence = st.columns(2)
        progress_ui = {
            "arousal": {"bar": prog_col_arousal.empty(), "text": prog_col_arousal.empty()},
            "valence": {"bar": prog_col_valence.empty(), "text": prog_col_valence.empty()}
        }

        if start_train:
            st.session_state.runtime["mrmr_results"] = {}
            st.session_state.runtime["training_progress_state"] = {}
            st.session_state.runtime["training_results"] = {}

            ctx = get_script_run_ctx()

            with st.spinner("Đang chạy MRMR global song song..."):
                mrmr_results = {}
                channels_map = {}
                
                def thread_safe_mrmr(tgt):
                    add_script_run_ctx(threading.current_thread(), ctx) 
                    res = run_mrmr_task(tgt, processed_records, k_value)
                    store_mrmr_result(tgt, res["channels"], source="computed", name=res["file_name"])
                    return tgt, res

                with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                    futures = [executor.submit(thread_safe_mrmr, target) for target in selected_targets]
                    for future in as_completed(futures):
                        tgt, mrmr_result = future.result()
                        channels_map[tgt] = mrmr_result["channels"]
                        mrmr_results[tgt] = mrmr_result
                
                st.session_state.runtime["mrmr_results"] = mrmr_results

            with mrmr_container:
                st.markdown("**Kết quả MRMR:**")
                for target in ["arousal", "valence"]:
                    if target in channels_map:
                        ch_names = [channel_name(c) for c in channels_map[target]]
                        st.markdown(f"**{target_label(target)}**: {', '.join(ch_names)}")
                st.markdown("---")

            results: dict[str, dict[str, Any]] = {}
            
            def thread_safe_train(tgt):
                add_script_run_ctx(threading.current_thread(), ctx)
                
                def _on_epoch(epoch: int, total: int, tr_loss: float, tr_acc: float, va_loss: float, va_acc: float) -> None:
                    per_target = epoch / max(total, 1)
                    prog_text = (
                        f"**{target_label(tgt)} - Epoch {epoch}/{total}** | "
                        f"Train Loss: `{tr_loss:.4f}` Acc: `{tr_acc:.3f}` | "
                        f"Val Loss: `{va_loss:.4f}` Acc: `{va_acc:.3f}`"
                    )
                    progress_ui[tgt]["bar"].progress(per_target)
                    progress_ui[tgt]["text"].markdown(prog_text)
                    
                    st.session_state.runtime["training_progress_state"][tgt] = {
                        "progress": per_target,
                        "text": prog_text
                    }

                res = fit_model_for_target(
                    tgt,
                    processed_records,
                    channels_map[tgt],
                    epochs,
                    lr,
                    batch_size,
                    dropout,
                    progress_callback=_on_epoch,
                )
                store_model_result(tgt, res["model"], res["checkpoint"], source="trained")
                return tgt, res

            with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                futures = [executor.submit(thread_safe_train, target) for target in selected_targets]
                for future in as_completed(futures):
                    tgt, result = future.result()
                    results[tgt] = result
            
            st.session_state.runtime["training_results"] = results
            st.rerun()

        else:
            mrmr_results = st.session_state.runtime.get("mrmr_results", {})
            if mrmr_results:
                with mrmr_container:
                    st.markdown("**Kết quả MRMR:**")
                    for target in ["arousal", "valence"]: 
                        if target in mrmr_results:
                            ch_names = [channel_name(c) for c in mrmr_results[target]["channels"]]
                            st.markdown(f"**{target_label(target)}**: {', '.join(ch_names)}")
                    st.markdown("---")

            state_prog = st.session_state.runtime.get("training_progress_state", {})
            for target in ["arousal", "valence"]:
                if target in state_prog:
                    progress_ui[target]["bar"].progress(state_prog[target]["progress"])
                    progress_ui[target]["text"].markdown(state_prog[target]["text"])

        training_results = st.session_state.runtime.get("training_results", {})
        if training_results:
            st.markdown("---")
            for target, result in training_results.items():
                st.subheader(f"{target_label(target)}")
                history = result["history"]
                st.write(f"Best val accuracy: `{result['best_val_acc']:.4f}`")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(history["train_loss"], label="Train Loss")
                axes[0].plot(history["val_loss"], label="Val Loss")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(history["train_acc"], label="Train Acc")
                axes[1].plot(history["val_acc"], label="Val Acc")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                fig.suptitle(f"{target_label(target)} – MRMR BiLSTM")
                st.pyplot(fig)
                plt.close(fig)
                
            st.button("Tiếp tục → Predict", on_click=goto, args=("Predict",))

    layout_with_manager(render_main)


def page_predict() -> None:
    """Predict page."""
    def render_main() -> None:
        st.title("🔮 Predict")
        processed_records = get_processed_records()
        if not processed_records:
            st.warning("Chưa có dữ liệu FFT để dự đoán.")
            return

        target_choices = {target: st.checkbox(target_label(target), value=True, key=f"predict_choice_{target}") for target in TARGETS}
        selected_targets = [target for target, enabled in target_choices.items() if enabled]
        if not selected_targets:
            st.warning("Hãy chọn ít nhất một nhãn để dự đoán.")
            return

        record_names = [record["name"] for record in processed_records]
        selected_name = st.selectbox("Chọn dữ liệu từ processed_data", record_names, key="predict_processed_select")
        selected_record = next(record for record in processed_records if record["name"] == selected_name)

        from app.data_normalization import resolve_model_entry
        missing = []
        for target in selected_targets:
            if resolve_model_entry(target) is None:
                missing.append(f"thiếu model cho {target_label(target)}")
        if missing:
            st.error("; ".join(sorted(set(missing))))
            return

        if st.button("🚀 Predict", type="primary"):
            file_manager_snapshot = snapshot_file_manager()
            with st.status("Đang chạy inference song parallel...", expanded=True) as status:
                results: dict[str, dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                    future_map = {executor.submit(predict_target, target, selected_record, file_manager_snapshot): target for target in selected_targets}
                    for future in as_completed(future_map):
                        target = future_map[future]
                        result = future.result()
                        results[target] = result
                        status.write(f"Hoàn thành {target_label(target)}")
                status.update(label="Predict hoàn tát", state="complete")
                st.session_state.runtime["prediction_results"] = results
        
        prediction_results = st.session_state.runtime.get("prediction_results", {})
        
        if prediction_results and "arousal" in prediction_results and "valence" in prediction_results:
            df_a = prediction_results["arousal"]["results"]
            df_v = prediction_results["valence"]["results"]
            
            if len(df_a) != len(df_v):
                st.error("Lỗi: Số lượng cửa sổ dự đoán giữa Arousal và Valence không khớp!")
                return
            
            total_windows = len(df_a)
            num_trials = 40
            wpt = total_windows // num_trials 
            
            df_a = df_a.copy()
            df_v = df_v.copy()
            df_a["trial_id"] = np.clip(np.arange(len(df_a)) // wpt, 0, num_trials - 1)
            df_v["trial_id"] = np.clip(np.arange(len(df_v)) // wpt, 0, num_trials - 1)
            
            trial_results = []
            
            for tid in range(num_trials):
                sub_a = df_a[df_a["trial_id"] == tid]
                sub_v = df_v[df_v["trial_id"] == tid]
                
                if len(sub_a) == 0 or len(sub_v) == 0:
                    continue
                
                a_true = sub_a["true_label"].values[0]
                v_true = sub_v["true_label"].values[0]
                
                a_pred = sub_a["prediction"].mode()[0]
                v_pred = sub_v["prediction"].mode()[0]
                
                trial_results.append({
                    "trial_id": tid,
                    "a_true": a_true, "a_pred": a_pred, "a_correct": a_true == a_pred,
                    "v_true": v_true, "v_pred": v_pred, "v_correct": v_true == v_pred,
                    "both_correct": (a_true == a_pred) and (v_true == v_pred),
                    "sub_a": sub_a,
                    "sub_v": sub_v
                })
            
            actual_num_trials = len(trial_results)
            if actual_num_trials == 0:
                st.error("Không tìm thấy dữ liệu Trial hợp lệ.")
                return
                
            acc_both = sum(r["both_correct"] for r in trial_results) / actual_num_trials
            acc_a = sum(r["a_correct"] for r in trial_results) / actual_num_trials
            acc_v = sum(r["v_correct"] for r in trial_results) / actual_num_trials
            
            st.markdown("---")
            st.subheader("📊 Tổng quan kết quả (Theo Trial)")
            st.caption(f"Đã phân tích {actual_num_trials} Trials. Một Trial được tính là đúng nếu nhãn chiếm đa số của các sample khớp với nhãn thực tế.")
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            col_metric1.metric("Trial Correct (Cả 2 cùng đúng)", f"{acc_both*100:.1f}%")
            col_metric2.metric("Trial Correct (Chỉ tính Arousal)", f"{acc_a*100:.1f}%")
            col_metric3.metric("Trial Correct (Chỉ tính Valence)", f"{acc_v*100:.1f}%")
            
            st.markdown("---")
            selected_trial_idx = st.slider("🎯 Kéo để chọn Trial cần kiểm tra chi tiết", 1, actual_num_trials, 1) - 1
            curr_res = trial_results[selected_trial_idx]
            
            chung_status = "✅ ĐÚNG CẢ 2" if curr_res["both_correct"] else "❌ SAI (Ít nhất 1 nhãn không khớp)"
            st.markdown(f"""
            <div style="background-color: #eaf5ff; border: 1px solid #cce0ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #1e293b;">
    <h4 style="margin-top: 0; color: #004d99;">Thông số Trial {curr_res['trial_id'] + 1}</h4>
    <b>• Nhãn thực tế:</b> Arousal = <b style='color: #e07b39;'>{curr_res['a_true']}</b> | Valence = <b style='color: #5b8dd9;'>{curr_res['v_true']}</b><br>
    <b>• Kết quả dự đoán (Đa số):</b> Arousal = <b>{curr_res['a_pred']}</b> | Valence = <b>{curr_res['v_pred']}</b> ➡️ <b>{chung_status}</b>
</div>
""", unsafe_allow_html=True)
            
            col_a_ui, col_v_ui = st.columns(2, gap="medium")
            
            with col_a_ui:
                st.markdown("<h4 style='text-align: center; color: #e07b39;'>AROUSAL</h4>", unsafe_allow_html=True)
                df_trial_a = curr_res["sub_a"].reset_index(drop=True)
                
                t_acc = (df_trial_a["prediction"] == df_trial_a["true_label"]).mean()
                st.info(f"**Tỉ lệ Sample đúng trong Trial này:** {t_acc*100:.1f}%\n\n**Dự đoán Trial:** {'✅ Đúng' if curr_res['a_correct'] else '❌ Sai'}")
                
                max_p_a = max(1, (len(df_trial_a) - 1) // 10 + 1)
                page_a = st.number_input("Khoảng Sample (10 samples/trang) - Arousal", min_value=1, max_value=max_p_a, step=1, key="page_a")
                
                s_idx = (page_a - 1) * 10
                e_idx = s_idx + 10
                st.dataframe(
                    df_trial_a.iloc[s_idx:e_idx][["window", "true_label", "prediction", "p_high", "p_low"]]
                    .rename(columns={"window": "Sample ID", "true_label": "Nhãn thực", "prediction": "Dự đoán", "p_high": "Tỉ lệ (High)", "p_low": "Tỉ lệ (Low)"}), 
                    use_container_width=True, hide_index=True
                )
            
            with col_v_ui:
                st.markdown("<h4 style='text-align: center; color: #5b8dd9;'>VALENCE</h4>", unsafe_allow_html=True)
                df_trial_v = curr_res["sub_v"].reset_index(drop=True)
                
                t_acc_v = (df_trial_v["prediction"] == df_trial_v["true_label"]).mean()
                st.info(f"**Tỉ lệ Sample đúng trong Trial này:** {t_acc_v*100:.1f}%\n\n**Dự đoán Trial:** {'✅ Đúng' if curr_res['v_correct'] else '❌ Sai'}")
                
                max_p_v = max(1, (len(df_trial_v) - 1) // 10 + 1)
                page_v = st.number_input("Khoảng Sample (10 samples/trang) - Valence", min_value=1, max_value=max_p_v, step=1, key="page_v")
                
                s_idx_v = (page_v - 1) * 10
                e_idx_v = s_idx_v + 10
                st.dataframe(
                    df_trial_v.iloc[s_idx_v:e_idx_v][["window", "true_label", "prediction", "p_high", "p_low"]]
                    .rename(columns={"window": "Sample ID", "true_label": "Nhãn thực", "prediction": "Dự đoán", "p_high": "Tỉ lệ (High)", "p_low": "Tỉ lệ (Low)"}), 
                    use_container_width=True, hide_index=True
                )

        elif prediction_results:
            st.info("💡 Bạn cần tick cả Arousal và Valence để bật chế độ xem so sánh 2 bên (Split-view). Dưới đây là kết quả đơn:")
            for tgt, res in prediction_results.items():
                st.write(f"**{target_label(tgt)}**")
                st.dataframe(res["results"].head(20), use_container_width=True, hide_index=True)

    layout_with_manager(render_main)


PAGE_FUNCS = {
    "Home": page_home,
    "Load Data": page_load_data,
    "Preprocess": page_preprocess,
    "Train Model": page_train,
    "Predict": page_predict,
}


def render_app() -> None:
    """Render main application."""
    from app.state_management import init_state, ensure_state
    
    ensure_state()
    current_page = st.session_state.get("page", "Home")
    if current_page not in PAGE_FUNCS:
        current_page = "Home"
        st.session_state.page = current_page

    with st.sidebar:
        st.title("🧠 EEG Emotion")
        st.caption("Recognition Dashboard")
        st.markdown("---")
        pages = {
            "Home": "🏠 Home",
            "Load Data": "📤 Load DEAP Data",
            "Preprocess": "⚡ Preprocess (FFT)",
            "Train Model": "🎓 Train Model",
            "Predict": "🔮 Predict",
        }
        for key, label in pages.items():
            is_active = current_page == key
            st.button(label, on_click=goto, args=(key,), use_container_width=True, type="primary" if is_active else "secondary")
        st.markdown("---")
        st.caption("DEAP Dataset · MRMR · BiLSTM")
        st.caption("Mainstream – Emotion Recognition")

    PAGE_FUNCS[current_page]()
