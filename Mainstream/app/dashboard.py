from __future__ import annotations

import io
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_model
from src.mrmr_selection import (
    MRMR_COMPONENTS,
    N_FREQUENCIES,
    N_CHANNELS,
    TEST_SPLIT_MODULO,
    build_mrmr_dataset,
    prepare_for_lstm,
    preprocess_subject_fft,
    run_mrmr_global_selection,
)
from src.preprocess import MRMR_BAND_NAMES

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
TARGETS = ("arousal", "valence")


def _init_state() -> None:
    defaults = {
        "page": "Home",
        "file_manager": {
            "raw_data": [],
            "processed_data": [],
            "mrmr_selection": {"arousal": None, "valence": None},
            "models": {"arousal": None, "valence": None},
        },
        "runtime": {
            "mrmr_results": {},
            "training_results": {},
            "prediction_results": {},
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def goto(page: str) -> None:
    st.session_state.page = page


# ------------------------------------------------------------------
# State helpers
# ------------------------------------------------------------------
def _ensure_state() -> None:
    if "file_manager" not in st.session_state or "runtime" not in st.session_state:
        _init_state()


def _file_manager() -> dict[str, Any]:
    _ensure_state()
    return st.session_state["file_manager"]


def _snapshot_file_manager() -> dict[str, Any]:
    fm = _file_manager()
    return {
        "raw_data": list(fm["raw_data"]),
        "processed_data": list(fm["processed_data"]),
        "mrmr_selection": dict(fm["mrmr_selection"]),
        "models": dict(fm["models"]),
    }


def _replace_by_name(items: list[dict[str, Any]], name: str, new_item: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = [item for item in items if item.get("name") != name]
    filtered.append(new_item)
    return filtered


def _target_label(target: str) -> str:
    return "Arousal" if target == "arousal" else "Valence"


def _channel_name(index: int) -> str:
    if 0 <= index < len(DEAP_ELECTRODES):
        return DEAP_ELECTRODES[index]
    return f"Ch{index}"


def _extract_channels(selection: Any) -> list[int]:
    if selection is None:
        return []
    if isinstance(selection, dict):
        if "channels" in selection and selection["channels"] is not None:
            return [int(value) for value in selection["channels"]]
        selection = selection.get("data", selection)
    if isinstance(selection, pd.DataFrame):
        if "channels" in selection.columns:
            series = selection["channels"]
        else:
            series = selection.iloc[:, 0]
        return [int(value) for value in series.tolist()]
    if isinstance(selection, (list, tuple, np.ndarray, pd.Series)):
        return [int(value) for value in selection]
    raise TypeError(f"Unsupported MRMR selection type: {type(selection)!r}")


def _resolve_mrmr_entry(target: str, file_manager: dict[str, Any] | None = None) -> dict[str, Any] | None:
    fm = file_manager or _file_manager()
    entry = fm["mrmr_selection"].get(target)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"name": None, "data": entry, "channels": _extract_channels(entry)}


def _resolve_model_entry(target: str, file_manager: dict[str, Any] | None = None) -> dict[str, Any] | None:
    fm = file_manager or _file_manager()
    entry = fm["models"].get(target)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"name": None, "model": entry}


def _get_processed_records() -> list[dict[str, Any]]:
    return _file_manager()["processed_data"]


def _get_raw_records() -> list[dict[str, Any]]:
    return _file_manager()["raw_data"]


def _normalize_raw_record(name: str, subject: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "subject": subject}


def _normalize_processed_record(name: str, data: np.ndarray, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {"name": name, "data": data}
    if meta:
        record.update(meta)
    return record


def _store_raw_data(uploaded_files: list[Any]) -> None:
    raw_records = _get_raw_records()
    for file_obj in uploaded_files:
        subject = pickle.load(io.BytesIO(file_obj.read()), encoding="latin1")
        raw_records = _replace_by_name(raw_records, file_obj.name, _normalize_raw_record(file_obj.name, subject))
    _file_manager()["raw_data"] = raw_records


def _store_processed_data(name: str, data: np.ndarray, meta: dict[str, Any] | None = None) -> None:
    processed_records = _get_processed_records()
    processed_records = _replace_by_name(processed_records, name, _normalize_processed_record(name, data, meta))
    _file_manager()["processed_data"] = processed_records


def _selection_dataframe(channels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "channels": [int(value) for value in channels],
        "channel_names": [_channel_name(int(value)) for value in channels],
    })


def _store_mrmr_result(target: str, channels: list[int], source: str, name: str | None = None) -> None:
    _file_manager()["mrmr_selection"][target] = {
        "name": name or f"mrmr_{target}.xlsx",
        "data": _selection_dataframe(channels),
        "channels": [int(value) for value in channels],
        "source": source,
    }


def _store_model_result(target: str, model: nn.Module, checkpoint: dict[str, Any], source: str, name: str | None = None) -> None:
    _file_manager()["models"][target] = {
        "name": name or f"{target}_mrmr_lstm.pth",
        "model": model,
        "checkpoint": checkpoint,
        "source": source,
    }


def _selection_to_download_bytes(target: str) -> bytes:
    entry = _resolve_mrmr_entry(target)
    if entry is None:
        return b""
    df = entry["data"]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=target.capitalize())
    return output.getvalue()


def _checkpoint_to_bytes(checkpoint: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _load_model_from_checkpoint(checkpoint: Any) -> nn.Module:
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    model = build_model("mrmr_lstm", input_size=1)
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    if state_dict is None:
        raise ValueError("Checkpoint does not contain a valid model state dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _upload_mrmr_file(target: str, uploaded_file: Any) -> None:
    if uploaded_file is None:
        raise ValueError("Chưa chọn file MRMR để upload")
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    else:
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
    channels = _extract_channels(df)
    _file_manager()["mrmr_selection"][target] = {
        "name": uploaded_file.name,
        "data": df,
        "channels": channels,
        "source": "upload",
    }


def _upload_model_file(target: str, uploaded_file: Any) -> None:
    if uploaded_file is None:
        raise ValueError("Chưa chọn file model để upload")
    checkpoint = torch.load(io.BytesIO(uploaded_file.getvalue()), map_location="cpu")
    model = _load_model_from_checkpoint(checkpoint)
    _store_model_result(target, model, checkpoint if isinstance(checkpoint, dict) else {"model": model.state_dict()}, "upload", uploaded_file.name)


def _load_selected_channels(target: str) -> list[int]:
    entry = _resolve_mrmr_entry(target)
    if entry is None:
        return []
    return _extract_channels(entry)


def _build_processed_arrays(processed_records: list[dict[str, Any]]) -> list[np.ndarray]:
    return [record["data"] for record in processed_records]


def _prepare_training_arrays(processed_records: list[dict[str, Any]], channels: list[int], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_arrays = _build_processed_arrays(processed_records)
    selected_per_subject = [channels] * len(processed_arrays)
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = build_mrmr_dataset(processed_arrays, selected_per_subject)
    return prepare_for_lstm(x_train_raw, x_test_raw, y_train_raw, y_test_raw, classify_type=target)


def _predict_windows(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim == 2:
        x = x[:, :, np.newaxis]

    if hasattr(model, "eval"):
        model.eval()
    x_tensor = torch.from_numpy(x.astype(np.float32))
    device = next(model.parameters()).device if isinstance(model, nn.Module) else torch.device("cpu")
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return probs, preds


def _prepare_prediction_inputs(processed_record: dict[str, Any], channels: list[int], target: str) -> tuple[np.ndarray, np.ndarray]:
    data = processed_record["data"]
    n_windows = data.shape[0]
    data_list = [data[i][0] for i in range(n_windows)]
    label_list = [data[i][1] for i in range(n_windows)]

    window_data = np.array(data_list)
    window_labels = np.array(label_list)
    n_ch = window_data.shape[1]
    x_all = window_data.transpose((1, 0, 2)).reshape(n_ch, -1).transpose((1, 0))
    x_df = pd.DataFrame(x_all)
    filtered = x_df[channels].to_numpy()

    reshaped = []
    for ch_data in filtered.T:
        reshaped.append(ch_data.reshape(-1, N_FREQUENCIES))
    x_flat = np.array(reshaped).transpose((1, 0, 2)).reshape(-1, len(channels) * N_FREQUENCIES)

    y_bin = window_labels[:, 0] if target == "arousal" else window_labels[:, 1]
    x_train, y_train, _, _ = prepare_for_lstm(x_flat, x_flat, np.column_stack([y_bin, y_bin]), np.column_stack([y_bin, y_bin]), classify_type=target)
    return x_train, y_train


def _channel_dataframe(channels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "channels": [int(channel) for channel in channels],
        "channel_names": [_channel_name(int(channel)) for channel in channels],
    })


def _run_mrmr_task(target: str, processed_records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    channels = run_mrmr_global_selection(_build_processed_arrays(processed_records), classify_type=target, K=k)
    return {
        "target": target,
        "channels": channels,
        "dataframe": _channel_dataframe(channels),
        "file_name": f"mrmr_{target}.xlsx",
    }


def _fit_model_for_target(target: str, processed_records: list[dict[str, Any]], channels: list[int], epochs: int, lr: float, batch_size: int, dropout: float) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = _prepare_training_arrays(processed_records, channels, target)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("mrmr_lstm", input_size=1, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(int(epochs)):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += len(yb)

        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * len(yb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total += len(yb)

        scheduler.step()

        tr_loss_ep = tr_loss / max(tr_total, 1)
        va_loss_ep = va_loss / max(va_total, 1)
        tr_acc_ep = tr_correct / max(tr_total, 1)
        va_acc_ep = va_correct / max(va_total, 1)

        history["train_loss"].append(tr_loss_ep)
        history["val_loss"].append(va_loss_ep)
        history["train_acc"].append(tr_acc_ep)
        history["val_acc"].append(va_acc_ep)

        if va_acc_ep >= best_val_acc:
            best_val_acc = va_acc_ep
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "target": target,
        "history": history,
        "selected_channels": channels,
        "input_size": 1,
    }
    return {
        "target": target,
        "model": model,
        "history": history,
        "checkpoint": checkpoint,
        "seq_len": x_train.shape[1],
        "best_val_acc": best_val_acc,
    }


def _predict_target(target: str, processed_record: dict[str, Any], file_manager: dict[str, Any]) -> dict[str, Any]:
    mrmr_entry = _resolve_mrmr_entry(target, file_manager)
    model_entry = _resolve_model_entry(target, file_manager)
    if mrmr_entry is None:
        raise ValueError(f"Thiếu file MRMR cho {_target_label(target)}")
    if model_entry is None or model_entry.get("model") is None:
        raise ValueError(f"Thiếu model cho {_target_label(target)}")

    channels = _extract_channels(mrmr_entry)
    model = model_entry["model"]
    x_input, y_input = _prepare_prediction_inputs(processed_record, channels, target)
    probs, preds = _predict_windows(model, x_input)
    accuracy = float((preds == y_input).mean()) if len(y_input) else 0.0

    results = pd.DataFrame({
        "window": np.arange(len(preds)),
        "prediction": ["High" if int(pred) == 1 else "Low" for pred in preds],
        "true_label": ["High" if int(label) == 1 else "Low" for label in y_input],
        "p_low": probs[:, 0],
        "p_high": probs[:, 1],
    })
    return {
        "target": target,
        "accuracy": accuracy,
        "results": results,
        "preds": preds,
        "probs": probs,
        "channels": channels,
    }


def _render_manager_panel() -> None:
    fm = _file_manager()
    st.subheader("🗂 Virtual File Manager")

    def _entry_status_line(target: str, entry: dict[str, Any] | None) -> str:
        file_name = (entry or {}).get("name")
        return f"{_target_label(target)}: {file_name if file_name else 'Trống'}"

    def _auto_upload_row(
        target: str,
        entry: dict[str, Any] | None,
        upload_type: str,
        file_types: list[str],
    ) -> None:
        left_col, right_col = st.columns([2.6, 1.2], gap="small")
        with left_col:
            st.markdown(f"**{_entry_status_line(target, entry)}**")
        with right_col:
            upload = st.file_uploader(
                f"Upload {_target_label(target)}",
                type=file_types,
                key=f"manager_{upload_type}_upload_{target}",
                label_visibility="collapsed",
            )

        # Auto-upload ngay khi người dùng chọn file, không cần nút Upload riêng.
        if upload is not None:
            content = upload.getvalue()
            marker_key = f"manager_{upload_type}_uploaded_marker_{target}"
            file_signature = (upload.name, len(content), hash(content))
            if st.session_state.get(marker_key) != file_signature:
                try:
                    if upload_type == "mrmr":
                        _upload_mrmr_file(target, upload)
                        st.success(f"Đã cập nhật MRMR cho {_target_label(target)}.")
                    else:
                        _upload_model_file(target, upload)
                        st.success(f"Đã cập nhật model cho {_target_label(target)}.")
                    st.session_state[marker_key] = file_signature
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể upload file cho {_target_label(target)}: {exc}")

    with st.expander("📂 Data", expanded=True):
        if fm["raw_data"]:
            for item in fm["raw_data"]:
                st.write(f"• {item['name']}")
        else:
            st.caption("Chưa có file .dat nào.")
        st.caption("Upload .dat chỉ thực hiện tại trang Load DEAP Data.")

    with st.expander("⚡ Processed Data", expanded=True):
        if fm["processed_data"]:
            for item in fm["processed_data"]:
                st.write(f"• {item['name']}")
        else:
            st.caption("Chưa có dữ liệu FFT.")

    with st.expander("🔬 MRMR Selection", expanded=True):
        for target in TARGETS:
            _auto_upload_row(
                target=target,
                entry=fm["mrmr_selection"].get(target),
                upload_type="mrmr",
                file_types=["xlsx", "xls", "csv"],
            )

    with st.expander("🎓 Model", expanded=True):
        for target in TARGETS:
            _auto_upload_row(
                target=target,
                entry=fm["models"].get(target),
                upload_type="model",
                file_types=["pth"],
            )


def _layout_with_manager(main_render_fn) -> None:
    main_col, manager_col = st.columns([3.5, 1.2], gap="medium")
    with main_col:
        main_render_fn()
    with manager_col:
        _render_manager_panel()


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------
def page_home() -> None:
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
        st.info("4. MRMR Selection\nChọn kênh cho Arousal và Valence")
    with col2:
        st.success("2. Preprocess\nTrích xuất FFT 5 dải tần")
        st.success("5. Train Model\nHuấn luyện BiLSTM theo từng nhãn")
    with col3:
        st.warning("3. Explore\nXem trước tín hiệu EEG")
        st.warning("6. Predict\nChạy suy luận song song")


def page_load_data() -> None:
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
                    _store_raw_data(uploaded)
                    status.update(label=f"Đã load {len(uploaded)} file .dat", state="complete")
                st.success(f"Đã load {len(uploaded)} file(s).")

        raw_records = _get_raw_records()
        if raw_records:
            st.markdown("---")
            st.subheader("Preview dữ liệu")
            current = raw_records[0]
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
                ax.set_title(f"EEG – {_channel_name(ch_idx)} | Trial {trial_idx}")
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

        st.button("Tiếp tục → Preprocess", on_click=goto, args=("Preprocess",))

    _layout_with_manager(render_main)


def page_preprocess() -> None:
    def render_main() -> None:
        st.title("⚡ Preprocess – FFT Feature Extraction")
        raw_records = _get_raw_records()
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
            processed = []
            progress = st.progress(0)
            for index, record in enumerate(raw_records):
                preprocessed = preprocess_subject_fft(record["subject"], window_size=int(window_size), step_size=int(step_size))
                processed.append(_normalize_processed_record(record["name"], preprocessed, {"window_size": int(window_size), "step_size": int(step_size)}))
                progress.progress((index + 1) / len(raw_records))
            _file_manager()["processed_data"] = processed
            st.success(f"Đã xử lý {len(processed)} subject(s).")

        processed_records = _get_processed_records()
        if processed_records:
            st.markdown("---")
            st.subheader("Kết quả Preprocess")
            current = processed_records[0]["data"]
            st.write(f"Subject: `{processed_records[0]['name']}` | Windows: `{current.shape[0]}` | Feature shape: `{current[0][0].shape}`")
            sample_features = current[0][0]
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(sample_features, aspect="auto", cmap="viridis")
            ax.set_yticks(range(sample_features.shape[0]))
            ax.set_yticklabels([_channel_name(i) for i in range(sample_features.shape[0])], fontsize=6)
            ax.set_xticks(range(sample_features.shape[1]))
            ax.set_xticklabels(MRMR_BAND_NAMES, fontsize=9)
            ax.set_title("FFT Feature Map – Window 0")
            plt.colorbar(im, ax=ax, label="Power")
            st.pyplot(fig)
            plt.close(fig)

        st.button("Tiếp tục → MRMR", on_click=goto, args=("MRMR Selection",))

    _layout_with_manager(render_main)


def page_mrmr() -> None:
    def render_main() -> None:
        st.title("🔬 MRMR Channel Selection")
        processed_records = _get_processed_records()
        if not processed_records:
            st.warning("Chưa có dữ liệu FFT. Hãy preprocess trước.")
            st.button("← Preprocess", on_click=goto, args=("Preprocess",))
            return

        target_choices = {target: st.checkbox(_target_label(target), value=True, key=f"mrmr_choice_{target}") for target in TARGETS}
        selected_targets = [target for target, enabled in target_choices.items() if enabled]
        if not selected_targets:
            st.warning("Hãy chọn ít nhất một nhãn để chạy MRMR.")
            return

        k_value = st.slider("Số kênh MRMR (K)", min_value=5, max_value=32, value=MRMR_COMPONENTS, step=1)

        if st.button("🔬 Chạy MRMR", type="primary"):
            with st.status("Đang chạy MRMR...", expanded=True) as status:
                tasks = {}
                results: dict[str, dict[str, Any]] = {}
                max_workers = len(selected_targets)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for target in selected_targets:
                        tasks[executor.submit(_run_mrmr_task, target, processed_records, int(k_value))] = target
                    for future in as_completed(tasks):
                        target = tasks[future]
                        result = future.result()
                        results[target] = result
                        _store_mrmr_result(target, result["channels"], source="computed", name=result["file_name"])
                        status.write(f"Hoàn thành {_target_label(target)}")
                status.update(label="MRMR hoàn tất", state="complete")
                st.session_state.runtime["mrmr_results"] = results

        runtime_results = st.session_state.runtime.get("mrmr_results", {})
        if runtime_results:
            st.markdown("---")
            for target, result in runtime_results.items():
                st.subheader(f"{_target_label(target)}")
                channels = result["channels"]
                channel_names = ", ".join(_channel_name(channel) for channel in channels)
                st.write(f"Kênh chọn: {channel_names}")
                st.dataframe(result["dataframe"], use_container_width=True, hide_index=True)
                st.download_button(
                    f"Tải {_target_label(target)} Excel",
                    data=_selection_to_download_bytes(target),
                    file_name=result["file_name"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_mrmr_page_{target}",
                )

        st.button("Tiếp tục → Train Model", on_click=goto, args=("Train Model",))

    _layout_with_manager(render_main)


def page_train() -> None:
    def render_main() -> None:
        st.title("🎓 Train Model")
        processed_records = _get_processed_records()
        if not processed_records:
            st.warning("Chưa có dữ liệu FFT. Hãy preprocess trước.")
            st.button("← Preprocess", on_click=goto, args=("Preprocess",))
            return

        target_choices = {target: st.checkbox(_target_label(target), value=True, key=f"train_choice_{target}") for target in TARGETS}
        selected_targets = [target for target, enabled in target_choices.items() if enabled]
        if not selected_targets:
            st.warning("Hãy chọn ít nhất một nhãn để train.")
            return

        col_a, col_b, col_c = st.columns(3)
        epochs = int(col_a.number_input("Epochs", value=50, min_value=1, max_value=500, step=5))
        lr = float(col_b.number_input("Learning rate", value=LR, min_value=1e-5, max_value=0.1, format="%.5f"))
        batch_size = int(col_c.number_input("Batch size", value=BATCH_SIZE, min_value=32, max_value=1024, step=32))
        dropout = float(st.slider("Dropout", 0.0, 0.8, 0.5, 0.05))

        if st.button("🚀 Bắt đầu Training", type="primary"):
            missing = []
            for target in selected_targets:
                if _resolve_mrmr_entry(target) is None:
                    missing.append(f"thiếu file MRMR cho {_target_label(target)}")
            if missing:
                st.error("; ".join(missing))
                return

            with st.status("Đang train song song...", expanded=True) as status:
                results: dict[str, dict[str, Any]] = {}
                with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                    future_map = {
                        executor.submit(
                            _fit_model_for_target,
                            target,
                            processed_records,
                            _load_selected_channels(target),
                            epochs,
                            lr,
                            batch_size,
                            dropout,
                        ): target
                        for target in selected_targets
                    }
                    for future in as_completed(future_map):
                        target = future_map[future]
                        result = future.result()
                        results[target] = result
                        _store_model_result(target, result["model"], result["checkpoint"], source="trained")
                        status.write(f"Hoàn thành {_target_label(target)}")
                status.update(label="Training hoàn tất", state="complete")
                st.session_state.runtime["training_results"] = results

        training_results = st.session_state.runtime.get("training_results", {})
        if training_results:
            st.markdown("---")
            for target, result in training_results.items():
                st.subheader(f"{_target_label(target)}")
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
                fig.suptitle(f"{_target_label(target)} – MRMR BiLSTM")
                st.pyplot(fig)
                plt.close(fig)
                st.download_button(
                    f"Download {_target_label(target)} checkpoint (.pth)",
                    data=_checkpoint_to_bytes(result["checkpoint"]),
                    file_name=f"{target}_mrmr_lstm.pth",
                    mime="application/octet-stream",
                    key=f"download_train_model_{target}",
                )

        st.button("Tiếp tục → Predict", on_click=goto, args=("Predict",))

    _layout_with_manager(render_main)


def page_predict() -> None:
    def render_main() -> None:
        st.title("🔮 Predict")
        processed_records = _get_processed_records()
        if not processed_records:
            st.warning("Chưa có dữ liệu FFT để dự đoán.")
            return

        target_choices = {target: st.checkbox(_target_label(target), value=True, key=f"predict_choice_{target}") for target in TARGETS}
        selected_targets = [target for target, enabled in target_choices.items() if enabled]
        if not selected_targets:
            st.warning("Hãy chọn ít nhất một nhãn để dự đoán.")
            return

        record_names = [record["name"] for record in processed_records]
        selected_name = st.selectbox("Chọn dữ liệu từ processed_data", record_names, key="predict_processed_select")
        selected_record = next(record for record in processed_records if record["name"] == selected_name)

        missing = []
        for target in selected_targets:
            if _resolve_mrmr_entry(target) is None:
                missing.append(f"thiếu file MRMR cho {_target_label(target)}")
            if _resolve_model_entry(target) is None:
                missing.append(f"thiếu model cho {_target_label(target)}")
        if missing:
            st.error("; ".join(sorted(set(missing))))
            return

        if st.button("🚀 Predict", type="primary"):
                file_manager_snapshot = _snapshot_file_manager()
                with st.status("Đang chạy inference song song...", expanded=True) as status:
                    results: dict[str, dict[str, Any]] = {}
                    with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                        future_map = {executor.submit(_predict_target, target, selected_record, file_manager_snapshot): target for target in selected_targets}
                        for future in as_completed(future_map):
                            target = future_map[future]
                            result = future.result()
                            results[target] = result
                            status.write(f"Hoàn thành {_target_label(target)}")
                    status.update(label="Predict hoàn tất", state="complete")
                    st.session_state.runtime["prediction_results"] = results
                    st.dataframe(result["results"].head(20), use_container_width=True, hide_index=True)

    _layout_with_manager(render_main)


PAGE_FUNCS = {
    "Home": page_home,
    "Load Data": page_load_data,
    "Preprocess": page_preprocess,
    "MRMR Selection": page_mrmr,
    "Train Model": page_train,
    "Predict": page_predict,
}


def render_app() -> None:
    _ensure_state()
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
            "MRMR Selection": "🔬 MRMR Channel Selection",
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


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
