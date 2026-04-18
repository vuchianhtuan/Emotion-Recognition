from __future__ import annotations
import io
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components # Thêm module này để chạy JS cuộn trang
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize as _normalize, StandardScaler

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
DEBUG_MODE = True  # Đổi thành False để ẩn toàn bộ tính năng kiểm thử
TEST_DATA_DIR = "test_data" # Tên thư mục bạn sẽ tạo trên server để chứa file test

def _init_state() -> None:
    defaults = {
        "page": "Home",
        "scroll_to_preview": False, # Biến cờ hiệu để kích hoạt auto-scroll
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

# Hàm callback khi click vào tên file trong File Manager
def _on_file_link_click(target_page: str, state_key: str, filename: str) -> None:
    st.session_state.page = target_page
    st.session_state[state_key] = filename
    st.session_state.scroll_to_preview = True # Bật cờ hiệu cuộn trang

# Hàm tiện ích chạy Javascript cuộn mượt xuống khu vực Preview
def _trigger_scroll_if_needed():
    if st.session_state.get("scroll_to_preview"):
        components.html(
            """
            <script>
                var parent = window.parent.document;
                var target = parent.getElementById('preview-section');
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            </script>
            """,
            height=0
        )
        st.session_state.scroll_to_preview = False

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

    input_size = 1
    seq_len = None
    if isinstance(checkpoint, dict):
        input_size = int(checkpoint.get("input_size", 1))
        seq_len = checkpoint.get("seq_len")

    model = build_model("mrmr_lstm", seq_len=seq_len, input_size=input_size)
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
    resolved_target = target
    if isinstance(checkpoint, dict) and checkpoint.get("target") in TARGETS:
        resolved_target = str(checkpoint.get("target"))
    _store_model_result(
        resolved_target,
        model,
        checkpoint if isinstance(checkpoint, dict) else {"model": model.state_dict()},
        "upload",
        uploaded_file.name,
    )

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
    x_train_norm = _normalize(x_train_raw).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x_train_norm)
    scaler_state = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
    }

    x_train, y_train, x_test, y_test = prepare_for_lstm(
        x_train_raw,
        x_test_raw,
        y_train_raw,
        y_test_raw,
        classify_type=target,
    )

    x_train = _reshape_flat_features_for_model(x_train.reshape(x_train.shape[0], -1), n_channels=len(channels))
    x_test = _reshape_flat_features_for_model(x_test.reshape(x_test.shape[0], -1), n_channels=len(channels))
    return x_train, y_train, x_test, y_test, scaler_state

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
    
    # Bóc tách trực tiếp features và labels để giữ nguyên 100% thứ tự thời gian
    x_list = []
    y_list = []
    for feat, lbl in data:
        # Chỉ lấy các kênh MRMR đã được chọn
        x_list.append(feat[channels, :])
        y_list.append(lbl)

    x_all = np.array(x_list, dtype=np.float32)
    # Flatten features từ (samples, channels, freqs) -> (samples, channels * freqs)
    x_flat = x_all.reshape(x_all.shape[0], -1)

    y_all = np.array(y_list, dtype=np.float32)
    
    # Đưa nhãn về dạng nhị phân (0/1) nếu đang ở scale 1-9 của DEAP (>= 5 là High)
    if y_all.max() > 1.0:
        y_bin = (y_all >= 5.0).astype(np.int64)
    else:
        y_bin = y_all.astype(np.int64)

    # ĐÃ SỬA LỖI Ở ĐÂY:
    # Đồng bộ index với hàm prepare_for_lstm trong quá trình training
    # Arousal sẽ được đánh giá ở cột 0, Valence ở cột 1
    target_idx = 0 if target == "arousal" else 1
    y_target = y_bin[:, target_idx]

    return x_flat, y_target

def _reshape_flat_features_for_model(x_2d: np.ndarray, n_channels: int | None = None) -> np.ndarray:
    if x_2d.ndim != 2:
        raise ValueError("Input phải là mảng 2D có shape (n_samples, n_features).")

    n_samples, n_features = x_2d.shape
    if n_features % N_FREQUENCIES != 0:
        raise ValueError(f"Feature dim={n_features} không chia hết cho số dải tần {N_FREQUENCIES}.")

    inferred_channels = n_features // N_FREQUENCIES
    channels = inferred_channels if n_channels is None else int(n_channels)
    if channels != inferred_channels:
        raise ValueError(
            f"Mismatch số kênh: dữ liệu có {inferred_channels} kênh, nhưng kỳ vọng {channels}."
        )

    x_cf = x_2d.reshape(n_samples, channels, N_FREQUENCIES)
    return x_cf.transpose(0, 2, 1).astype(np.float32)

def _flatten_model_features(x_3d: np.ndarray) -> np.ndarray:
    if x_3d.ndim != 3:
        raise ValueError("Input phải là mảng 3D.")

    if x_3d.shape[1] == N_FREQUENCIES:
        x_cf = x_3d.transpose(0, 2, 1)
    elif x_3d.shape[2] == N_FREQUENCIES:
        x_cf = x_3d
    else:
        raise ValueError(
            f"Không suy ra được layout từ shape={x_3d.shape}. Cần có một trục bằng {N_FREQUENCIES}."
        )
    return x_cf.reshape(x_cf.shape[0], -1).astype(np.float32)

def _apply_saved_scaler(x: np.ndarray, scaler_state: dict, already_l2_normalized: bool = False) -> np.ndarray:
    if x.ndim == 3:
        if x.shape[-1] == 1:
            x_2d = x.reshape(x.shape[0], x.shape[1]).astype(np.float32)
            original_layout = "legacy"
            legacy_seq_len = x.shape[1]
        else:
            x_2d = _flatten_model_features(x)
            original_layout = "channel_frequency"
            original_channels = x.shape[2] if x.shape[1] == N_FREQUENCIES else x.shape[1]
    elif x.ndim == 2:
        x_2d = x.astype(np.float32)
        original_layout = "flat"
    else:
        raise ValueError("Input features phải có shape 2D hoặc 3D.")

    x_norm = x_2d if already_l2_normalized else _normalize(x_2d).astype(np.float32)
    mean = np.asarray(scaler_state["mean"], dtype=np.float32)
    scale = np.asarray(scaler_state["scale"], dtype=np.float32)

    if x_norm.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Feature dim mismatch: input={x_norm.shape[1]}, model expects={mean.shape[0]}."
        )

    x_scaled = (x_norm - mean) / (scale + 1e-8)
    if original_layout == "legacy":
        return x_scaled.reshape(x_scaled.shape[0], legacy_seq_len, 1).astype(np.float32)
    if original_layout == "channel_frequency":
        return _reshape_flat_features_for_model(x_scaled, n_channels=original_channels)
    return x_scaled.astype(np.float32)

def _to_model_input_layout(features: np.ndarray, model: torch.nn.Module, selected_channels: list[int] | None = None) -> np.ndarray:
    expected_input_size = getattr(getattr(model, "lstm", None), "input_size", None)

    if features.ndim == 2:
        return _reshape_flat_features_for_model(features, n_channels=expected_input_size)

    if features.ndim != 3:
        raise ValueError("Input features phải có shape 2D hoặc 3D.")

    if features.shape[1] == N_FREQUENCIES:
        if expected_input_size is not None and features.shape[2] != expected_input_size:
            raise ValueError(
                f"Model yêu cầu input_size={expected_input_size} nhưng dữ liệu có {features.shape[2]} kênh."
            )
        return features.astype(np.float32)

    if features.shape[-1] == 1:
        flat = features.reshape(features.shape[0], features.shape[1]).astype(np.float32)
        return _reshape_flat_features_for_model(flat, n_channels=expected_input_size)

    if features.shape[2] == N_FREQUENCIES:
        flat = _flatten_model_features(features)
        return _reshape_flat_features_for_model(flat, n_channels=expected_input_size)

    channel_hint = len(selected_channels) if selected_channels else expected_input_size
    flat = _flatten_model_features(features)
    return _reshape_flat_features_for_model(flat, n_channels=channel_hint)

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

def _fit_model_for_target(
    target: str,
    processed_records: list[dict[str, Any]],
    channels: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    dropout: float,
    progress_callback: Callable[[int, int, float, float, float, float], None] | None = None,
) -> dict[str, Any]:
    x_train, y_train, x_test, y_test, scaler_state = _prepare_training_arrays(processed_records, channels, target)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("mrmr_lstm", seq_len=x_train.shape[1], input_size=x_train.shape[2], dropout=dropout).to(device)
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

        if progress_callback is not None:
            progress_callback(epoch + 1, int(epochs), tr_loss_ep, tr_acc_ep, va_loss_ep, va_acc_ep)

        if va_acc_ep >= best_val_acc:
            best_val_acc = va_acc_ep
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    checkpoint = {
        "model": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "seq_len": x_train.shape[1],
        "input_size": model.lstm.input_size,
        "target": target,
        "history": history,
        "scaler": scaler_state,
        "channels": channels,
        "selected_channels": channels,
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
    model_entry = _resolve_model_entry(target, file_manager)
    if model_entry is None or model_entry.get("model") is None:
        raise ValueError(f"Thiếu model cho {_target_label(target)}")

    checkpoint = model_entry.get("checkpoint", {}) if isinstance(model_entry, dict) else {}
    channels = checkpoint.get("channels") or checkpoint.get("selected_channels")
    if channels is None:
        raise ValueError(f"Model cho {_target_label(target)} không có thông tin channels MRMR.")
    channels = [int(value) for value in channels]
    scaler_state = checkpoint.get("scaler")

    model = model_entry["model"]
    x_flat, y_input = _prepare_prediction_inputs(processed_record, channels, target)
    x_norm = _normalize(x_flat).astype(np.float32)
    if scaler_state is not None:
        x_scaled = _apply_saved_scaler(x_norm, scaler_state, already_l2_normalized=True)
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_norm).astype(np.float32)
    x_input = _to_model_input_layout(x_scaled, model, channels)

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
    # --- CSS HACK: Thu gọn nút link text và Tối ưu hoá khu vực Upload Model ---
    st.markdown(
        """
        <style>
        /* CSS cho nút link file Text */
        [data-testid="stExpanderDetails"] button[kind="tertiary"] {
            padding: 0px !important;
            min-height: 22px !important;
            height: 22px !important;
            line-height: 1.2 !important;
            justify-content: flex-start !important;
        }
        [data-testid="stExpanderDetails"] button[kind="tertiary"] p {
            font-size: 15px !important; margin: 0 !important; color: #1f77b4;
        }
        [data-testid="stExpanderDetails"] button[kind="tertiary"]:hover p {
            text-decoration: underline; color: #ff4b4b;
        }
        
        /* CSS biến File Uploader thành nút gọn gàng, xoá chữ 200MB */
        [data-testid="stFileUploadDropzone"] {
            padding: 0px !important;
            min-height: 38px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 6px !important;
        }
        [data-testid="stFileUploadDropzone"] small {
            display: none !important; /* Xoá chữ Limit 200MB per file */
        }
        [data-testid="stFileUploadDropzone"] svg {
            display: none !important; /* Xoá icon đám mây */
        }
        [data-testid="stFileUploadDropzone"] div[data-testid="stMarkdownContainer"] {
            margin: 0 !important;
            line-height: 1 !important;
        }
        [data-testid="stFileUploadDropzone"] span {
            font-size: 14px !important;
            font-weight: 500 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    fm = _file_manager()
    st.subheader("🗂 Virtual File Manager")

    with st.expander("📂 Data", expanded=True):
        if fm["raw_data"]:
            for item in fm["raw_data"][:2]:
                st.button(f"• {item['name']}", key=f"raw_link_{item['name']}", type="tertiary", 
                          on_click=_on_file_link_click, args=("Load Data", "dashboard_load_data_selected_name", item["name"]))
            
            if len(fm["raw_data"]) > 2:
                if "expand_raw_data" not in st.session_state:
                    st.session_state.expand_raw_data = False
                remaining_count = len(fm["raw_data"]) - 2
                if not st.session_state.expand_raw_data:
                    if st.button(f"Mở rộng (+{remaining_count} file) ▾", key="btn_expand_raw_data", use_container_width=True):
                        st.session_state.expand_raw_data = True
                        st.rerun()
                else:
                    for item in fm["raw_data"][2:]:
                        st.button(f"• {item['name']}", key=f"raw_link_exp_{item['name']}", type="tertiary", 
                                  on_click=_on_file_link_click, args=("Load Data", "dashboard_load_data_selected_name", item["name"]))
                    if st.button("Thu gọn ▴", key="btn_collapse_raw_data", use_container_width=True):
                        st.session_state.expand_raw_data = False
                        st.rerun()
        else:
            st.caption("Chưa có file .dat nào.")

    with st.expander("⚡ Processed Data", expanded=True):
        if fm["processed_data"]:
            for item in fm["processed_data"][:2]:
                st.button(f"• {item['name']}", key=f"proc_link_{item['name']}", type="tertiary", 
                          on_click=_on_file_link_click, args=("Preprocess", "dashboard_preprocess_selected_name", item["name"]))
            
            if len(fm["processed_data"]) > 2:
                if "expand_processed_data" not in st.session_state:
                    st.session_state.expand_processed_data = False
                remaining_count = len(fm["processed_data"]) - 2
                if not st.session_state.expand_processed_data:
                    if st.button(f"Mở rộng (+{remaining_count} file) ▾", key="btn_expand_processed_data", use_container_width=True):
                        st.session_state.expand_processed_data = True
                        st.rerun()
                else:
                    for item in fm["processed_data"][2:]:
                        st.button(f"• {item['name']}", key=f"proc_link_exp_{item['name']}", type="tertiary", 
                                  on_click=_on_file_link_click, args=("Preprocess", "dashboard_preprocess_selected_name", item["name"]))
                    if st.button("Thu gọn ▴", key="btn_collapse_processed_data", use_container_width=True):
                        st.session_state.expand_processed_data = False
                        st.rerun()
        else:
            st.caption("Chưa có dữ liệu FFT.")

    with st.expander("🎓 Model", expanded=True):
        for target in TARGETS:
            entry = fm["models"].get(target)
            file_name = (entry or {}).get("name")
            
            # Hiển thị tiêu đề Target và Trạng thái
            st.markdown(f"<div style='font-size: 14.5px; margin-bottom: 5px; line-height: 1.2;'><b>{_target_label(target)}:</b> <span style='color: #666;'>{file_name if file_name else 'Trống'}</span></div>", unsafe_allow_html=True)
            
            # Khung chứa nút bấm (Thay đổi linh hoạt dựa trên việc có model hay chưa)
            if entry and entry.get("checkpoint"):
                col_up, col_down = st.columns(2, gap="small")
                with col_up:
                    upload = st.file_uploader(f"Up_{target}", type=["pth"], key=f"mgr_upl_{target}", label_visibility="collapsed")
                with col_down:
                    st.download_button(
                        "📥 Download", 
                        data=_checkpoint_to_bytes(entry["checkpoint"]),
                        file_name=file_name or f"{target}_mrmr_lstm.pth",
                        mime="application/octet-stream",
                        key=f"mgr_dwn_{target}",
                        use_container_width=True
                    )
            else:
                upload = st.file_uploader(f"Up_{target}", type=["pth"], key=f"mgr_upl_{target}", label_visibility="collapsed")

            # Auto-upload Logic
            if upload is not None:
                content = upload.getvalue()
                marker_key = f"manager_model_uploaded_marker_{target}"
                file_signature = (upload.name, len(content), hash(content))
                if st.session_state.get(marker_key) != file_signature:
                    try:
                        _upload_model_file(target, upload)
                        st.session_state[marker_key] = file_signature
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")
            
            # Khoảng cách giữa Arousal và Valence
            st.write("")

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
        st.info("4. Train Model\nTự chạy MRMR global và train theo nhãn")
    with col2:
        st.success("2. Preprocess\nTrích xuất FFT 5 dải tần")
        st.success("5. Download Model\nLưu checkpoint kèm channels + scaler")
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
                                    _store_raw_data([mock_file])
                                    status.update(label=f"Đã load {selected_test_file} siêu tốc!", state="complete")
                                st.rerun() 
                    else:
                        st.info(f"Thư mục `{TEST_DATA_DIR}` đang trống.")
                else:
                    st.warning(f"Chưa tìm thấy thư mục `{TEST_DATA_DIR}` trên server. Hãy tạo nó!")

        raw_records = _get_raw_records()
        if raw_records:
            preview_name_key = "dashboard_load_data_selected_name"
            record_names = [record["name"] for record in raw_records]
            options = ["None"] + record_names 
            current_name = st.session_state.get(preview_name_key, "None")
            if current_name not in options:
                current_name = "None"
                st.session_state[preview_name_key] = current_name

            # ĐIỂM NEO HTML ĐỂ CUỘN CHUỘT XUỐNG
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
            else:
                st.info("Vui lòng chọn một file phía trên để xem trước dữ liệu.")

        st.button("Tiếp tục → Preprocess", on_click=goto, args=("Preprocess",))

    _layout_with_manager(render_main)
    _trigger_scroll_if_needed() # Kích hoạt cuộn chuột nếu được yêu cầu từ File Manager

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
            processed = [None] * len(raw_records)
            progress = st.progress(0)
            status_text = st.empty()
            
            status_text.info(f"Đang phân bổ tác vụ FFT lên toàn bộ CPU cores (Đa tiến trình)...")
            
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_idx = {
                    executor.submit(
                        preprocess_subject_fft, 
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
                        processed[idx] = _normalize_processed_record(
                            record["name"], preprocessed, 
                            {"window_size": int(window_size), "step_size": int(step_size)}
                        )
                    except Exception as exc:
                        st.error(f"Lỗi khi xử lý {record['name']}: {exc}")
                        
                    completed += 1
                    status_text.info(f"Đang xử lý song song... Hoàn thành {completed}/{len(raw_records)} file.")
                    progress.progress(completed / len(raw_records))
            
            _file_manager()["processed_data"] = [p for p in processed if p is not None]
            status_text.success("Preprocess hoàn tất.")
            st.success(f"Đã trích xuất FFT cho {len(_file_manager()['processed_data'])} subject(s).")

        processed_records = _get_processed_records()
        if processed_records:
            preview_name_key = "dashboard_preprocess_selected_name"
            record_names = [record["name"] for record in processed_records]
            options = ["None"] + record_names
            current_name = st.session_state.get(preview_name_key, "None")
            
            if current_name not in options:
                current_name = "None"
                st.session_state[preview_name_key] = current_name

            # ĐIỂM NEO HTML ĐỂ CUỘN CHUỘT XUỐNG
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
                    ax.set_yticklabels([_channel_name(i) for i in range(sample_features.shape[0])], fontsize=6)
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

    _layout_with_manager(render_main)
    _trigger_scroll_if_needed() # Kích hoạt cuộn chuột nếu được yêu cầu từ File Manager

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

        with st.expander("⚙️ Advanced Hyperparameters (Epochs, LR, Batch Size, Dropout)", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            epochs = int(col_a.number_input("Epochs", value=50, min_value=1, max_value=500, step=5))
            lr = float(col_b.number_input("Learning rate", value=LR, min_value=1e-5, max_value=0.1, format="%.5f"))
            batch_size = int(col_c.number_input("Batch size", value=BATCH_SIZE, min_value=32, max_value=1024, step=32))
            dropout = float(st.slider("Dropout", 0.0, 0.8, 0.5, 0.05))
        
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

        # KHI BẤM NÚT BẮT ĐẦU TRAINING
        if start_train:
            st.session_state.runtime["mrmr_results"] = {}
            st.session_state.runtime["training_progress_state"] = {}
            st.session_state.runtime["training_results"] = {}

            # LẤY NGỮ CẢNH CỦA LUỒNG CHÍNH ĐỂ TRUYỀN CHO CÁC LUỒNG CON
            ctx = get_script_run_ctx()

            # --- 1. CHẠY MRMR SONG SONG ---
            with st.spinner("Đang chạy MRMR global song song..."):
                mrmr_results = {}
                channels_map = {}
                
                def thread_safe_mrmr(tgt):
                    # Bơm chính xác ctx vào luồng phụ
                    add_script_run_ctx(threading.current_thread(), ctx) 
                    res = _run_mrmr_task(tgt, processed_records, k_value)
                    _store_mrmr_result(tgt, res["channels"], source="computed", name=res["file_name"])
                    return tgt, res

                with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                    futures = [executor.submit(thread_safe_mrmr, target) for target in selected_targets]
                    for future in as_completed(futures):
                        tgt, mrmr_result = future.result()
                        channels_map[tgt] = mrmr_result["channels"]
                        mrmr_results[tgt] = mrmr_result
                
                st.session_state.runtime["mrmr_results"] = mrmr_results

            # Hiển thị text MRMR
            with mrmr_container:
                st.markdown("**Kết quả MRMR:**")
                for target in ["arousal", "valence"]:
                    if target in channels_map:
                        ch_names = [_channel_name(c) for c in channels_map[target]]
                        st.markdown(f"**{_target_label(target)}**: {', '.join(ch_names)}")
                st.markdown("---")

            # --- 2. CHẠY TRAINING SONG SONG ---
            results = {}
            
            def thread_safe_train(tgt):
                # Bơm chính xác ctx vào luồng phụ
                add_script_run_ctx(threading.current_thread(), ctx)
                
                def _on_epoch(epoch: int, total: int, tr_loss: float, tr_acc: float, va_loss: float, va_acc: float) -> None:
                    per_target = epoch / max(total, 1)
                    prog_text = (
                        f"**{_target_label(tgt)} - Epoch {epoch}/{total}** | "
                        f"Train Loss: `{tr_loss:.4f}` Acc: `{tr_acc:.3f}` | "
                        f"Val Loss: `{va_loss:.4f}` Acc: `{va_acc:.3f}`"
                    )
                    progress_ui[tgt]["bar"].progress(per_target)
                    progress_ui[tgt]["text"].markdown(prog_text)
                    
                    st.session_state.runtime["training_progress_state"][tgt] = {
                        "progress": per_target,
                        "text": prog_text
                    }

                res = _fit_model_for_target(
                    tgt,
                    processed_records,
                    channels_map[tgt],
                    epochs,
                    lr,
                    batch_size,
                    dropout,
                    progress_callback=_on_epoch,
                )
                _store_model_result(tgt, res["model"], res["checkpoint"], source="trained")
                return tgt, res

            with ThreadPoolExecutor(max_workers=len(selected_targets)) as executor:
                futures = [executor.submit(thread_safe_train, target) for target in selected_targets]
                for future in as_completed(futures):
                    tgt, result = future.result()
                    results[tgt] = result
            
            st.session_state.runtime["training_results"] = results
            st.rerun()

        # KHI KHÔNG BẤM NÚT (Load lại trang) -> Render lại state
        elif not start_train:
            mrmr_results = st.session_state.runtime.get("mrmr_results", {})
            if mrmr_results:
                with mrmr_container:
                    st.markdown("**Kết quả MRMR:**")
                    for target in ["arousal", "valence"]: 
                        if target in mrmr_results:
                            ch_names = [_channel_name(c) for c in mrmr_results[target]["channels"]]
                            st.markdown(f"**{_target_label(target)}**: {', '.join(ch_names)}")
                    st.markdown("---")

            state_prog = st.session_state.runtime.get("training_progress_state", {})
            for target in ["arousal", "valence"]:
                if target in state_prog:
                    progress_ui[target]["bar"].progress(state_prog[target]["progress"])
                    progress_ui[target]["text"].markdown(state_prog[target]["text"])

        # 3. KẾT QUẢ ĐỒ THỊ
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
                # Đã loại bỏ hoàn toàn st.download_button ở đây
                
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
        
        # --- HIỂN THỊ GIAO DIỆN KẾT QUẢ ĐÃ ĐƯỢC CHIA ĐÔI ---
        prediction_results = st.session_state.runtime.get("prediction_results", {})
        
        if prediction_results and "arousal" in prediction_results and "valence" in prediction_results:
            df_a = prediction_results["arousal"]["results"]
            df_v = prediction_results["valence"]["results"]
            
            # Đảm bảo index trùng khớp
            if len(df_a) != len(df_v):
                st.error("Lỗi: Số lượng cửa sổ dự đoán giữa Arousal và Valence không khớp!")
                return
            
            total_windows = len(df_a)
            # Bộ dữ liệu DEAP có 40 trial/subject.
            num_trials = 40
            # Số lượng sample (window) trên mỗi trial
            wpt = total_windows // num_trials 
            
            # Thêm cột trial_id để Groupby chính xác (Sử dụng numpy để tránh lỗi Index)
            df_a = df_a.copy()
            df_v = df_v.copy()
            df_a["trial_id"] = np.clip(np.arange(len(df_a)) // wpt, 0, num_trials - 1)
            df_v["trial_id"] = np.clip(np.arange(len(df_v)) // wpt, 0, num_trials - 1)
            
            trial_results = []
            
            # Duyệt qua từng trial_id (từ 0 đến 39)
            for tid in range(num_trials):
                # Lọc data của trial hiện tại
                sub_a = df_a[df_a["trial_id"] == tid]
                sub_v = df_v[df_v["trial_id"] == tid]
                
                # Nếu trial bị trống (do dữ liệu bị cắt ngắn), bỏ qua
                if len(sub_a) == 0 or len(sub_v) == 0:
                    continue
                
                # Lấy nhãn thực tế (vì trong 1 trial, DEAP chỉ có 1 nhãn duy nhất cho tất cả sample)
                a_true = sub_a["true_label"].values[0]
                v_true = sub_v["true_label"].values[0]
                
                # Bầu chọn đa số cho Trial
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
            
            # Tính toán % số lượng Trial đúng
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
            # Thanh kéo để chọn Trial (từ 1 đến actual_num_trials)
            selected_trial_idx = st.slider("🎯 Kéo để chọn Trial cần kiểm tra chi tiết", 1, actual_num_trials, 1) - 1
            curr_res = trial_results[selected_trial_idx]
            
            # Hiển thị thông tin chung của Trial được chọn
            chung_status = "✅ ĐÚNG CẢ 2" if curr_res["both_correct"] else "❌ SAI (Ít nhất 1 nhãn không khớp)"
            st.markdown(f"""
            <div style="background-color: #eaf5ff; border: 1px solid #cce0ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="margin-top: 0; color: #004d99;">Thông số Trial {curr_res['trial_id'] + 1}</h4>
                <b>• Nhãn thực tế:</b> Arousal = <b style='color: #e07b39;'>{curr_res['a_true']}</b> | Valence = <b style='color: #5b8dd9;'>{curr_res['v_true']}</b><br>
                <b>• Kết quả dự đoán (Đa số):</b> Arousal = <b>{curr_res['a_pred']}</b> | Valence = <b>{curr_res['v_pred']}</b> ➡️ <b>{chung_status}</b>
            </div>
            """, unsafe_allow_html=True)
            
            # Chia đôi màn hình cho Arousal và Valence
            col_a_ui, col_v_ui = st.columns(2, gap="medium")
            
            # --- CỘT BÊN TRÁI: AROUSAL ---
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
            
            # --- CỘT BÊN PHẢI: VALENCE ---
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
                st.write(f"**{_target_label(tgt)}**")
                st.dataframe(res["results"].head(20), use_container_width=True, hide_index=True)

    _layout_with_manager(render_main)

PAGE_FUNCS = {
    "Home": page_home,
    "Load Data": page_load_data,
    "Preprocess": page_preprocess,
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