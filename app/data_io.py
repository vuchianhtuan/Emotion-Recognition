"""Data input/output operations."""

from __future__ import annotations
from typing import Any
import io
import pickle
import pandas as pd
import torch
import torch.nn as nn
from app.state_management import file_manager
from app.data_normalization import (
    extract_channels,
    normalize_raw_record,
    normalize_processed_record,
    resolve_mrmr_entry,
)
from app.ui_helpers import selection_dataframe, replace_by_name


def store_raw_data(uploaded_files: list[Any]) -> None:
    """Store uploaded raw data files."""
    raw_records = file_manager()["raw_data"]
    for file_obj in uploaded_files:
        subject = pickle.load(io.BytesIO(file_obj.read()), encoding="latin1")
        raw_records = replace_by_name(raw_records, file_obj.name, normalize_raw_record(file_obj.name, subject))
    file_manager()["raw_data"] = raw_records


def store_processed_data(name: str, data: np.ndarray, meta: dict[str, Any] | None = None) -> None:
    """Store processed data."""
    import numpy as np
    processed_records = file_manager()["processed_data"]
    processed_records = replace_by_name(processed_records, name, normalize_processed_record(name, data, meta))
    file_manager()["processed_data"] = processed_records


def store_mrmr_result(target: str, channels: list[int], source: str, name: str | None = None) -> None:
    """Store MRMR selection result."""
    file_manager()["mrmr_selection"][target] = {
        "name": name or f"mrmr_{target}.xlsx",
        "data": selection_dataframe(channels),
        "channels": [int(value) for value in channels],
        "source": source,
    }


def store_model_result(target: str, model: nn.Module, checkpoint: dict[str, Any], source: str, name: str | None = None) -> None:
    """Store model training result."""
    file_manager()["models"][target] = {
        "name": name or f"{target}_mrmr_lstm.pth",
        "model": model,
        "checkpoint": checkpoint,
        "source": source,
    }


def selection_to_download_bytes(target: str) -> bytes:
    """Convert MRMR selection to downloadable bytes."""
    entry = resolve_mrmr_entry(target)
    if entry is None:
        return b""
    df = entry["data"]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=target.capitalize())
    return output.getvalue()


def checkpoint_to_bytes(checkpoint: dict[str, Any]) -> bytes:
    """Convert checkpoint to downloadable bytes."""
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def load_uploaded_checkpoint(payload: bytes) -> Any:
    """Load checkpoint across PyTorch versions."""
    buffer = io.BytesIO(payload)
    try:
        return torch.load(buffer, map_location="cpu")
    except Exception as exc:
        message = str(exc)
        requires_legacy_unpickle = (
            "Weights only load failed" in message
            or "Unsupported global" in message
            or "weights_only" in message
        )
        if not requires_legacy_unpickle:
            raise

        buffer.seek(0)
        try:
            return torch.load(buffer, map_location="cpu", weights_only=False)
        except TypeError:
            buffer.seek(0)
            return torch.load(buffer, map_location="cpu")


def load_model_from_checkpoint(checkpoint: Any) -> nn.Module:
    """Load model from checkpoint."""
    from src.models import build_model
    
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


def upload_mrmr_file(target: str, uploaded_file: Any) -> None:
    """Upload and store MRMR selection file."""
    if uploaded_file is None:
        raise ValueError("Chưa chọn file MRMR để upload")
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    else:
        df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
    channels = extract_channels(df)
    file_manager()["mrmr_selection"][target] = {
        "name": uploaded_file.name,
        "data": df,
        "channels": channels,
        "source": "upload",
    }


def upload_model_file(target: str, uploaded_file: Any) -> None:
    """Upload and store model file."""
    from app.config import TARGETS
    
    if uploaded_file is None:
        raise ValueError("Chưa chọn file model để upload")
    checkpoint = load_uploaded_checkpoint(uploaded_file.getvalue())
    model = load_model_from_checkpoint(checkpoint)
    resolved_target = target
    if isinstance(checkpoint, dict) and checkpoint.get("target") in TARGETS:
        resolved_target = str(checkpoint.get("target"))
    store_model_result(
        resolved_target,
        model,
        checkpoint if isinstance(checkpoint, dict) else {"model": model.state_dict()},
        "upload",
        uploaded_file.name,
    )
