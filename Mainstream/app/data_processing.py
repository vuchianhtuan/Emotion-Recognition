"""Data processing and feature transformation utilities."""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize as _normalize, StandardScaler
from src.mrmr_selection import (
    N_FREQUENCIES,
    build_mrmr_dataset,
    prepare_for_lstm,
)


def get_processed_arrays(processed_records: list[dict[str, Any]]) -> list[np.ndarray]:
    """Build array list from processed records."""
    return [record["data"] for record in processed_records]


def prepare_training_arrays(
    processed_records: list[dict[str, Any]],
    channels: list[int],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prepare training data arrays."""
    processed_arrays = get_processed_arrays(processed_records)
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

    x_train = reshape_flat_features_for_model(x_train.reshape(x_train.shape[0], -1), n_channels=len(channels))
    x_test = reshape_flat_features_for_model(x_test.reshape(x_test.shape[0], -1), n_channels=len(channels))
    return x_train, y_train, x_test, y_test, scaler_state


def prepare_prediction_inputs(processed_record: dict[str, Any], channels: list[int], target: str) -> tuple[np.ndarray, np.ndarray]:
    """Prepare prediction input data."""
    data = processed_record["data"]
    
    x_list = []
    y_list = []
    for feat, lbl in data:
        x_list.append(feat[channels, :])
        y_list.append(lbl)

    x_all = np.array(x_list, dtype=np.float32)
    x_flat = x_all.reshape(x_all.shape[0], -1)
    y_all = np.array(y_list, dtype=np.float32)
    
    if y_all.max() > 1.0:
        y_bin = (y_all >= 5.0).astype(np.int64)
    else:
        y_bin = y_all.astype(np.int64)

    target_idx = 0 if target == "arousal" else 1
    y_target = y_bin[:, target_idx]

    return x_flat, y_target


def reshape_flat_features_for_model(x_2d: np.ndarray, n_channels: int | None = None) -> np.ndarray:
    """Reshape flat features to model input format."""
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


def flatten_model_features(x_3d: np.ndarray) -> np.ndarray:
    """Flatten 3D model features to 2D."""
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


def apply_saved_scaler(x: np.ndarray, scaler_state: dict, already_l2_normalized: bool = False) -> np.ndarray:
    """Apply saved scaler to features."""
    if x.ndim == 3:
        if x.shape[-1] == 1:
            x_2d = x.reshape(x.shape[0], x.shape[1]).astype(np.float32)
            original_layout = "legacy"
            legacy_seq_len = x.shape[1]
        else:
            x_2d = flatten_model_features(x)
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
        return reshape_flat_features_for_model(x_scaled, n_channels=original_channels)
    return x_scaled.astype(np.float32)


def to_model_input_layout(features: np.ndarray, model: torch.nn.Module, selected_channels: list[int] | None = None) -> np.ndarray:
    """Convert features to model input layout."""
    expected_input_size = getattr(getattr(model, "lstm", None), "input_size", None)

    if features.ndim == 2:
        return reshape_flat_features_for_model(features, n_channels=expected_input_size)

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
        return reshape_flat_features_for_model(flat, n_channels=expected_input_size)

    if features.shape[2] == N_FREQUENCIES:
        flat = flatten_model_features(features)
        return reshape_flat_features_for_model(flat, n_channels=expected_input_size)

    channel_hint = len(selected_channels) if selected_channels else expected_input_size
    flat = flatten_model_features(features)
    return reshape_flat_features_for_model(flat, n_channels=channel_hint)
