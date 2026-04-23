"""Model training, prediction, and MRMR utilities."""

from __future__ import annotations
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize as _normalize, StandardScaler
from src.models import build_model
from src.mrmr_selection import (
    build_mrmr_dataset,
    prepare_for_lstm,
    run_mrmr_global_selection,
)
from app.data_processing import (
    get_processed_arrays,
    prepare_training_arrays,
    prepare_prediction_inputs,
    apply_saved_scaler,
    to_model_input_layout,
)
from app.ui_helpers import channel_dataframe
from app.data_normalization import resolve_model_entry


def run_mrmr_task(target: str, processed_records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Run MRMR feature selection task."""
    channels = run_mrmr_global_selection(get_processed_arrays(processed_records), classify_type=target, K=k)
    return {
        "target": target,
        "channels": channels,
        "dataframe": channel_dataframe(channels),
        "file_name": f"mrmr_{target}.xlsx",
    }


def predict_windows(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run prediction on windows."""
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


def fit_model_for_target(
    target: str,
    processed_records: list[dict[str, Any]],
    channels: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    dropout: float,
    progress_callback: Callable[[int, int, float, float, float, float], None] | None = None,
) -> dict[str, Any]:
    """Train model for target."""
    x_train, y_train, x_test, y_test, scaler_state = prepare_training_arrays(processed_records, channels, target)

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


def predict_target(target: str, processed_record: dict[str, Any], file_manager: dict[str, Any]) -> dict[str, Any]:
    """Run prediction for target."""
    model_entry = resolve_model_entry(target, file_manager)
    if model_entry is None or model_entry.get("model") is None:
        from app.ui_helpers import target_label
        raise ValueError(f"Thiếu model cho {target_label(target)}")

    checkpoint = model_entry.get("checkpoint", {}) if isinstance(model_entry, dict) else {}
    channels = checkpoint.get("channels") or checkpoint.get("selected_channels")
    if channels is None:
        from app.ui_helpers import target_label
        raise ValueError(f"Model cho {target_label(target)} không có thông tin channels MRMR.")
    channels = [int(value) for value in channels]
    scaler_state = checkpoint.get("scaler")

    model = model_entry["model"]
    x_flat, y_input = prepare_prediction_inputs(processed_record, channels, target)
    x_norm = _normalize(x_flat).astype(np.float32)
    if scaler_state is not None:
        x_scaled = apply_saved_scaler(x_norm, scaler_state, already_l2_normalized=True)
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_norm).astype(np.float32)
    x_input = to_model_input_layout(x_scaled, model, channels)

    probs, preds = predict_windows(model, x_input)
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
