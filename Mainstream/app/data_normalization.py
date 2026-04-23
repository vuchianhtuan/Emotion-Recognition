"""Data normalization and record management utilities."""

from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from app.state_management import file_manager
from app.ui_helpers import selection_dataframe


def normalize_raw_record(name: str, subject: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw data record."""
    return {"name": name, "subject": subject}


def normalize_processed_record(name: str, data: np.ndarray, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize processed data record."""
    record = {"name": name, "data": data}
    if meta:
        record.update(meta)
    return record


def extract_channels(selection: Any) -> list[int]:
    """Extract channel indices from various selection formats."""
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


def resolve_mrmr_entry(target: str, file_manager_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Resolve MRMR selection entry for target."""
    fm = file_manager_state or file_manager()
    entry = fm["mrmr_selection"].get(target)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"name": None, "data": entry, "channels": extract_channels(entry)}


def resolve_model_entry(target: str, file_manager_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Resolve model entry for target."""
    fm = file_manager_state or file_manager()
    entry = fm["models"].get(target)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"name": None, "model": entry}


def get_processed_records() -> list[dict[str, Any]]:
    """Get all processed data records."""
    return file_manager()["processed_data"]


def get_raw_records() -> list[dict[str, Any]]:
    """Get all raw data records."""
    return file_manager()["raw_data"]
