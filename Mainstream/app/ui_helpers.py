"""UI helper functions and utilities."""

from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from app.config import DEAP_ELECTRODES


def target_label(target: str) -> str:
    """Get display label for target."""
    return "Arousal" if target == "arousal" else "Valence"


def channel_name(index: int) -> str:
    """Get channel name or id."""
    if 0 <= index < len(DEAP_ELECTRODES):
        return DEAP_ELECTRODES[index]
    return f"Ch{index}"


def replace_by_name(items: list[dict[str, Any]], name: str, new_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace or add item in list by name field."""
    filtered = [item for item in items if item.get("name") != name]
    filtered.append(new_item)
    return filtered


def selection_dataframe(channels: list[int]) -> pd.DataFrame:
    """Create dataframe from selected channels."""
    return pd.DataFrame({
        "channels": [int(value) for value in channels],
        "channel_names": [channel_name(int(value)) for value in channels],
    })


def channel_dataframe(channels: list[int]) -> pd.DataFrame:
    """Create dataframe with channel information."""
    return pd.DataFrame({
        "channels": [int(channel) for channel in channels],
        "channel_names": [channel_name(int(channel)) for channel in channels],
    })
