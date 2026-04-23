"""Session state management for the application."""

from __future__ import annotations
from typing import Any
import streamlit as st
import streamlit.components.v1 as components


def init_state() -> None:
    """Initialize session state with default values."""
    defaults = {
        "page": "Home",
        "scroll_to_preview": False,
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


def ensure_state() -> None:
    """Ensure required state keys exist."""
    if "file_manager" not in st.session_state or "runtime" not in st.session_state:
        init_state()


def goto(page: str) -> None:
    """Navigate to a specific page."""
    st.session_state.page = page


def on_file_link_click(target_page: str, state_key: str, filename: str) -> None:
    """Callback when file link is clicked."""
    st.session_state.page = target_page
    st.session_state[state_key] = filename
    st.session_state.scroll_to_preview = True


def trigger_scroll_if_needed() -> None:
    """Trigger smooth scroll to preview section if needed."""
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


def file_manager() -> dict[str, Any]:
    """Get file manager from session state."""
    ensure_state()
    return st.session_state["file_manager"]


def snapshot_file_manager() -> dict[str, Any]:
    """Create a snapshot of current file manager state."""
    fm = file_manager()
    return {
        "raw_data": list(fm["raw_data"]),
        "processed_data": list(fm["processed_data"]),
        "mrmr_selection": dict(fm["mrmr_selection"]),
        "models": dict(fm["models"]),
    }
