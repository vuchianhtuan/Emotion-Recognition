"""Main entry point for the EEG Emotion Recognition application."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.state_management import init_state
from app.pages import render_app

st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main() -> None:
    """Main entry point."""
    init_state()
    render_app()

if __name__ == "__main__":
    main()
