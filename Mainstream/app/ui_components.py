"""UI components and panel rendering."""

from __future__ import annotations
from typing import Any
import streamlit as st
from app.config import TARGETS
from app.state_management import file_manager
from app.ui_helpers import target_label
from app.data_io import checkpoint_to_bytes, upload_model_file


def render_manager_panel() -> None:
    """Render the virtual file manager panel."""
    st.markdown(
        """
        <style>
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
        
        [data-testid="stFileUploadDropzone"] {
            padding: 0px !important;
            min-height: 38px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 6px !important;
        }
        [data-testid="stFileUploadDropzone"] small {
            display: none !important;
        }
        [data-testid="stFileUploadDropzone"] svg {
            display: none !important;
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
    
    fm = file_manager()
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
            
            st.markdown(f"<div style='font-size: 14.5px; margin-bottom: 5px; line-height: 1.2;'><b>{target_label(target)}:</b> <span style='color: #666;'>{file_name if file_name else 'Trống'}</span></div>", unsafe_allow_html=True)
            
            if entry and entry.get("checkpoint"):
                col_up, col_down = st.columns(2, gap="small")
                with col_up:
                    upload = st.file_uploader(f"Up_{target}", type=["pth"], key=f"mgr_upl_{target}", label_visibility="collapsed")
                with col_down:
                    st.download_button(
                        "📥 Download", 
                        data=checkpoint_to_bytes(entry["checkpoint"]),
                        file_name=file_name or f"{target}_mrmr_lstm.pth",
                        mime="application/octet-stream",
                        key=f"mgr_dwn_{target}",
                        use_container_width=True
                    )
            else:
                upload = st.file_uploader(f"Up_{target}", type=["pth"], key=f"mgr_upl_{target}", label_visibility="collapsed")

            if upload is not None:
                content = upload.getvalue()
                marker_key = f"manager_model_uploaded_marker_{target}"
                file_signature = (upload.name, len(content), hash(content))
                if st.session_state.get(marker_key) != file_signature:
                    try:
                        upload_model_file(target, upload)
                        st.session_state[marker_key] = file_signature
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")
            
            st.write("")


def _on_file_link_click(target_page: str, state_key: str, filename: str) -> None:
    """Callback when file link is clicked."""
    from app.state_management import on_file_link_click
    on_file_link_click(target_page, state_key, filename)


def layout_with_manager(main_render_fn) -> None:
    """Layout with manager panel."""
    main_col, manager_col = st.columns([3.5, 1.2], gap="medium")
    with main_col:
        main_render_fn()
    with manager_col:
        render_manager_panel()
