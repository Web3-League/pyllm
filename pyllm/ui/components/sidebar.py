"""Sidebar component."""

import streamlit as st

from pyllm.ui.api import get_client
from pyllm.ui.state import clear_messages

# Try to import streamlit_shadcn_ui, fallback to native streamlit
try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except ImportError:
    HAS_SHADCN = False


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## PyLLM Chat")

        # API status
        client = get_client()
        health = client.health()

        if health and health.get("model_loaded"):
            if HAS_SHADCN and hasattr(ui, 'badge'):
                ui.badge(text="Connected", variant="default", key="status_badge")
            else:
                st.success("Connected")
        elif health:
            if HAS_SHADCN and hasattr(ui, 'badge'):
                ui.badge(text="No Model", variant="secondary", key="status_badge")
            else:
                st.warning("No Model")
        else:
            if HAS_SHADCN and hasattr(ui, 'badge'):
                ui.badge(text="Offline", variant="destructive", key="status_badge")
            else:
                st.error("Offline")

        st.markdown("---")

        # Generation settings
        st.markdown("### Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            key="temperature",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=256,
            step=64,
            key="max_tokens",
        )

        st.markdown("---")

        # System prompt
        st.markdown("### System Prompt")
        system_prompt = st.text_area(
            "System",
            key="system_prompt",
            height=100,
            placeholder="You are a helpful assistant...",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Actions
        if HAS_SHADCN and hasattr(ui, 'button'):
            if ui.button("Clear Chat", key="clear_btn", variant="outline"):
                clear_messages()
                st.rerun()
        else:
            if st.button("Clear Chat", key="clear_btn"):
                clear_messages()
                st.rerun()

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
