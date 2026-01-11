"""Sidebar component with Shadcn-inspired design."""

import streamlit as st

from pyllm.ui.api import get_client
from pyllm.ui.state import clear_messages

# Try to import streamlit_shadcn_ui
try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except ImportError:
    HAS_SHADCN = False


def render_status_badge(status: str, text: str):
    """Render a custom status badge."""
    status_class = {
        "connected": "status-connected",
        "warning": "status-warning",
        "error": "status-error",
    }.get(status, "status-warning")

    st.markdown(f"""
    <div class="status-badge {status_class}">
        <span class="status-dot"></span>
        {text}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(text: str):
    """Render a styled section header."""
    st.markdown(f"""
    <div class="section-header">{text}</div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with improved styling."""
    with st.sidebar:
        # Header with logo
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">P</div>
            <div class="sidebar-title">PyLLM</div>
        </div>
        """, unsafe_allow_html=True)

        # API status
        client = get_client()
        health = client.health()

        if health and health.get("model_loaded"):
            model_name = health.get("model_name", "Unknown")
            render_status_badge("connected", "Connected")

            # Model info card
            st.markdown(f"""
            <div class="model-info">
                <div class="model-name">{model_name}</div>
                <div class="model-details">Model loaded and ready</div>
            </div>
            """, unsafe_allow_html=True)
        elif health:
            render_status_badge("warning", "No Model")
            st.markdown("""
            <div class="model-info">
                <div class="model-name">No model loaded</div>
                <div class="model-details">Load a model to start chatting</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_status_badge("error", "Offline")
            st.markdown("""
            <div class="model-info">
                <div class="model-name">Server offline</div>
                <div class="model-details">Check if PyLLM server is running</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Generation settings
        render_section_header("Generation Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            key="temperature",
            help="Higher values make output more random, lower values more deterministic",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=256,
            step=64,
            key="max_tokens",
            help="Maximum number of tokens to generate",
        )

        st.markdown("---")

        # System prompt
        render_section_header("System Prompt")
        system_prompt = st.text_area(
            "System",
            key="system_prompt",
            height=100,
            placeholder="You are a helpful assistant...",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Actions
        render_section_header("Actions")

        col1, col2 = st.columns(2)

        with col1:
            if HAS_SHADCN and hasattr(ui, 'button'):
                if ui.button("Clear Chat", key="clear_btn", variant="outline"):
                    clear_messages()
                    st.rerun()
            else:
                st.markdown('<div class="outline-button">', unsafe_allow_html=True)
                if st.button("Clear Chat", key="clear_btn"):
                    clear_messages()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if HAS_SHADCN and hasattr(ui, 'button'):
                if ui.button("New Chat", key="new_btn", variant="outline"):
                    clear_messages()
                    st.rerun()
            else:
                st.markdown('<div class="outline-button">', unsafe_allow_html=True)
                if st.button("New Chat", key="new_btn"):
                    clear_messages()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="
            font-size: 0.75rem;
            color: #6b7280;
            text-align: center;
            padding: 0.5rem;
        ">
            PyLLM Inference v1.8.4
        </div>
        """, unsafe_allow_html=True)

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
