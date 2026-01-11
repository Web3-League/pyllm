"""Chat components with Shadcn-inspired design."""

import streamlit as st
from typing import Dict
import html

# Try to import streamlit_shadcn_ui
try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except ImportError:
    HAS_SHADCN = False


def escape_html(text: str) -> str:
    """Escape HTML characters in text."""
    return html.escape(text)


def render_message(msg: Dict[str, str], index: int):
    """Render a chat message with improved styling."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    escaped_content = escape_html(content)

    if role == "user":
        st.markdown(f"""
        <div class="message-container">
            <div class="user-message">
                <div class="message-header">
                    <div class="message-avatar user-avatar">U</div>
                    <span class="message-role">You</span>
                </div>
                <div class="message-content">{escaped_content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="message-container">
            <div class="assistant-message">
                <div class="message-header">
                    <div class="message-avatar assistant-avatar">P</div>
                    <span class="message-role">Assistant</span>
                </div>
                <div class="message-content">{escaped_content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"""
        <div class="system-message">
            System: {escaped_content}
        </div>
        """, unsafe_allow_html=True)


def render_chat_input():
    """Render chat input area with Shadcn components."""
    col1, col2 = st.columns([6, 1])

    with col1:
        message = st.text_area(
            "Message",
            key="chat_input",
            height=80,
            placeholder="Type your message...",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        if HAS_SHADCN and hasattr(ui, 'button'):
            send = ui.button("Send", key="send_btn", variant="default")
        else:
            send = st.button("Send", key="send_btn", type="primary")

    return message if send else None


def render_streaming_message(content: str):
    """Render a streaming message (used during generation)."""
    escaped_content = escape_html(content)
    return f"""
    <div class="message-container">
        <div class="assistant-message">
            <div class="message-header">
                <div class="message-avatar assistant-avatar">P</div>
                <span class="message-role">Assistant</span>
            </div>
            <div class="message-content">{escaped_content}</div>
        </div>
    </div>
    """
