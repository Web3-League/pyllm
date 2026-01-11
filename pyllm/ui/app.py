"""
PyLLM Chat - Streamlit Application.

LLM chat interface with streaming responses.
"""

import streamlit as st

from pyllm.ui.state import init_session_state, get_messages, add_message, set_generating, is_generating
from pyllm.ui.styles import apply_styles
from pyllm.ui.api import get_client
from pyllm.ui.components.chat import render_message, render_chat_input, render_streaming_message
from pyllm.ui.components.sidebar import render_sidebar

# Try to import streamlit_shadcn_ui
try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except ImportError:
    HAS_SHADCN = False


def main():
    """Main application."""
    st.set_page_config(
        page_title="PyLLM Chat",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_styles()
    init_session_state()

    # Sidebar
    settings = render_sidebar()

    # Main content header
    st.markdown("""
    <div class="header-container">
        <div class="header-logo">P</div>
        <div class="header-title">PyLLM Chat</div>
    </div>
    """, unsafe_allow_html=True)

    # Messages container
    messages_container = st.container()

    with messages_container:
        messages = get_messages()

        if not messages:
            st.markdown("""
            <div class="welcome-container">
                <div class="welcome-icon">P</div>
                <h3 class="welcome-title">Welcome to PyLLM Chat</h3>
                <p class="welcome-subtitle">Start a conversation by typing a message below.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, msg in enumerate(messages):
                render_message(msg, i)

    st.markdown("---")

    # Chat input
    user_message = render_chat_input()

    if user_message and not is_generating():
        # Add user message
        add_message("user", user_message)

        # Build messages for API
        api_messages = []

        if settings["system_prompt"]:
            api_messages.append({
                "role": "system",
                "content": settings["system_prompt"]
            })

        for msg in get_messages():
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Stream response
        set_generating(True)

        with st.spinner("Generating..."):
            client = get_client()
            response_tokens = []

            response_placeholder = st.empty()

            for token in client.chat_stream(
                api_messages,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
            ):
                response_tokens.append(token)
                response_placeholder.markdown(
                    render_streaming_message("".join(response_tokens)),
                    unsafe_allow_html=True
                )

            response = "".join(response_tokens)
            add_message("assistant", response)

        set_generating(False)
        st.rerun()


if __name__ == "__main__":
    main()
