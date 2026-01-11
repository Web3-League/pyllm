"""Custom Shadcn-inspired components for PyLLM UI.

These components provide Shadcn-style UI elements when streamlit-shadcn-ui
is not available or when custom styling is needed.
"""

import streamlit as st
from typing import Optional, List, Dict, Any

# Try to import streamlit_shadcn_ui
try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except ImportError:
    HAS_SHADCN = False


def card(title: Optional[str] = None, description: Optional[str] = None, content: Optional[str] = None):
    """Render a Shadcn-style card component."""
    html_parts = ['<div class="card">']

    if title:
        html_parts.append(f'<div class="card-title" style="font-weight: 600; font-size: 1rem; margin-bottom: 0.25rem;">{title}</div>')

    if description:
        html_parts.append(f'<div class="card-description" style="color: #a1a1aa; font-size: 0.875rem; margin-bottom: 0.75rem;">{description}</div>')

    if content:
        html_parts.append(f'<div class="card-content">{content}</div>')

    html_parts.append('</div>')

    st.markdown(''.join(html_parts), unsafe_allow_html=True)


def badge(text: str, variant: str = "default", key: Optional[str] = None):
    """Render a Shadcn-style badge component.

    Args:
        text: Badge text
        variant: One of "default", "secondary", "destructive", "outline"
        key: Optional unique key
    """
    if HAS_SHADCN and hasattr(ui, 'badge'):
        return ui.badge(text=text, variant=variant, key=key)

    colors = {
        "default": ("rgba(59, 130, 246, 0.1)", "#3b82f6"),
        "secondary": ("rgba(39, 39, 42, 1)", "#a1a1aa"),
        "destructive": ("rgba(239, 68, 68, 0.1)", "#ef4444"),
        "outline": ("transparent", "#fafafa"),
    }

    bg, color = colors.get(variant, colors["default"])
    border = "1px solid #27272a" if variant == "outline" else "none"

    st.markdown(f"""
    <span style="
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.625rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        background: {bg};
        color: {color};
        border: {border};
    ">{text}</span>
    """, unsafe_allow_html=True)


def alert(title: str, description: str = "", variant: str = "default"):
    """Render a Shadcn-style alert component.

    Args:
        title: Alert title
        description: Alert description
        variant: One of "default", "destructive"
    """
    colors = {
        "default": ("#27272a", "#fafafa", "#a1a1aa"),
        "destructive": ("rgba(127, 29, 29, 0.5)", "#ef4444", "#fca5a5"),
    }

    bg, title_color, desc_color = colors.get(variant, colors["default"])

    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid #27272a;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    ">
        <div style="font-weight: 500; color: {title_color}; margin-bottom: 0.25rem;">{title}</div>
        <div style="font-size: 0.875rem; color: {desc_color};">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def tabs(items: List[Dict[str, str]], key: str = "tabs") -> str:
    """Render a Shadcn-style tabs component.

    Args:
        items: List of dicts with "label" and "value" keys
        key: Unique key for the component

    Returns:
        Selected tab value
    """
    if HAS_SHADCN and hasattr(ui, 'tabs'):
        return ui.tabs(options=items, default_value=items[0]["value"], key=key)

    # Fallback to native streamlit tabs
    labels = [item["label"] for item in items]
    selected = st.radio("", labels, horizontal=True, label_visibility="collapsed", key=key)

    # Map label back to value
    for item in items:
        if item["label"] == selected:
            return item["value"]

    return items[0]["value"]


def progress(value: float, max_value: float = 100.0):
    """Render a Shadcn-style progress bar.

    Args:
        value: Current value
        max_value: Maximum value
    """
    percentage = min(100, max(0, (value / max_value) * 100))

    st.markdown(f"""
    <div style="
        background: #27272a;
        border-radius: 9999px;
        height: 8px;
        overflow: hidden;
    ">
        <div style="
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            width: {percentage}%;
            height: 100%;
            transition: width 0.3s ease;
        "></div>
    </div>
    """, unsafe_allow_html=True)


def separator():
    """Render a Shadcn-style separator/divider."""
    st.markdown("""
    <div style="
        border-top: 1px solid #27272a;
        margin: 1rem 0;
    "></div>
    """, unsafe_allow_html=True)


def skeleton(width: str = "100%", height: str = "1rem"):
    """Render a Shadcn-style skeleton loading placeholder.

    Args:
        width: CSS width
        height: CSS height
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #27272a 25%, #3f3f46 50%, #27272a 75%);
        background-size: 200% 100%;
        animation: skeleton-loading 1.5s infinite;
        border-radius: 0.375rem;
        width: {width};
        height: {height};
    "></div>
    <style>
    @keyframes skeleton-loading {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}
    </style>
    """, unsafe_allow_html=True)


def avatar(text: str, size: str = "md", variant: str = "default"):
    """Render a Shadcn-style avatar component.

    Args:
        text: Text to display (usually initials)
        size: One of "sm", "md", "lg"
        variant: One of "default", "primary"
    """
    sizes = {
        "sm": ("24px", "0.75rem"),
        "md": ("32px", "0.875rem"),
        "lg": ("48px", "1.25rem"),
    }

    colors = {
        "default": ("#27272a", "#fafafa"),
        "primary": ("#3b82f6", "#ffffff"),
    }

    dim, font = sizes.get(size, sizes["md"])
    bg, color = colors.get(variant, colors["default"])

    st.markdown(f"""
    <div style="
        width: {dim};
        height: {dim};
        border-radius: 50%;
        background: {bg};
        color: {color};
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: {font};
        font-weight: 600;
    ">{text}</div>
    """, unsafe_allow_html=True)


def tooltip(content: str, text: str):
    """Render text with a tooltip on hover.

    Args:
        content: Main text content
        text: Tooltip text
    """
    st.markdown(f"""
    <span style="
        position: relative;
        cursor: help;
        border-bottom: 1px dotted #6b7280;
    " title="{text}">{content}</span>
    """, unsafe_allow_html=True)


def kbd(text: str):
    """Render a keyboard shortcut indicator.

    Args:
        text: Keyboard shortcut text (e.g., "Ctrl+Enter")
    """
    st.markdown(f"""
    <kbd style="
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 0.25rem;
        padding: 0.125rem 0.375rem;
        font-size: 0.75rem;
        font-family: monospace;
        color: #a1a1aa;
    ">{text}</kbd>
    """, unsafe_allow_html=True)
