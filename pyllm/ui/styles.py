"""Custom styles for PyLLM UI - Shadcn-inspired design."""

import streamlit as st


def apply_styles():
    """Apply custom CSS with Shadcn-inspired design system."""
    st.markdown("""
    <style>
    /* ===== CSS Variables (Shadcn theme) ===== */
    :root {
        --background: #09090b;
        --foreground: #fafafa;
        --card: #09090b;
        --card-foreground: #fafafa;
        --popover: #09090b;
        --popover-foreground: #fafafa;
        --primary: #fafafa;
        --primary-foreground: #18181b;
        --secondary: #27272a;
        --secondary-foreground: #fafafa;
        --muted: #27272a;
        --muted-foreground: #a1a1aa;
        --accent: #27272a;
        --accent-foreground: #fafafa;
        --destructive: #7f1d1d;
        --destructive-foreground: #fafafa;
        --border: #27272a;
        --input: #27272a;
        --ring: #d4d4d8;
        --radius: 0.5rem;
        --blue-primary: #3b82f6;
        --blue-hover: #2563eb;
        --blue-dark: #1e40af;
    }

    /* ===== Base ===== */
    .stApp {
        background-color: var(--background);
        color: var(--foreground);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== Header ===== */
    .header-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }

    .header-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--blue-primary) 0%, var(--blue-dark) 100%);
        border-radius: var(--radius);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.25rem;
        color: white;
    }

    .header-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--foreground);
    }

    /* ===== Welcome Screen ===== */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        max-width: 400px;
        margin: 0 auto;
    }

    .welcome-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 1.5rem;
        background: linear-gradient(135deg, var(--blue-primary) 0%, var(--blue-dark) 100%);
        border-radius: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 2rem;
        color: white;
    }

    .welcome-title {
        color: var(--foreground);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        color: var(--muted-foreground);
        font-size: 0.875rem;
    }

    /* ===== Chat Messages ===== */
    .message-container {
        margin: 1rem 0;
        animation: fadeIn 0.2s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background: var(--blue-dark);
        padding: 1rem 1.25rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.75rem 0;
        margin-left: 15%;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .assistant-message {
        background: var(--secondary);
        padding: 1rem 1.25rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.75rem 0;
        margin-right: 15%;
        border: 1px solid var(--border);
    }

    .message-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .message-avatar {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .user-avatar {
        background: var(--blue-primary);
        color: white;
    }

    .assistant-avatar {
        background: var(--accent);
        color: var(--foreground);
        border: 1px solid var(--border);
    }

    .message-role {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .message-content {
        color: var(--foreground);
        white-space: pre-wrap;
        line-height: 1.6;
        font-size: 0.9375rem;
    }

    .system-message {
        text-align: center;
        color: var(--muted-foreground);
        font-size: 0.875rem;
        padding: 0.75rem;
        font-style: italic;
        background: var(--muted);
        border-radius: var(--radius);
        margin: 0.5rem 2rem;
    }

    /* ===== Input Area ===== */
    .stTextArea textarea {
        background-color: var(--input) !important;
        border: 1px solid var(--border) !important;
        color: var(--foreground) !important;
        border-radius: var(--radius) !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9375rem !important;
        transition: border-color 0.2s ease;
    }

    .stTextArea textarea:focus {
        border-color: var(--ring) !important;
        box-shadow: 0 0 0 2px rgba(212, 212, 216, 0.2) !important;
    }

    .stTextArea textarea::placeholder {
        color: var(--muted-foreground) !important;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background-color: var(--blue-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: var(--blue-hover) !important;
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Outline button variant */
    .outline-button > button {
        background-color: transparent !important;
        color: var(--foreground) !important;
        border: 1px solid var(--border) !important;
    }

    .outline-button > button:hover {
        background-color: var(--accent) !important;
    }

    /* ===== Sidebar ===== */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: var(--background) !important;
        border-right: 1px solid var(--border);
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .sidebar-logo {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--blue-primary) 0%, var(--blue-dark) 100%);
        border-radius: 0.375rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        color: white;
    }

    .sidebar-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--foreground);
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }

    .status-connected {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
    }

    .status-connected .status-dot {
        background: #22c55e;
    }

    .status-warning {
        background: rgba(234, 179, 8, 0.1);
        color: #eab308;
    }

    .status-warning .status-dot {
        background: #eab308;
    }

    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }

    .status-error .status-dot {
        background: #ef4444;
    }

    /* ===== Sliders ===== */
    .stSlider > div > div > div {
        background-color: var(--blue-primary) !important;
    }

    .stSlider > div > div > div > div {
        background-color: var(--blue-primary) !important;
    }

    /* ===== Dividers ===== */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1rem 0 !important;
    }

    /* ===== Cards ===== */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem;
    }

    /* ===== Section Headers ===== */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--muted);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3f3f46;
    }

    /* ===== Spinner ===== */
    .stSpinner > div {
        border-color: var(--blue-primary) transparent transparent transparent !important;
    }

    /* ===== Markdown ===== */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--foreground);
    }

    .stMarkdown p {
        color: var(--foreground);
    }

    .stMarkdown code {
        background: var(--muted);
        padding: 0.125rem 0.375rem;
        border-radius: 0.25rem;
        font-size: 0.875em;
    }

    .stMarkdown pre {
        background: var(--muted) !important;
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }

    /* ===== Model Info Card ===== */
    .model-info {
        background: var(--secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }

    .model-name {
        font-weight: 500;
        color: var(--foreground);
        font-size: 0.875rem;
    }

    .model-details {
        font-size: 0.75rem;
        color: var(--muted-foreground);
        margin-top: 0.25rem;
    }

    /* ===== Sidebar fixes ===== */
    [data-testid="stSidebar"] {
        overflow-y: auto;
        overflow-x: hidden;
    }

    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto;
        overflow-x: hidden;
        height: 100vh;
        padding-bottom: 2rem;
    }

    /* Hide scrollbar but keep functionality */
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 4px;
    }

    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: var(--muted);
        border-radius: 2px;
    }

    /* Adjust main content */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
