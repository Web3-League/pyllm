"""
PyLLM - Python LLM Inference with Streaming Chat

A complete LLM inference platform with streaming responses,
chat interface, and support for multiple model backends.
"""

__version__ = "1.8.5"
__author__ = "nano3"

from pyllm.core.config import Config
from pyllm.inference.engine import InferenceEngine

__all__ = ["Config", "InferenceEngine", "__version__"]
