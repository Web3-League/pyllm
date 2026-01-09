#!/usr/bin/env python3
"""Setup script for PyLLM."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyllm-inference",
    version="1.4.3",
    author="nano3",
    author_email="",
    description="PyLLM: LLM Inference with Streaming Chat and OpenAI-compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Web3-League/pyllm",
    project_urls={
        "Documentation": "https://github.com/Web3-League/pyllm#readme",
        "Bug Reports": "https://github.com/Web3-League/pyllm/issues",
        "Source": "https://github.com/Web3-League/pyllm",
    },
    packages=find_packages(),
    package_data={
        "pyllm": ["ui/*.py", "ui/**/*.py"],
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "safetensors>=0.3.0",
        "tokenizers>=0.15.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "ui": [
            "streamlit>=1.28.0",
            "streamlit-shadcn-ui>=0.1.0",
            "requests>=2.31.0",
        ],
        "inl": [
            "inl-llm>=3.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyllm=pyllm.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm inference streaming chat openai transformers",
    license="MIT",
)
