# PyLLM

**Python LLM Inference Server** with streaming, OpenAI-compatible API, and chat interface.

[![PyPI version](https://badge.fury.io/py/pyllm-inference.svg)](https://badge.fury.io/py/pyllm-inference)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Streaming Generation** - Token-by-token streaming output
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI endpoints
- **Multiple Backends** - Complexity, Complexity-Deep, HuggingFace Transformers
- **Image Generation** - ComplexityDiT + ComplexityVAE support
- **Chat Templates** - ChatML, Llama, Mistral, Alpaca, etc.
- **Streamlit UI** - Beautiful chat interface
- **KV Cache** - Fast autoregressive generation

## Installation

```bash
pip install pyllm-inference

# With Streamlit UI
pip install pyllm-inference[ui]

# With Complexity models
pip install pyllm-inference complexity complexity-deep
```

## Supported Models

| Package | Description | Auto-detected |
|---------|-------------|---------------|
| `complexity` | Token-Routed MLP, QK Norm | Yes |
| `complexity-deep` | + INL Dynamics | Yes |
| `complexity-diffusion` | DiT + VAE for images | Yes |
| `transformers` | HuggingFace models | Yes |

## Quick Start

### Start the API Server

```bash
# Start server with model
pyllm serve --model path/to/model

# With specific port
pyllm serve --model Pacific-Prime/complexity-tiny --port 8000
```

### Start the Chat UI

```bash
pyllm ui --api-url http://localhost:8000
```

### Command Line Generation

```bash
# Generate text
pyllm generate --model path/to/model --prompt "Hello, world!"

# Interactive chat
pyllm chat --model path/to/model
```

## Python Usage

```python
from pyllm.inference import InferenceEngine, GenerationConfig

# Load model (auto-detects architecture)
engine = InferenceEngine()
engine.load("Pacific-Prime/complexity-tiny")

# Generate with streaming
for token in engine.generate("The quick brown fox"):
    print(token, end="", flush=True)

# With custom config
config = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
)
for token in engine.generate("Hello", config):
    print(token, end="", flush=True)
```

## Image Generation

```python
from pyllm.inference.diffusion import DiffusionEngine, ImageGenerationConfig

# Load models
engine = DiffusionEngine()
engine.load(
    vae_path="path/to/vae.safetensors",
    dit_path="path/to/dit.safetensors",
)

# Generate image
config = ImageGenerationConfig(
    width=256,
    height=256,
    num_inference_steps=50,
)
result = engine.generate("a beautiful landscape", config)

# Save
result.to_pil().save("output.png")
```

## API Endpoints

### OpenAI-Compatible

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (streaming) |
| POST | `/v1/generate` | Text generation (streaming) |

### Example: With OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="pyllm",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Architecture Detection

PyLLM automatically detects model architecture from:

1. `config.json` - `model_type` field
2. State dict keys - `mlp.experts`, `dynamics`, `q_norm`
3. Architecture field - `ComplexityForCausalLM`, `DeepForCausalLM`

```python
# These all work automatically:
engine.load("Pacific-Prime/complexity-tiny")  # complexity
engine.load("path/to/complexity-deep-model")  # complexity-deep (has dynamics)
engine.load("gpt2")  # transformers
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYLLM_MODEL_PATH` | Model path | None |
| `PYLLM_DEVICE` | Device (cuda/cpu) | cuda |
| `PYLLM_HOST` | Server host | 0.0.0.0 |
| `PYLLM_PORT` | Server port | 8000 |

### Config File

```json
{
  "model": {
    "path": "path/to/model",
    "device": "cuda",
    "max_seq_len": 2048
  },
  "generation": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2
  }
}
```

## Related Packages

- **complexity** - Base LLM architecture
- **complexity-deep** - LLM with INL Dynamics
- **complexity-diffusion** - DiT for image generation

## License

MIT
