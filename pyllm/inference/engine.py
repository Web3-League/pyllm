"""
PyLLM Inference Engine - Simplified for Complexity/ComplexityDeep models.
"""

import logging
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator, List
from tokenizers import Tokenizer as HFTokenizer

logger = logging.getLogger("pyllm.engine")


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "complexity-deep"
    path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "float16"
    max_seq_len: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256


@dataclass
class GenerationConfig:
    """Generation configuration."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256
    do_sample: bool = True
    # INL Dynamics for inference (velocity-aware generation) - ENABLED BY DEFAULT
    use_dynamics: bool = True
    dynamics_strength: float = 0.1
    dynamics_alpha: float = 0.9   # Inertia (momentum)
    dynamics_beta: float = 0.1    # Correction strength


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str


class TokenizerWrapper:
    """Simple tokenizer wrapper."""

    def __init__(self, tokenizer: HFTokenizer, config: dict = None):
        self._tokenizer = tokenizer
        self._config = config or {}
        self.eos_token_id = self._config.get("eos_token_id", 0)
        self.bos_token_id = self._config.get("bos_token_id", 2)
        self.pad_token_id = self._config.get("pad_token_id", 1)

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = self._tokenizer.encode(text).ids
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class InferenceEngine:
    """
    Simplified inference engine for Complexity/ComplexityDeep models.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    def _detect_device(self) -> torch.device:
        """Detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self, model_path: str) -> None:
        """Load model from path."""
        path = Path(model_path)

        # Handle both directory and file paths
        if path.is_file():
            model_dir = path.parent
        else:
            model_dir = path

        logger.info(f"Loading model from {model_dir}")

        # Detect device
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")

        # Load config.json
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")

        import json
        with open(config_path) as f:
            model_config = json.load(f)

        model_type = model_config.get("model_type", "")
        architectures = model_config.get("architectures", [])

        logger.info(f"Model type: {model_type}, architectures: {architectures}")

        # Try ComplexityDeep first (with INL Dynamics)
        model_loaded = False

        if "complexity" in model_type.lower() or "DeepForCausalLM" in str(architectures):
            try:
                from complexity_deep import DeepForCausalLM
                logger.info("Trying ComplexityDeep...")
                self.model = DeepForCausalLM.from_pretrained(
                    str(model_dir),
                    device=str(self.device)
                )
                model_loaded = True
                logger.info("Loaded ComplexityDeep model successfully")
            except ImportError:
                logger.warning("complexity_deep not installed")
            except Exception as e:
                logger.error(f"ComplexityDeep load failed: {e}")

        # Fallback to basic Complexity
        if not model_loaded:
            try:
                from complexity import ComplexityForCausalLM
                logger.info("Trying Complexity...")
                self.model = ComplexityForCausalLM.from_pretrained(
                    str(model_dir),
                    device=str(self.device)
                )
                model_loaded = True
                logger.info("Loaded Complexity model successfully")
            except ImportError:
                logger.warning("complexity not installed")
            except Exception as e:
                logger.error(f"Complexity load failed: {e}")

        if not model_loaded:
            raise RuntimeError(
                "Could not load model. Install: pip install complexity-deep"
            )

        self.model.eval()

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        raw_tokenizer = HFTokenizer.from_file(str(tokenizer_path))

        # Load tokenizer config
        tokenizer_config = {}
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                tokenizer_config = json.load(f)

        # Get special token IDs from model config
        tokenizer_config["eos_token_id"] = model_config.get("eos_token_id", 0)
        tokenizer_config["bos_token_id"] = model_config.get("bos_token_id", 2)
        tokenizer_config["pad_token_id"] = model_config.get("pad_token_id", 1)

        self.tokenizer = TokenizerWrapper(raw_tokenizer, tokenizer_config)

        self._loaded = True
        logger.info("Model and tokenizer loaded successfully")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """Generate text from prompt with streaming."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = config or GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        logger.info(f"Generating with temp={config.temperature}, top_k={config.top_k}, top_p={config.top_p}")
        if config.use_dynamics:
            logger.info(f"INL Dynamics enabled: strength={config.dynamics_strength}, alpha={config.dynamics_alpha}, beta={config.dynamics_beta}")

        # Use model's native generate
        with torch.no_grad():
            # Build generate kwargs - only pass dynamics params if model supports them
            generate_kwargs = {
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature if config.temperature > 0 else 1.0,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "do_sample": config.do_sample and config.temperature > 0,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            # Add full INL Dynamics params (PID with mu)
            if config.use_dynamics:
                generate_kwargs["use_dynamics"] = True
                generate_kwargs["dynamics_strength"] = config.dynamics_strength
                generate_kwargs["dynamics_alpha"] = config.dynamics_alpha
                generate_kwargs["dynamics_beta"] = config.dynamics_beta

            output_ids = self.model.generate(input_ids, **generate_kwargs)

        # Yield new tokens only
        new_tokens = output_ids[0, input_ids.shape[1]:]
        for token_id in new_tokens:
            if token_id.item() == self.tokenizer.eos_token_id:
                break
            token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=True)
            yield token_text

    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """Chat - just uses last message content as prompt."""
        prompt = messages[-1].content if messages else ""
        yield from self.generate(prompt, config)

    def complete(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate complete response (non-streaming)."""
        return "".join(self.generate(prompt, config))

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
