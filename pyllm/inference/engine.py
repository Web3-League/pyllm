"""LLM Inference Engine - supports multiple backends."""

import logging
from typing import Optional, Generator, Dict, Any, List
from dataclasses import dataclass
import torch

from pyllm.core.config import ModelConfig

logger = logging.getLogger("pyllm.inference")


@dataclass
class Message:
    """Chat message."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class GenerationConfig:
    """Generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256
    do_sample: bool = True
    stream: bool = True


class INLTokenizerWrapper:
    """Wrapper for HuggingFace tokenizers to provide consistent API."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.eos_token_id = 2  # </s>
        self.pad_token_id = 0  # <pad>
        self.bos_token_id = 1  # <s>

    def __call__(self, text: str, return_tensors: str = None, **kwargs):
        encoding = self._tokenizer.encode(text)
        input_ids = encoding.ids

        if return_tensors == "pt":
            return type('Encoding', (), {'input_ids': torch.tensor([input_ids])})()

        return type('Encoding', (), {'input_ids': input_ids})()

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids


class InferenceEngine:
    """
    LLM Inference Engine.

    Supports:
    - INL-LLM models
    - HuggingFace transformers
    - Streaming generation
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model."""
        path = model_path or self.config.path

        if not path:
            raise ValueError("Model path required")

        logger.info(f"Loading model from {path}")

        # Determine device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        # Try loading as INL-LLM first
        if self._try_load_inl_llm(path):
            logger.info("Loaded as INL-LLM model")
        elif self._try_load_transformers(path):
            logger.info("Loaded as Transformers model")
        else:
            raise RuntimeError(f"Could not load model from {path}")

        self._loaded = True
        logger.info("Model loaded successfully")

    def _try_load_inl_llm(self, path: str) -> bool:
        """Try loading as INL-LLM model."""
        try:
            import json
            from pathlib import Path
            from tokenizers import Tokenizer as HFTokenizer

            path_obj = Path(path)

            # Determine model directory and weights file
            if path_obj.is_file() and (path.endswith(".safetensors") or path.endswith(".pt")):
                model_dir = path_obj.parent
                weights_path = path_obj
            elif path_obj.is_dir():
                model_dir = path_obj
                if (model_dir / "model.safetensors").exists():
                    weights_path = model_dir / "model.safetensors"
                elif (model_dir / "pytorch_model.bin").exists():
                    weights_path = model_dir / "pytorch_model.bin"
                else:
                    return False
            else:
                return False

            # Load config.json if it exists
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    model_config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
            else:
                logger.warning("No config.json found, using defaults")
                model_config = {}

            # Extract model parameters - support both HuggingFace and INL style
            if "inl_config" in model_config:
                inl_cfg = model_config["inl_config"]
                vocab_size = model_config.get("vocab_size", 100000)
                d_model = inl_cfg.get("d_model", 768)
                num_layers = inl_cfg.get("num_layers", 12)
                num_heads = inl_cfg.get("num_heads", 12)
                num_kv_heads = inl_cfg.get("num_kv_heads", 4)
                feedforward_dim = inl_cfg.get("feedforward_dim", 3072)
                num_iterations = inl_cfg.get("num_iterations_per_layer", 2)
            else:
                vocab_size = model_config.get("vocab_size", 100000)
                d_model = model_config.get("hidden_size", 768)
                num_layers = model_config.get("num_hidden_layers", 12)
                num_heads = model_config.get("num_attention_heads", 12)
                num_kv_heads = model_config.get("num_key_value_heads", 4)
                feedforward_dim = model_config.get("intermediate_size", 3072)
                num_iterations = model_config.get("num_iterations_per_layer", 2)

            logger.info(f"Model config: vocab={vocab_size}, d_model={d_model}, layers={num_layers}, heads={num_heads}")

            # Try INL-LLM v3 first, then v2
            model_loaded = False

            try:
                from inl_llm_v3.models.integrator_language_model import UltraOptimizedIntegratorLanguageModel
                self.model = UltraOptimizedIntegratorLanguageModel(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    feedforward_dim=feedforward_dim,
                    num_iterations_per_layer=num_iterations,
                    max_seq_len=self.config.max_seq_len
                )
                model_loaded = True
                logger.info("Using INL-LLM v3 architecture")
            except ImportError:
                pass

            if not model_loaded:
                try:
                    from inl_llm import IntegratorLanguageModel
                    self.model = IntegratorLanguageModel(
                        vocab_size=vocab_size,
                        d_model=d_model,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        num_iterations_per_layer=num_iterations,
                        feedforward_dim=feedforward_dim,
                        max_seq_len=self.config.max_seq_len
                    )
                    model_loaded = True
                    logger.info("Using INL-LLM v2 architecture")
                except ImportError:
                    pass

            if not model_loaded:
                logger.error("No INL-LLM module found. Install with: pip install inl-llm-v3")
                return False

            # Load weights
            if str(weights_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_path))
            else:
                checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Load tokenizer
            tokenizer_path = model_dir / "tokenizer.json"
            if tokenizer_path.exists():
                self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
                self.tokenizer = INLTokenizerWrapper(self.tokenizer)
                logger.info(f"Loaded tokenizer from {tokenizer_path}")
            else:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("No tokenizer.json found, using GPT-2 tokenizer")

            return True

        except Exception as e:
            logger.debug(f"INL-LLM load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _try_load_transformers(self, path: str) -> bool:
        """Try loading as HuggingFace Transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=getattr(torch, self.config.dtype),
                device_map="auto" if self.device.type == "cuda" else None,
            )

            if self.device.type != "cuda":
                self.model.to(self.device)

            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            return True

        except Exception as e:
            logger.debug(f"Transformers load failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text from prompt with streaming.

        Yields tokens one at a time.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = config or GenerationConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=self.config.max_new_tokens,
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        # Generate with streaming
        with torch.no_grad():
            for token in self._generate_tokens(input_ids, config):
                yield token

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> Generator[str, None, None]:
        """Generate tokens one at a time."""
        generated = input_ids.clone()
        past_tokens = set(input_ids[0].tolist())

        for _ in range(config.max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(generated)

                # Get logits for last token
                # INL-LLM returns (logits, iterations, budget_info)
                if isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                for token_id in past_tokens:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= config.repetition_penalty
                    else:
                        logits[0, token_id] /= config.repetition_penalty

            # Apply temperature
            if config.temperature > 0:
                logits = logits / config.temperature

            # Apply top-k
            if config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus sampling)
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            if config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            # Update state
            generated = torch.cat([generated, next_token], dim=-1)
            past_tokens.add(next_token.item())

            # Truncate if needed
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]

    def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Chat with the model.

        Formats messages into a prompt and generates response.
        """
        prompt = self._format_chat(messages)
        yield from self.generate(prompt, config)

    def _format_chat(self, messages: List[Message]) -> str:
        """Format chat messages into a prompt."""
        lines = []

        for msg in messages:
            if msg.role == "system":
                lines.append(f"System: {msg.content}")
            elif msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"Assistant: {msg.content}")

        lines.append("Assistant:")
        return "\n".join(lines)

    def complete(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate complete response (non-streaming).
        """
        tokens = list(self.generate(prompt, config))
        return "".join(tokens)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
