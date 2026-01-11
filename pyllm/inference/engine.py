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
    # Anti-repetition settings
    no_repeat_ngram_size: int = 3  # Prevent repeating n-grams of this size
    repetition_window: int = 64  # Window to check for repetitions
    early_stop_on_loop: bool = True  # Stop if loop detected
    loop_threshold: int = 3  # Number of repeated sequences before stopping


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
    - Complexity models (Token-Routed MLP)
    - HuggingFace transformers
    - Streaming generation
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
        self._backend = None  # tpu-inl backend

    def _detect_device(self) -> torch.device:
        """
        Detect the best available device.

        Supports: CUDA, CPU only.
        DirectML/MPS/XPU removed - they degrade inference quality.
        """
        device_config = self.config.device

        # Auto-detection: CUDA if available, otherwise CPU
        if device_config == "auto":
            if torch.cuda.is_available():
                logger.info("Using CUDA")
                return torch.device("cuda")
            else:
                logger.info("Using CPU")
                return torch.device("cpu")

        # Manual device selection
        if device_config == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")

        # Default to CPU
        logger.info("Using CPU")
        return torch.device("cpu")

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model."""
        path = model_path or self.config.path

        if not path:
            raise ValueError("Model path required")

        logger.info(f"Loading model from {path}")

        # Determine device (with tpu-inl auto-detection support)
        self.device = self._detect_device()

        logger.info(f"Using device: {self.device}")

        # Try loading as INL-LLM/Complexity first
        if self._try_load_inl_llm(path):
            logger.info("Loaded as INL-LLM/Complexity model")
        elif self._try_load_transformers(path):
            logger.info("Loaded as Transformers model")
        else:
            raise RuntimeError(f"Could not load model from {path}")

        self._loaded = True
        logger.info("Model loaded successfully")

    def _try_load_inl_llm(self, path: str) -> bool:
        """Try loading as INL-LLM or Complexity model."""
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
                # Support both INL style (d_model) and HuggingFace style (hidden_size)
                d_model = model_config.get("d_model", model_config.get("hidden_size", 768))
                num_layers = model_config.get("num_layers", model_config.get("num_hidden_layers", 12))
                num_heads = model_config.get("num_heads", model_config.get("num_attention_heads", 12))
                num_kv_heads = model_config.get("num_kv_heads", model_config.get("num_key_value_heads", 4))
                feedforward_dim = model_config.get("feedforward_dim", model_config.get("intermediate_size", 3072))
                num_iterations = model_config.get("num_iterations_per_layer", 2)

            # Get max_seq_len from model config (override pyllm default if specified)
            max_seq_len = model_config.get("max_seq_len", model_config.get("max_position_embeddings", self.config.max_seq_len))

            logger.info(f"Model config: vocab={vocab_size}, d_model={d_model}, layers={num_layers}, heads={num_heads}, ff={feedforward_dim}, max_seq={max_seq_len}")

            # Detect model type from config
            model_type = model_config.get("model_type", "")
            is_complexity = (
                model_type == "complexity" or
                model_type == "complexity-model" or
                model_type == "complexity-deep" or
                "ComplexityForCausalLM" in str(model_config.get("architectures", [])) or
                "DeepForCausalLM" in str(model_config.get("architectures", []))
            )

            # Load weights first to detect architecture
            if str(weights_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_path))
            else:
                checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

            # Detect ComplexityDeep architecture from state dict keys
            has_token_routed_mlp = any("mlp.experts" in k for k in state_dict.keys())
            has_qk_norm = any("q_norm" in k or "k_norm" in k for k in state_dict.keys())
            has_dynamics = any("dynamics" in k for k in state_dict.keys())

            if has_token_routed_mlp or has_qk_norm or has_dynamics:
                is_complexity = True
                features = []
                if has_token_routed_mlp:
                    features.append("Token-Routed MLP")
                if has_qk_norm:
                    features.append("QK Norm")
                if has_dynamics:
                    features.append("INL Dynamics")
                logger.info(f"Detected ComplexityDeep architecture from state dict keys ({', '.join(features)})")

            model_loaded = False

            # Determine which architecture to use
            is_deep = (
                model_type == "complexity-deep" or
                has_dynamics or
                "DeepForCausalLM" in str(model_config.get("architectures", []))
            )

            # Try Complexity Deep model (with INL Dynamics)
            if is_complexity and is_deep and not model_loaded:
                try:
                    from complexity_deep import DeepConfig, DeepForCausalLM

                    # Get Complexity-specific config
                    num_experts = model_config.get("num_experts", 4)
                    use_qk_norm = model_config.get("use_qk_norm", True)
                    use_token_routed_mlp = model_config.get("use_token_routed_mlp", True)

                    config = DeepConfig(
                        vocab_size=vocab_size,
                        hidden_size=d_model,
                        intermediate_size=feedforward_dim,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_heads,
                        num_key_value_heads=num_kv_heads,
                        max_position_embeddings=self.config.max_seq_len,
                        use_token_routed_mlp=use_token_routed_mlp,
                        num_experts=num_experts,
                        use_qk_norm=use_qk_norm,
                    )
                    self.model = DeepForCausalLM(config)
                    model_loaded = True
                    logger.info("Using ComplexityDeep architecture (Token-Routed MLP + INL Dynamics)")
                except ImportError as e:
                    logger.debug(f"ComplexityDeep import failed: {e}")

            # Try Complexity Model (basic, without INL Dynamics)
            if is_complexity and not model_loaded:
                try:
                    from complexity import ComplexityConfig, ComplexityForCausalLM

                    # Get Complexity-specific config
                    num_experts = model_config.get("num_experts", 4)
                    use_qk_norm = model_config.get("use_qk_norm", True)
                    use_token_routed_mlp = model_config.get("use_token_routed_mlp", True)

                    config = ComplexityConfig(
                        vocab_size=vocab_size,
                        hidden_size=d_model,
                        intermediate_size=feedforward_dim,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_heads,
                        num_key_value_heads=num_kv_heads,
                        max_position_embeddings=self.config.max_seq_len,
                        use_token_routed_mlp=use_token_routed_mlp,
                        num_experts=num_experts,
                        use_qk_norm=use_qk_norm,
                    )
                    self.model = ComplexityForCausalLM(config)
                    model_loaded = True
                    logger.info("Using Complexity architecture (Token-Routed MLP)")
                except ImportError as e:
                    logger.debug(f"Complexity import failed: {e}")

            if not model_loaded:
                logger.error("No compatible model module found. Install with: pip install complexity or pip install complexity-deep")
                return False

            # Load weights
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Try to compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile (reduce-overhead mode)")
                except Exception as e:
                    logger.debug(f"torch.compile not available: {e}")

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
            logger.debug(f"INL-LLM/Complexity load failed: {e}")
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

    def _get_ngrams(self, token_ids: List[int], n: int) -> set:
        """Extract all n-grams from a list of token ids."""
        ngrams = set()
        for i in range(len(token_ids) - n + 1):
            ngram = tuple(token_ids[i:i + n])
            ngrams.add(ngram)
        return ngrams

    def _detect_loop(self, token_ids: List[int], min_pattern: int = 4, max_pattern: int = 20) -> bool:
        """Detect if the last tokens form a repeating loop."""
        if len(token_ids) < min_pattern * 2:
            return False

        # Check for patterns of different sizes
        for pattern_len in range(min_pattern, min(max_pattern, len(token_ids) // 2) + 1):
            pattern = token_ids[-pattern_len:]
            prev_pattern = token_ids[-pattern_len * 2:-pattern_len]
            if pattern == prev_pattern:
                return True
        return False

    def _supports_kv_cache(self) -> bool:
        """Check if the loaded model supports KV caching."""
        if self.model is None:
            return False

        # Check for INL-LLM v3 KV cache support
        # INL-LLM v3 forward signature: forward(input_ids, ..., past_key_values=None, use_cache=False)
        import inspect
        try:
            sig = inspect.signature(self.model.forward)
            has_use_cache = 'use_cache' in sig.parameters
            has_past_kv = 'past_key_values' in sig.parameters
            if has_use_cache and has_past_kv:
                logger.info("KV cache supported (INL-LLM v3)")
                return True
        except Exception:
            pass

        # Check for HuggingFace transformers style
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
            logger.info("KV cache supported (HuggingFace style)")
            return True

        logger.debug("KV cache not supported for this model")
        return False

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> Generator[str, None, None]:
        """Generate tokens one at a time with advanced anti-repetition and KV caching."""
        generated = input_ids.clone()
        generated_list = input_ids[0].tolist()
        past_tokens = set(generated_list)
        loop_count = 0

        # KV Cache for INL-LLM (huge speedup!)
        past_key_values = None
        use_kv_cache = self._supports_kv_cache()

        for step in range(config.max_new_tokens):
            # Forward pass with KV cache optimization
            with torch.no_grad():
                if use_kv_cache and step > 0:
                    # Only process last token with cached KV
                    model_input = generated[:, -1:]
                    outputs = self.model(model_input, past_key_values=past_key_values, use_cache=True)
                else:
                    # First step: process full sequence
                    if use_kv_cache:
                        outputs = self.model(generated, use_cache=True)
                    else:
                        outputs = self.model(generated)

                # Get logits for last token and update cache
                # INL-LLM returns (logits, aux_info, past_key_values) when use_cache=True
                # Complexity returns CausalLMOutput with .logits
                if isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                    # Update KV cache if returned (INL-LLM v3 returns it as 3rd element)
                    if use_kv_cache and len(outputs) >= 3 and outputs[2] is not None:
                        past_key_values = outputs[2]
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits[:, -1, :]
                    if use_kv_cache and hasattr(outputs, "past_key_values"):
                        past_key_values = outputs.past_key_values
                else:
                    logits = outputs[:, -1, :]

            # Apply repetition penalty to past tokens
            if config.repetition_penalty != 1.0:
                # Use window for recent tokens (stronger penalty)
                window_start = max(0, len(generated_list) - config.repetition_window)
                recent_tokens = set(generated_list[window_start:])

                for token_id in recent_tokens:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= config.repetition_penalty
                    else:
                        logits[0, token_id] /= config.repetition_penalty

            # Ban n-grams that would create repetition
            if config.no_repeat_ngram_size > 0 and len(generated_list) >= config.no_repeat_ngram_size:
                # Get recent context for n-gram checking
                window_start = max(0, len(generated_list) - config.repetition_window)
                context = generated_list[window_start:]

                # Get existing n-grams in context
                existing_ngrams = self._get_ngrams(context, config.no_repeat_ngram_size)

                # Check what n-gram would be formed with each possible next token
                prefix = tuple(context[-(config.no_repeat_ngram_size - 1):])
                for ngram in existing_ngrams:
                    if ngram[:-1] == prefix:
                        # This token would create a repeated n-gram
                        banned_token = ngram[-1]
                        logits[0, banned_token] = float('-inf')

            # Temperature = 0 means greedy decoding
            use_greedy = config.temperature <= 0 or not config.do_sample

            if not use_greedy:
                # Apply temperature
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

                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (temperature=0)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            # Update state
            generated = torch.cat([generated, next_token], dim=-1)
            generated_list.append(next_token_id)
            past_tokens.add(next_token_id)

            # Check for loops (early stopping)
            if config.early_stop_on_loop and len(generated_list) > 20:
                if self._detect_loop(generated_list):
                    loop_count += 1
                    if loop_count >= config.loop_threshold:
                        logger.warning(f"Loop detected at step {step}, stopping generation")
                        break
                else:
                    loop_count = 0

            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text

            # Truncate if needed
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]
                generated_list = generated_list[-self.config.max_seq_len:]

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
        """
        Format chat messages into a prompt.

        For base models (not instruction-tuned), we use completion mode:
        concatenate all messages as plain text without special formatting.
        The model continues the text naturally (code or text).
        """
        # Completion mode: concatenate all messages as plain text
        # No "User:" or "Assistant:" prefixes - just natural text continuation
        parts = []
        for msg in messages:
            if msg.content.strip():
                parts.append(msg.content)

        return "\n\n".join(parts)

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
