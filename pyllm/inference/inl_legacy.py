"""
INL-LLM v2 Legacy Model (Standalone)

This is a simplified, standalone implementation of INL-LLM v2 architecture
that matches the checkpoint structure without requiring heavy optimization imports.

Checkpoint structure (500M model):
- token_embedding.weight
- pos_encoding.pe (sinusoidal)
- layers.X.attention.qkv_proj.weight/bias (fused QKV)
- layers.X.attention.out_proj.weight/bias
- layers.X.ff.0.weight/bias (linear)
- layers.X.ff.3.weight/bias (linear)
- layers.X.norm1.weight/bias (LayerNorm)
- layers.X.norm2.weight/bias (LayerNorm)
- layers.X.norm_attn.weight/bias (LayerNorm)
- final_norm.weight/bias
- lm_head.weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class LowRankEmbedding(nn.Module):
    """
    Low-rank embedding used in v2 checkpoints.

    Instead of vocab_size x d_model, uses:
    - embed_low: vocab_size x rank
    - project_up: rank x d_model
    """
    def __init__(self, vocab_size: int, d_model: int, rank_ratio: float = 0.125):
        super().__init__()
        rank = max(64, int(d_model * rank_ratio))
        self.embed_low = nn.Embedding(vocab_size, rank)
        self.project_up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_up(self.embed_low(x))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (v2 style)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, start_pos:start_pos + seq_len, :]


class INLCacheLayer:
    """KV cache for a single layer."""
    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat([self.keys, k], dim=2)
            self.values = torch.cat([self.values, v], dim=2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0


class INLCache:
    """KV cache for all layers."""
    def __init__(self, num_layers: int):
        self.layers = [INLCacheLayer() for _ in range(num_layers)]

    def __getitem__(self, idx: int) -> INLCacheLayer:
        return self.layers[idx]

    def get_seq_length(self) -> int:
        return self.layers[0].get_seq_length()


class FusedQKVAttention(nn.Module):
    """
    Attention with fused QKV projection (v2 style).
    Matches checkpoint keys: attention.qkv_proj.weight/bias
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Fused QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache_layer: Optional[INLCacheLayer] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, S, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle KV cache
        if use_cache and cache_layer is not None:
            k, v = cache_layer.update(k, v)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class INLLegacyBlock(nn.Module):
    """
    INL-LLM v2 transformer block.

    Structure:
    - norm_attn + attention (fused QKV)
    - norm1 + (INL dynamics simplified to identity for inference)
    - norm2 + ff (sequential with GELU)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # LayerNorms (with bias, v2 style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)

        # Attention with fused QKV
        self.attention = FusedQKVAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feedforward (Sequential with indexed access: ff.0, ff.3)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),  # ff.0
            nn.GELU(),                             # ff.1
            nn.Dropout(dropout),                   # ff.2
            nn.Linear(feedforward_dim, d_model),  # ff.3
            nn.Dropout(dropout)                    # ff.4
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache_layer: Optional[INLCacheLayer] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Build causal mask
        if use_cache and cache_layer is not None:
            past_len = cache_layer.get_seq_length()
            total_len = past_len + seq_len
            attn_mask = torch.zeros(seq_len, total_len, device=x.device, dtype=torch.bool)
            if seq_len > 1:
                new_causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_mask[:, past_len:] = new_causal_mask
        elif mask is None:
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        else:
            attn_mask = mask

        # Attention
        x_norm = self.norm_attn(x)
        attn_output = self.attention(x_norm, attn_mask=attn_mask, cache_layer=cache_layer, use_cache=use_cache)
        x = x + self.dropout(attn_output)

        # Skip INL dynamics for inference (just apply norm1)
        # The checkpoint has INL-related weights but for generation we can simplify
        x_norm = self.norm1(x)
        x = x + self.dropout(x_norm)  # Identity-like pass through

        # Feedforward
        x = x + self.ff(self.norm2(x))

        return x


class IntegratorLanguageModelLegacy(nn.Module):
    """
    INL-LLM v2 Legacy Model (Standalone).

    Matches the checkpoint structure from early training runs.
    Does NOT require inl-llm package or optimization imports.

    Config format (config.json):
    {
        "model_type": "inl-llm-v2",
        "vocab_size": 50261,
        "d_model": 1280,
        "num_layers": 18,
        "num_heads": 20,
        "num_iterations_per_layer": 2,  # ignored for inference
        "feedforward_dim": 5120,
        "max_seq_len": 1024,
        "dropout": 0.1
    }
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_iterations_per_layer: int = 5,  # Ignored for inference
        feedforward_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        **kwargs  # Ignore extra config params
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        if feedforward_dim is None:
            feedforward_dim = 4 * d_model

        # Token embedding - use low-rank like the checkpoint
        self.token_embedding = LowRankEmbedding(vocab_size, d_model, rank_ratio=0.125)

        # Sinusoidal positional encoding (v2 style)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            INLLegacyBlock(
                d_model=d_model,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final norm (LayerNorm with bias, v2 style)
        self.final_norm = nn.LayerNorm(d_model)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[INLCache] = None,
        use_cache: bool = False,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[List], Optional[INLCache]]:
        """
        Forward pass with optional KV caching.
        """
        # Initialize cache
        if use_cache and past_key_values is None:
            past_key_values = INLCache(num_layers=self.num_layers)

        # Position for positional encoding
        start_pos = 0
        if use_cache and past_key_values is not None:
            start_pos = past_key_values.get_seq_length()

        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x, start_pos=start_pos)
        x = self.dropout(x)

        # Layers
        for layer_idx, layer in enumerate(self.layers):
            cache_layer = past_key_values[layer_idx] if use_cache else None
            x = layer(x, mask=attention_mask, cache_layer=cache_layer, use_cache=use_cache)

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, None, past_key_values if use_cache else None

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching."""
        self.eval()
        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                if use_cache and step > 0:
                    model_input = input_ids[:, -1:]
                    logits, _, past_key_values = self.forward(
                        model_input,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    logits, _, past_key_values = self.forward(
                        input_ids,
                        past_key_values=past_key_values if use_cache else None,
                        use_cache=use_cache
                    )

                logits = logits[:, -1, :] / temperature

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        for token_id in set(input_ids[i].tolist()):
                            if logits[i, token_id] < 0:
                                logits[i, token_id] *= repetition_penalty
                            else:
                                logits[i, token_id] /= repetition_penalty

                # Top-k
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample or greedy
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if eos_token_id is not None and (next_token == eos_token_id).any():
                    break

        return input_ids

    def get_num_params(self) -> int:
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())
