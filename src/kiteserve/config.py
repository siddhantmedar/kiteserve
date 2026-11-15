"""Configuration objects.

ModelConfig  -- the model's shape, read from HuggingFace's config.json.
EngineConfig -- how the serving engine runs (cache size, batching mode, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int  # query heads
    num_kv_heads: int  # key/value heads (GQA: fewer than query heads)
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ModelConfig:
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())

        # Generation config lists every token that ends a reply
        # (Qwen3: <|im_end|> and <|endoftext|>); config.json has only one.
        eos = cfg["eos_token_id"]
        gen_path = path / "generation_config.json"
        if gen_path.exists():
            eos = json.loads(gen_path.read_text()).get("eos_token_id", eos)
        eos_ids = tuple(eos) if isinstance(eos, list) else (eos,)

        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=float(cfg["rope_theta"]),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            eos_token_ids=eos_ids,
        )

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """K and V, for every layer, for one token. Uses KV heads, not query heads."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes


@dataclass
class EngineConfig:
    model_path: str
    block_size: int = 16
    max_num_seqs: int = 64
    max_model_len: int = 2048  # max tokens per sequence: prompt + generated output
    kv_memory_gb: float = 8.0
    kv_policy: Literal["on_demand", "reserve_max"] = "on_demand"
    batching: Literal["continuous", "static"] = "continuous"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
