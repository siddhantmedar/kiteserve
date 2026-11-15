"""Download Qwen3 weights and copy them into a model.

The model's parameter names must mirror HuggingFace's, e.g.
``model.layers.0.self_attn.q_proj.weight``. Then loading is just matching
names and copying tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def download(model_id: str = DEFAULT_MODEL) -> str:
    """Return a local directory with config, tokenizer and safetensors (downloads once)."""
    return snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )


@torch.no_grad()
def load_weights(model: nn.Module, path: str | Path) -> None:
    """Copy every safetensors tensor in ``path`` into the same-named parameter of ``model``.

    Copying (rather than replacing) keeps each parameter's dtype and device,
    so a bf16 checkpoint can fill an fp32 model for exact-match tests.
    Raises ValueError listing any names that don't line up.
    """
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    weights: dict[str, torch.Tensor] = {}
    for f in files:
        weights.update(load_file(f))

    params = dict(model.named_parameters(remove_duplicate=False))

    # Tied embeddings: some checkpoints omit lm_head and expect it to reuse embed_tokens.
    if "lm_head.weight" in params and "lm_head.weight" not in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    missing = sorted(set(params) - set(weights))
    unexpected = sorted(set(weights) - set(params))
    if missing or unexpected:
        raise ValueError(
            "weight names don't match the model.\n"
            f"  in model but not checkpoint: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"  in checkpoint but not model: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
        )

    for name, param in params.items():
        src = weights[name]
        if param.shape != src.shape:
            raise ValueError(f"{name}: model shape {tuple(param.shape)} != checkpoint {tuple(src.shape)}")
        param.copy_(src)
