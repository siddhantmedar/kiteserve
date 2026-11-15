"""Shared test fixtures: the model on disk, its tokenizer, and the HF reference model."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiteserve.config import ModelConfig
from kiteserve.loader import download


@pytest.fixture(scope="session")
def model_path() -> str:
    return download()


@pytest.fixture(scope="session")
def model_config(model_path) -> ModelConfig:
    return ModelConfig.from_pretrained(model_path)


@pytest.fixture(scope="session")
def tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="session")
def hf_model_fp32(model_path):
    """The reference: HuggingFace's own Qwen3, fp32, on the GPU."""
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    return model.to("cuda").eval()


# Prompts used by the correctness tests: short, medium, and long with Unicode.
PROMPTS = [
    "The capital of France is",
    (
        "Explain in one paragraph why the sky is blue, and mention Rayleigh scattering "
        "and the wavelength dependence of scattering intensity."
    ),
    ("Here is a story. " * 40) + "Übersetzung: 你好，世界! café naïve résumé 🚀 — the end.",
]
