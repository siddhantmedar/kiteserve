"""Setup: the environment works, the config is read correctly, and weights load."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from kiteserve.config import ModelConfig
from kiteserve.loader import load_weights


def test_cuda_available():
    assert torch.cuda.is_available()


def test_model_config_matches_qwen3_0_6b(model_config: ModelConfig):
    assert model_config.num_layers == 28
    assert model_config.num_heads == 16
    assert model_config.num_kv_heads == 8
    assert model_config.head_dim == 128
    assert model_config.hidden_size == 1024
    assert model_config.tie_word_embeddings is True
    assert set(model_config.eos_token_ids) == {151645, 151643}


def test_kv_bytes_per_token(model_config: ModelConfig):
    # 2 (K and V) x 28 layers x 8 kv heads x 128 dims x 2 bytes = 112 KB
    assert model_config.kv_bytes_per_token() == 2 * 28 * 8 * 128 * 2 == 114_688


def test_load_weights_fills_an_hf_skeleton(model_path, hf_model_fp32):
    """Load into a randomly initialised model with HF's names; it must equal the real one."""
    skeleton = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path), dtype=torch.float32)
    load_weights(skeleton, model_path)
    ref = dict(hf_model_fp32.named_parameters())
    for name, p in skeleton.named_parameters():
        assert torch.equal(p, ref[name].cpu()), name


def test_load_weights_rejects_wrong_names(model_path):
    with pytest.raises(ValueError, match="don't match"):
        load_weights(torch.nn.Linear(4, 4), model_path)
