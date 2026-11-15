"""The Qwen3 forward pass (no KV cache) must match HuggingFace.

Run:  uv run pytest tests/test_model.py -v

How to read failures: test_layer_outputs_match_hf checks layer 0 first and
stops at the first layer that disagrees. If layer 0 fails, the bug is in
embeddings / RMSNorm / attention / MLP of a single layer -- not deep in the stack.
"""

import pytest
import torch

from kiteserve.loader import load_weights
from kiteserve.model import Qwen3ForCausalLM

from .conftest import PROMPTS


def _ids(tokenizer, prompt: str) -> torch.Tensor:
    return torch.tensor(tokenizer(prompt).input_ids, device="cuda")


def _build(model_config, model_path, dtype) -> Qwen3ForCausalLM:
    model = Qwen3ForCausalLM(model_config).to(device="cuda", dtype=dtype)
    load_weights(model, model_path)
    return model.eval()


@pytest.fixture(scope="module")
def ours_fp32(model_config, model_path):
    return _build(model_config, model_path, torch.float32)


@pytest.fixture(scope="module")
def ours_bf16(model_config, model_path):
    return _build(model_config, model_path, torch.bfloat16)


# --- Unit tests: check one piece at a time, before the whole model exists -------------
# Run just these with:  uv run pytest tests/test_model.py -k "rmsnorm or attention or mlp" -v


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_hf(dtype, hf_model_fp32):
    from kiteserve.model import RMSNorm

    hf_norm = hf_model_fp32.model.layers[0].input_layernorm
    ours = RMSNorm(hf_norm.weight.shape[0], hf_norm.variance_epsilon).cuda()
    # strict=True also checks that `weight` is a real nn.Parameter (the loader needs that)
    ours.load_state_dict(hf_norm.state_dict(), strict=True)
    ours, hf_norm = ours.to(dtype), _copy_to(hf_norm, dtype)

    x = torch.randn(7, 3, hf_norm.weight.shape[0], device="cuda", dtype=dtype) * 5
    got = ours(x)
    assert got.dtype == dtype, f"RMSNorm must return the input's dtype ({dtype}), got {got.dtype}"
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(got, hf_norm(x), atol=tol, rtol=tol)


@torch.no_grad()
@pytest.mark.parametrize("prompt", PROMPTS[:2], ids=["short", "medium"])
def test_attention_layer0_matches_hf(prompt, tokenizer, hf_model_fp32, model_config):
    """Feed HF's layer-0 attention its real input; our Qwen3Attention must give the same output."""
    from kiteserve.model import Qwen3Attention

    hf_attn = hf_model_fp32.model.layers[0].self_attn
    captured = {}

    def hook(_m, args, kwargs, out):
        captured["x"] = (args[0] if args else kwargs["hidden_states"])[0]  # [N, hidden]
        captured["out"] = (out[0] if isinstance(out, tuple) else out)[0]

    handle = hf_attn.register_forward_hook(hook, with_kwargs=True)
    ids = _ids(tokenizer, prompt)
    hf_model_fp32(ids[None])
    handle.remove()

    ours = Qwen3Attention(model_config, layer_idx=0).cuda().float()
    ours.load_state_dict(hf_attn.state_dict(), strict=True)
    got = ours(captured["x"], torch.arange(len(ids), device="cuda"))

    assert got.shape == captured["out"].shape, f"shape {tuple(got.shape)} != HF {tuple(captured['out'].shape)}"
    torch.testing.assert_close(got, captured["out"], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_mlp_layer0_matches_hf(hf_model_fp32, model_config):
    from kiteserve.model import Qwen3MLP

    hf_mlp = hf_model_fp32.model.layers[0].mlp
    ours = Qwen3MLP(model_config).cuda().float()
    # strict=True fails on extra biases (HF has none) or different parameter names
    ours.load_state_dict(hf_mlp.state_dict(), strict=True)
    x = torch.randn(9, model_config.hidden_size, device="cuda")
    torch.testing.assert_close(ours(x), hf_mlp(x), atol=1e-4, rtol=1e-4)


def _copy_to(module, dtype):
    import copy

    return copy.deepcopy(module).to(dtype)


# --- Whole-model tests ----------------------------------------------------------------


def _record_layer_outputs(layers) -> tuple[list[torch.Tensor], list]:
    """Forward hooks that save each decoder layer's output hidden states."""
    outputs, handles = [], []

    def hook(_module, _inputs, out):
        out = out[0] if isinstance(out, tuple) else out
        outputs.append(out.detach().reshape(-1, out.shape[-1]))  # -> [N, hidden]

    for layer in layers:
        handles.append(layer.register_forward_hook(hook))
    return outputs, handles


@torch.no_grad()
@pytest.mark.parametrize("prompt", PROMPTS, ids=["short", "medium", "long_unicode"])
def test_layer_outputs_match_hf(prompt, tokenizer, hf_model_fp32, ours_fp32, model_config):
    ids = _ids(tokenizer, prompt)
    positions = torch.arange(len(ids), device="cuda")

    ref, h1 = _record_layer_outputs(hf_model_fp32.model.layers)
    hf_model_fp32(ids[None])
    got, h2 = _record_layer_outputs(ours_fp32.model.layers)
    ours_fp32(ids, positions)
    for h in h1 + h2:
        h.remove()

    assert len(got) == model_config.num_layers, "expected one output per decoder layer"
    for i, (g, r) in enumerate(zip(got, ref)):
        assert g.shape == r.shape, f"layer {i}: shape {tuple(g.shape)} != HF {tuple(r.shape)}"
        # 1e-3: fp32 rounding differences accumulate with depth (different but equivalent op order
        # gives ~1e-4 by layer 13 on a 223-token prompt). Deep-layer hidden values reach the thousands,
        # so the rtol term matters: the largest absolute gap is ~2e-3. Real bugs show O(0.1-10).
        torch.testing.assert_close(g, r, atol=1e-3, rtol=1e-3, msg=lambda m, i=i: f"layer {i} differs from HF:\n{m}")


@torch.no_grad()
@pytest.mark.parametrize("prompt", PROMPTS, ids=["short", "medium", "long_unicode"])
def test_logits_match_hf(prompt, tokenizer, hf_model_fp32, ours_fp32, model_config):
    ids = _ids(tokenizer, prompt)
    logits = ours_fp32(ids, torch.arange(len(ids), device="cuda"))
    ref = hf_model_fp32(ids[None]).logits[0]

    assert logits.shape == (len(ids), model_config.vocab_size), "forward must return [N, vocab] logits"
    max_diff = (logits - ref).abs().max().item()
    assert max_diff < 1e-3, f"max |logit difference| = {max_diff:.2e}"


@torch.no_grad()
def test_bf16_top_token_agrees(tokenizer, hf_model_fp32, ours_bf16):
    """bf16 has ~3 significant digits, so logits drift; the *top* token should still agree.

    Near-ties (two tokens with almost equal logits) can legitimately flip in bf16 --
    a straightforward correct implementation agrees at ~99.6% here -- so the bar is 97%.
    A real bug (wrong RoPE, missing norm) drops this far lower.
    """
    agree = total = 0
    for prompt in PROMPTS:
        ids = _ids(tokenizer, prompt)
        ours = ours_bf16(ids, torch.arange(len(ids), device="cuda")).argmax(-1)
        ref = hf_model_fp32(ids[None]).logits[0].argmax(-1)
        agree += (ours == ref).sum().item()
        total += len(ids)
    assert agree / total >= 0.97, f"top token agreed at only {agree}/{total} positions"
