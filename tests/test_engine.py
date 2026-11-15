"""Sampling and the engine, end to end.

Run:  uv run pytest tests/test_engine.py -v      (then the whole suite: uv run pytest -v)

Why some tests run the engine in fp32: bf16 results change slightly with batch size (a 3-row matmul
and a 1-row matmul round differently), so in bf16 "batched output == solo output" can legitimately
fail on a near-tie. Exact-equality tests use fp32; the HF comparison uses bf16 with a top-5 tolerance.
"""

import pytest
import torch

from kiteserve.config import EngineConfig
from kiteserve.engine import LLMEngine
from kiteserve.sampling import SamplingParams, sample

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "The three primary colors are",
    "List of prime numbers: 2, 3, 5,",
    "Water boils at a temperature of",
    "To make a good cup of tea, first",
    "In 1969, Neil Armstrong",
]


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    return [tokenizer(p).input_ids for p in PROMPTS]


@pytest.fixture(scope="module")
def engine_bf16(model_path):
    return LLMEngine(EngineConfig(model_path=model_path, kv_memory_gb=2.0))


@pytest.fixture(scope="module")
def engine_fp32(model_path):
    return LLMEngine(EngineConfig(model_path=model_path, kv_memory_gb=2.0, dtype=torch.float32))


# --- sample() ---------------------------------------------------------------------------------


def test_sample_greedy_is_argmax():
    logits = torch.randn(6, 50, device="cuda")
    assert torch.equal(sample(logits, torch.zeros(6, device="cuda")), logits.argmax(-1))


def test_sample_temperature_follows_softmax():
    """T=2 on logits [0, 1, 2] -> softmax([0, .5, 1]) = [.186, .307, .506]. Greedy rows stay greedy."""
    torch.manual_seed(0)
    n = 20_000
    logits = torch.tensor([[0.0, 1.0, 2.0]], device="cuda").repeat(n, 1)
    temps = torch.full((n,), 2.0, device="cuda")
    temps[::2] = 0.0  # every other row is greedy
    out = sample(logits, temps)

    assert out.dtype == torch.long and out.shape == (n,)
    assert (out[::2] == 2).all(), "temperature 0 rows must be argmax"
    freq = torch.bincount(out[1::2], minlength=3).float() / (n // 2)
    expected = torch.softmax(torch.tensor([0.0, 0.5, 1.0]), -1).to("cuda")
    assert torch.allclose(freq, expected, atol=0.02), f"sampled frequencies {freq.tolist()} != {expected.tolist()}"


# --- Input validation -------------------------------------------------------------------------------


def test_rejects_too_long(engine_bf16):
    """prompt + max_new_tokens must fit in max_model_len (2048 by default)."""
    with pytest.raises(ValueError):
        engine_bf16.add_request([7] * 2038, SamplingParams(max_new_tokens=11))  # 2049 > 2048
    engine_bf16.add_request([7] * 2038, SamplingParams(max_new_tokens=10))  # exactly 2048: allowed
    while engine_bf16.has_unfinished():  # drain it so later tests start clean
        engine_bf16.step()


def test_rejects_tiny_kv_budget(model_path):
    """A cache that can't hold one max_model_len sequence would preempt forever.

    0.01 GiB holds 5 blocks of 16 tokens (bf16); a 2048-token sequence needs 128.
    The engine checks this before loading the model, so a bad config fails instantly.
    """
    with pytest.raises(ValueError):
        LLMEngine(EngineConfig(model_path=model_path, kv_memory_gb=0.01))


def test_empty_inputs(engine_bf16):
    """An empty prompt is rejected, and generate([]) returns [] without leaving work behind."""
    with pytest.raises(ValueError):
        engine_bf16.add_request([], SamplingParams())
    assert engine_bf16.generate([], SamplingParams()) == []
    assert not engine_bf16.has_unfinished()


# --- step() --------------------------------------------------------------------------------------


def test_step_events(engine_bf16, prompt_ids):
    """step() returns (seq_id, token, finished) for every sequence it ran this step."""
    a = engine_bf16.add_request(prompt_ids[0], SamplingParams(max_new_tokens=3, ignore_eos=True))
    b = engine_bf16.add_request(prompt_ids[1], SamplingParams(max_new_tokens=5, ignore_eos=True))
    assert a != b

    first = engine_bf16.step()
    assert [sid for sid, _, _ in first] == [a], "the first step is a's prefill: one event, for a"

    events = list(first)
    while engine_bf16.has_unfinished():
        events += engine_bf16.step()
    for sid, n in [(a, 3), (b, 5)]:
        mine = [(tok, fin) for s, tok, fin in events if s == sid]
        assert len(mine) == n, f"seq {sid}: expected {n} tokens, got {len(mine)}"
        assert [fin for _, fin in mine] == [False] * (n - 1) + [True], "finished=True exactly once, on the last token"
        assert all(isinstance(tok, int) for tok, _ in mine), "tokens must be Python ints (use .tolist())"


# --- Correctness against HuggingFace ---------------------------------------------------------------


@torch.no_grad()
def test_greedy_matches_hf(engine_bf16, prompt_ids, hf_model_fp32, model_config):
    """8 prompts x 64 greedy tokens in bf16, checked against HF fp32.

    For every generated token, feed HF the same prefix (prompt + our tokens so far) and ask for its
    top-5 next tokens: ours must be among them. Comparing this way (teacher forcing) still works
    after a legitimate near-tie flip, where a plain "same output text" check would fail for the
    rest of the sequence. Top-1 must also agree almost everywhere.
    """
    outs = engine_bf16.generate(prompt_ids, SamplingParams(max_new_tokens=64))
    assert len(outs) == len(prompt_ids)

    agree = total = 0
    eos = set(model_config.eos_token_ids)
    for i, (p, out) in enumerate(zip(prompt_ids, outs)):
        assert 1 <= len(out) <= 64
        assert not eos & set(out[:-1]), f"prompt {i}: kept generating after EOS"
        if len(out) < 64:
            assert out[-1] in eos, f"prompt {i}: stopped after {len(out)} tokens without EOS"

        logits = hf_model_fp32(torch.tensor([p + out], device="cuda")).logits[0]
        pred = logits[len(p) - 1 : len(p) - 1 + len(out)]  # row j predicts out[j]
        top5 = pred.topk(5, dim=-1).indices.cpu()
        ours = torch.tensor(out)
        bad = [j for j in range(len(out)) if ours[j] not in top5[j]]
        assert not bad, f"prompt {i}: tokens at {bad} are not in HF's top-5 (first bad: position {bad[0]})"
        agree += (top5[:, 0] == ours).sum().item()
        total += len(out)
    assert agree / total >= 0.95, f"top-1 agreed with HF on only {agree}/{total} tokens"


def test_batch_equals_solo(engine_fp32, prompt_ids):
    """Each sequence's output in a batch == its output run alone, and outputs come back in input order."""
    params = SamplingParams(max_new_tokens=32)
    batched = engine_fp32.generate(prompt_ids, params)
    for i, p in enumerate(prompt_ids):
        solo = engine_fp32.generate([p], params)[0]
        assert batched[i] == solo, f"prompt {i}: batched output differs from solo"


def test_preemption_still_correct(model_path, engine_fp32, prompt_ids):
    """A cache of 24 blocks (fp32, 16 tokens each) for 8 sequences of up to ~78 tokens forces preemption.

    Preempted sequences are recomputed from prompt + output, so the tokens must not change.
    """
    params = SamplingParams(max_new_tokens=64, ignore_eos=True)
    reference = engine_fp32.generate(prompt_ids, params)

    tiny = LLMEngine(EngineConfig(model_path=model_path, kv_memory_gb=0.085, max_model_len=256, dtype=torch.float32))
    free_before = tiny.scheduler.allocator.num_free
    assert free_before == 24
    outs = tiny.generate(prompt_ids, params)

    assert tiny.scheduler.num_preemptions > 0, "the tiny cache should have forced preemption"
    assert tiny.scheduler.allocator.num_free == free_before, "blocks leaked"
    for i, (got, ref) in enumerate(zip(outs, reference)):
        assert got == ref, f"prompt {i}: output changed after preemption"
