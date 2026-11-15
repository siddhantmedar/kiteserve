"""Attention that reads and writes the paged KV cache.

Run:  uv run pytest tests/test_decode.py -v      (and re-run tests/test_model.py: the no-cache path must keep working)

The idea of every test: run the model the SERVING way (prefill once, then one token per step,
reading the history from the KV cache) and compare with the trusted path (recompute the
whole prefix with no cache, attn_meta=None). The logits must agree.

The cache is filled with random garbage before each test. If attention ever reads a slot it
shouldn't (padding, a block that belongs to another sequence, a slot not written yet), the
logits come out wrong, instead of accidentally matching because the slot happened to be zero.

"""

import random

import pytest
import torch

from kiteserve.kv_cache import AttnMeta, KVCache, slot_for
from kiteserve.loader import load_weights
from kiteserve.model import Qwen3ForCausalLM

from .conftest import PROMPTS

BS = 16
NUM_BLOCKS = 32
ATOL = 1e-3  # fp32; cached vs recomputed attention adds up the same numbers in a different order


def _t(values) -> torch.Tensor:
    return torch.tensor(values, device="cuda")


@pytest.fixture(scope="module")
def model(model_config, model_path):
    m = Qwen3ForCausalLM(model_config).to(device="cuda", dtype=torch.float32)
    load_weights(m, model_path)
    return m.eval()


@pytest.fixture
def cache(model_config):
    c = KVCache(model_config, num_blocks=NUM_BLOCKS, block_size=BS, dtype=torch.float32)
    torch.manual_seed(0)
    c.k.normal_()  # garbage everywhere (see module docstring)
    c.v.normal_()
    return c


@pytest.fixture
def pool():
    """Free block ids in shuffled order, so a sequence's blocks are never neatly 0, 1, 2, ..."""
    blocks = list(range(NUM_BLOCKS))
    random.Random(0).shuffle(blocks)
    return blocks


@pytest.fixture(scope="module")
def token_ids(tokenizer):
    ids = tokenizer(PROMPTS[2]).input_ids  # the long prompt, used as a source of real tokens
    assert len(ids) >= 140
    return ids


class Seq:
    """A tiny stand-in for the scheduler's Sequence: its tokens plus its block table."""

    def __init__(self, ids, pool):
        self.ids = list(ids)
        self.table: list[int] = []
        self.pool = pool

    def ensure_blocks(self, num_tokens):
        while len(self.table) * BS < num_tokens:
            self.table.append(self.pool.pop())


# --- How the engine calls the model ---------------------------------------------------------------


@torch.no_grad()
def no_cache_logits(model, ids) -> torch.Tensor:
    """The no-cache path: attn_meta=None, recompute everything. Returns [len(ids), vocab]."""
    return model(_t(ids), torch.arange(len(ids), device="cuda"))


@torch.no_grad()
def prefill(model, cache, seq) -> torch.Tensor:
    """One sequence, all its tokens at once. Every token's K/V gets written to the cache."""
    n = len(seq.ids)
    seq.ensure_blocks(n)
    meta = AttnMeta(is_prefill=True, slot_mapping=_t([slot_for(seq.table, p, BS) for p in range(n)]))
    return model(_t(seq.ids), torch.arange(n, device="cuda"), attn_meta=meta, kv_cache=cache)


@torch.no_grad()
def decode(model, cache, seqs, new_tokens) -> torch.Tensor:
    """One decode step: each sequence b gets ONE new token, new_tokens[b]. Returns [B, vocab].

    The new token's K/V is written at its slot; attention reads the whole history from the cache.
    """
    for s, tok in zip(seqs, new_tokens):
        s.ids.append(tok)
        s.ensure_blocks(len(s.ids))
    ctx = [len(s.ids) for s in seqs]  # context_lens INCLUDE the new token
    m = max(len(s.table) for s in seqs)
    meta = AttnMeta(
        is_prefill=False,
        slot_mapping=_t([slot_for(s.table, c - 1, BS) for s, c in zip(seqs, ctx)]),
        block_tables=_t([s.table + [0] * (m - len(s.table)) for s in seqs]),  # pad with block 0
        context_lens=_t(ctx),
    )
    positions = _t([c - 1 for c in ctx])  # the new token's position in its own sequence
    return model(_t(new_tokens), positions, attn_meta=meta, kv_cache=cache)


# --- Tests ----------------------------------------------------------------------------------------


def test_prefill_with_cache_matches_no_cache(model, cache, pool, token_ids, model_config):
    seq = Seq(token_ids[:40], pool)
    k_before, v_before = cache.k.clone(), cache.v.clone()

    got = prefill(model, cache, seq)
    torch.testing.assert_close(got, no_cache_logits(model, seq.ids), atol=1e-4, rtol=1e-4)

    # every token's K and V must now be in the cache, in every layer, and nothing else touched
    flat = (model_config.num_layers, NUM_BLOCKS * BS, model_config.num_kv_heads, model_config.head_dim)
    written = torch.zeros(NUM_BLOCKS * BS, dtype=torch.bool, device="cuda")
    written[[slot_for(seq.table, p, BS) for p in range(len(seq.ids))]] = True
    for name, now, before in [("K", cache.k, k_before), ("V", cache.v, v_before)]:
        changed = (now.view(flat) != before.view(flat)).flatten(2).any(-1)  # [layers, slots]
        assert changed[:, written].all(), f"prefill didn't write {name} for every token in every layer"
        assert not changed[:, ~written].any(), f"prefill wrote {name} into slots it doesn't own"


def test_decode_matches_recompute(model, cache, pool, token_ids):
    """Prefill 28 tokens, then feed the next 10 one at a time (context 29 .. 38).

    Context 33 is the first token of a new block (block size 16), so a block boundary is crossed.
    """
    seq = Seq(token_ids[:28], pool)
    prefill(model, cache, seq)
    for step, tok in enumerate(token_ids[28:38]):
        got = decode(model, cache, [seq], [tok])
        assert got.shape == (1, model.cfg.vocab_size), f"decode must return [B, vocab], got {tuple(got.shape)}"
        ref = no_cache_logits(model, seq.ids)[-1]
        torch.testing.assert_close(
            got[0], ref, atol=ATOL, rtol=ATOL,
            msg=lambda m, step=step: f"decode step {step} (context {len(seq.ids)}) differs from recompute:\n{m}",
        )


def test_batched_decode_matches_solo(model, cache, pool, token_ids):
    """Three sequences of lengths 5, 16 and 33 decoded TOGETHER, 4 steps.

    Each row must equal that sequence computed on its own. Their block tables have different
    lengths (padded with block 0, which holds someone else's data or garbage), so this checks the
    length mask, and that row b uses sequence b's history and nobody else's.
    """
    starts, lengths = [0, 50, 100], [5, 16, 33]
    seqs = [Seq(token_ids[s : s + n], pool) for s, n in zip(starts, lengths)]
    for s in seqs:
        prefill(model, cache, s)

    for step in range(4):
        new = [token_ids[s + n + step] for s, n in zip(starts, lengths)]
        got = decode(model, cache, seqs, new)
        assert got.shape == (3, model.cfg.vocab_size)
        for b, s in enumerate(seqs):
            ref = no_cache_logits(model, s.ids)[-1]
            torch.testing.assert_close(
                got[b], ref, atol=ATOL, rtol=ATOL,
                msg=lambda m, step=step, b=b, s=s: f"step {step}, sequence {b} (context {len(s.ids)}) differs:\n{m}",
            )
