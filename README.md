<p align="center">
  <img src="assets/kiteserve-mark.png" alt="kiteserve logo" width="120">
</p>

<h1 align="center">kiteserve</h1>

A small LLM inference engine in PyTorch for **Qwen3-0.6B**, with a **paged KV cache**, **continuous batching** and **recompute preemption**. Its outputs are checked against HuggingFace Transformers.

It's built to show how a serving engine works, end to end, in a small amount of readable code. It isn't meant to compete with vLLM.

## Features

- **Paged KV cache:** one pre-allocated K/V pool split into 16-token blocks, a block table per sequence, and a reference-counted block allocator.
- **Continuous batching:** waiting requests are admitted first-come-first-served as soon as a slot (up to 64 running) and enough KV blocks are free, without waiting for the current batch to finish. A static-batching mode is included for comparison.
- **Two KV allocation policies:** allocate blocks on demand (default), or reserve `max_model_len` per request up front.
- **Recompute preemption:** when the cache runs out of blocks, the most recently admitted sequence is evicted and later re-prefilled from its prompt plus the tokens it had generated.
- **Paged decode attention:** each layer writes the new token's K/V into its slot, gathers each sequence's history through its block table, and runs attention with an explicit length mask and grouped-query attention (16 query heads, 8 KV heads).
- **Per-request sampling:** greedy and temperature-sampled requests in the same batch.

## How it works

Each call to `LLMEngine.step()` does one unit of work:

```
LLMEngine.step()
  │
  ├─ Scheduler.schedule()        either PREFILL one waiting request, or DECODE one token for every running request
  │                              (out of blocks → preempt the newest request and recompute it later)
  │
  ├─ build input_ids, positions, AttnMeta     AttnMeta = where new K/V goes + each sequence's block table and length
  │
  ├─ model.forward(...) → logits
  │      each layer writes the new K/V into the cache, then:
  │        prefill: causal attention over the request's own tokens
  │        decode:  gather history through the block tables → length-masked attention
  │
  ├─ sample(logits, temperatures)
  └─ Scheduler.update(...)       append tokens, finish requests, free their blocks
```

| File | What it does |
|---|---|
| `model.py` | Qwen3 forward pass over a flat batch of tokens: `forward(input_ids[N], positions[N], attn_meta, kv_cache) → logits[N, vocab]`. Knows nothing about requests. |
| `kv_cache.py` | The K/V pool `[layers, num_blocks, block_size, kv_heads, head_dim]`, the block allocator, and write/gather by slot and block table. |
| `scheduler.py` | Requests, waiting and running queues, block accounting, batching mode, KV policy and preemption. Pure Python, no GPU code. |
| `sampling.py` | Greedy and temperature sampling, per row. |
| `engine.py` | Wires it together: `add_request`, `step`, `generate`. |
| `config.py`, `loader.py` | Model and engine settings; downloading and loading the HuggingFace weights. |

## Correctness

`uv run pytest -v` runs 62 tests (needs a CUDA GPU; the first run downloads Qwen3-0.6B).

- **Forward pass vs HuggingFace (fp32):** on the test prompts, all 28 layer outputs match within a 1e-3 absolute + relative tolerance, and the final logits within 1e-3.
- **Cache vs recompute (fp32):** prefill and decode through the paged cache, for one sequence and for a batch with different lengths, match recomputing the whole prefix. The cache is filled with random values first, to help catch reads from the wrong slot.
- **Generation vs HuggingFace:** on 8 short prompts, every greedy bf16 token is in HF's top-5 (teacher-forced), and at least 95% match HF's top-1.
- **Batching and preemption don't change outputs (fp32):** batched generation equals running each request alone, and a cache small enough to force preemption gives the same tokens with no leaked blocks.
- **Scheduler invariants:** randomized 1,000-request workloads (scheduler only, 3 configurations) leave no leaked or double-owned blocks, and every request finishes.

The numerical checks use short contexts (prompts up to 223 tokens).

## Quick start

```bash
uv sync
uv run pytest -v
```

```python
from transformers import AutoTokenizer

from kiteserve.config import EngineConfig
from kiteserve.engine import LLMEngine
from kiteserve.loader import download
from kiteserve.sampling import SamplingParams

path = download()  # Qwen/Qwen3-0.6B from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(path)
engine = LLMEngine(EngineConfig(model_path=path))

prompts = [tokenizer("The capital of France is").input_ids, tokenizer("1, 2, 3,").input_ids]
outputs = engine.generate(prompts, SamplingParams(max_new_tokens=32))
print([tokenizer.decode(o) for o in outputs])
```

## Limitations and next steps

1. **Decode attention gathers a padded copy of the KV cache** instead of reading blocks in place. Next step: a paged attention kernel (Triton, or FlashInfer / FlashAttention with a block table).
2. **One prefill per step, no chunked prefill.** While a new request is prefilled, running requests pause for that step.
3. **No CUDA graphs.** At small batch sizes, per-step overhead (most likely kernel launches) dominates.
4. **No prefix caching.** The reference-counted allocator is the groundwork for sharing blocks between requests.
5. **Offline only:** no HTTP server or network streaming endpoint. `step()` returns per-token `(seq_id, token, finished)` events, and `generate()` drives the loop.
6. Greedy and temperature sampling only (no top-k/top-p), on a single GPU.
