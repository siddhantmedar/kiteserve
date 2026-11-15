"""The paged KV cache: slot math, the block allocator, sizing, write and gather.

Run:  uv run pytest tests/test_kv_cache.py -v
"""

import random

import pytest
import torch

from kiteserve.kv_cache import BlockAllocator, KVCache, slot_for

BS = 16  # block size used throughout these tests


# --- slot_for ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pos, expected",
    [
        (0, 7 * 16 + 0),  # first token: block 7, offset 0
        (15, 7 * 16 + 15),  # last token of the first block
        (16, 3 * 16 + 0),  # crosses into the second block (block 3)
        (17, 3 * 16 + 1),
        (31, 3 * 16 + 15),  # last token of the second block
        (32, 9 * 16 + 0),  # third block (block 9)
    ],
)
def test_slot_for(pos, expected):
    assert slot_for([7, 3, 9], pos, BS) == expected


# --- BlockAllocator ---------------------------------------------------------------------


def test_allocator_hands_out_every_block_once():
    alloc = BlockAllocator(8)
    assert alloc.num_free == 8
    got = [alloc.allocate() for _ in range(8)]
    assert sorted(got) == list(range(8)), "each block id 0..7 exactly once"
    assert alloc.num_free == 0
    with pytest.raises(RuntimeError):
        alloc.allocate()


def test_allocator_free_makes_block_reusable():
    alloc = BlockAllocator(2)
    a, _ = alloc.allocate(), alloc.allocate()
    alloc.free(a)
    assert alloc.num_free == 1
    assert alloc.ref_count(a) == 0
    assert alloc.allocate() == a, "the only free block must be the one just freed"


def test_allocator_double_free_raises():
    alloc = BlockAllocator(4)
    b = alloc.allocate()
    alloc.free(b)
    with pytest.raises(ValueError):
        alloc.free(b)
    with pytest.raises(ValueError):
        alloc.free(3)  # never allocated


def test_allocator_refcount():
    alloc = BlockAllocator(4)
    b = alloc.allocate()
    assert alloc.ref_count(b) == 1
    alloc.incref(b)
    assert alloc.ref_count(b) == 2
    alloc.free(b)
    assert alloc.ref_count(b) == 1 and alloc.num_free == 3, "still used by one owner"
    alloc.free(b)
    assert alloc.ref_count(b) == 0 and alloc.num_free == 4
    with pytest.raises(ValueError):
        alloc.incref(b)  # can't share a block nobody owns


# --- KVCache: sizing and shapes -----------------------------------------------------------


def test_num_blocks_for(model_config):
    # one block = 16 tokens x 112 KiB (K+V, 28 layers, 8 kv heads, 128 dims, 2 bytes) = 1,835,008 bytes
    # 1 GiB / 1,835,008 = 585.1  -> 585 whole blocks
    assert KVCache.num_blocks_for(model_config, BS, kv_memory_gb=1.0) == 585
    assert KVCache.num_blocks_for(model_config, BS, kv_memory_gb=1.0, dtype=torch.float32) == 292


def test_cache_shapes(model_config):
    cache = KVCache(model_config, num_blocks=8, block_size=BS, dtype=torch.float32)
    expected = (28, 8, BS, 8, 128)  # [layers, blocks, block_size, kv_heads, head_dim]
    assert tuple(cache.k.shape) == expected and tuple(cache.v.shape) == expected
    assert cache.k.device.type == "cuda" and cache.k.dtype == torch.float32


# --- KVCache: write and gather --------------------------------------------------------------


def _small_cache(model_config, num_blocks=16):
    return KVCache(model_config, num_blocks=num_blocks, block_size=BS, dtype=torch.float32)


def test_write_lands_in_the_right_slot(model_config):
    cache = _small_cache(model_config)
    kvh, hd = model_config.num_kv_heads, model_config.head_dim
    k = torch.randn(3, kvh, hd, device="cuda")
    v = torch.randn(3, kvh, hd, device="cuda")
    slots = torch.tensor([5 * BS + 2, 0, 11 * BS + 15], device="cuda")
    cache.write(layer=4, k=k, v=v, slot_mapping=slots)

    for i, (block, offset) in enumerate([(5, 2), (0, 0), (11, 15)]):
        assert torch.equal(cache.k[4, block, offset], k[i]), f"token {i} K not at block {block} offset {offset}"
        assert torch.equal(cache.v[4, block, offset], v[i]), f"token {i} V not at block {block} offset {offset}"
    assert cache.k[3].abs().sum() == 0 and cache.k[5].abs().sum() == 0, "write must only touch its own layer"


def test_write_gather_roundtrip_shuffled_blocks(model_config):
    """Several sequences, blocks handed out in random order: gather must return each history intact."""
    random.seed(0)
    torch.manual_seed(0)
    cache = _small_cache(model_config, num_blocks=16)
    kvh, hd = model_config.num_kv_heads, model_config.head_dim
    lengths = [1, 15, 16, 17, 40]  # around the block boundaries (BS=16)

    free = list(range(16))
    random.shuffle(free)  # physical blocks deliberately NOT in order
    tables, refs_k, refs_v, all_slots, all_k, all_v = [], [], [], [], [], []
    for n in lengths:
        table = [free.pop() for _ in range((n + BS - 1) // BS)]
        k, v = torch.randn(n, kvh, hd, device="cuda"), torch.randn(n, kvh, hd, device="cuda")
        tables.append(table)
        refs_k.append(k)
        refs_v.append(v)
        all_slots += [slot_for(table, p, BS) for p in range(n)]
        all_k.append(k)
        all_v.append(v)

    # one write call for all sequences at once, like a real batched step
    cache.write(2, torch.cat(all_k), torch.cat(all_v), torch.tensor(all_slots, device="cuda"))

    max_blocks = max(len(t) for t in tables)  # 3 (for the 40-token sequence)
    block_tables = torch.tensor([t + [0] * (max_blocks - len(t)) for t in tables], device="cuda")
    context_lens = torch.tensor(lengths, device="cuda")
    gk, gv = cache.gather(2, block_tables, context_lens)

    assert gk.shape == (len(lengths), max_blocks * BS, kvh, hd), f"got {tuple(gk.shape)}"
    for b, n in enumerate(lengths):
        assert torch.equal(gk[b, :n], refs_k[b]), f"sequence {b} (len {n}): K history wrong"
        assert torch.equal(gv[b, :n], refs_v[b]), f"sequence {b} (len {n}): V history wrong"
