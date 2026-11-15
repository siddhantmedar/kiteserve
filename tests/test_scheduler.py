"""The scheduler (pure bookkeeping -- no GPU, no model).

Run:  uv run pytest tests/test_scheduler.py -v
test_no_leaks_randomized (the big one) runs last.

How these tests drive the scheduler (the engine does the same, plus the actual model):
    result = sched.schedule()                  # what to run this step (or None: nothing to do)
    tokens = <the model's next token for each seq in result.seqs>
    sched.update(result.seqs, tokens, eos_ids)  # append tokens, finish sequences, free blocks
"""

import math
import random

import pytest

from kiteserve.config import EngineConfig
from kiteserve.kv_cache import BlockAllocator
from kiteserve.scheduler import Scheduler, SeqStatus, Sequence

EOS = 999
EOS_IDS = (EOS,)
TOK = 7  # an ordinary (non-EOS) token


def _make(num_blocks, block_size=16, max_num_seqs=64, max_model_len=2048, **kw):
    cfg = EngineConfig(
        model_path="unused", block_size=block_size, max_num_seqs=max_num_seqs, max_model_len=max_model_len, **kw
    )
    alloc = BlockAllocator(num_blocks)
    return Scheduler(cfg, alloc), alloc


def _seq(seq_id, prompt_len, max_new_tokens=100, **kw):
    return Sequence(seq_id=seq_id, prompt_ids=[TOK] * prompt_len, max_new_tokens=max_new_tokens, **kw)


def _step(sched, token=TOK):
    """schedule() + update() with the same token for every sequence. Returns the ScheduleResult."""
    result = sched.schedule()
    assert result is not None, "schedule() returned None but there is work to do"
    sched.update(result.seqs, [token] * len(result.seqs), EOS_IDS)
    return result


def _ids(seqs):
    return [s.seq_id for s in seqs]


# --- Sequence basics ---------------------------------------------------------------------------


def test_sequence_properties():
    s = _seq(0, prompt_len=3)
    assert s.status == SeqStatus.WAITING and s.output_ids == [] and s.block_table == []
    s.output_ids += [5, 6]
    assert s.num_tokens == 5
    assert s.all_ids == [TOK, TOK, TOK, 5, 6]


# --- Prefill vs decode -----------------------------------------------------------------------


def test_prefill_first_one_at_a_time():
    sched, _ = _make(num_blocks=64)
    for i in range(3):
        sched.add(_seq(i, prompt_len=10))

    for i in range(3):  # three prefill steps, one sequence each, in arrival order
        r = _step(sched)
        assert r.is_prefill and _ids(r.seqs) == [i]
        assert r.seqs[0].status == SeqStatus.RUNNING

    r = _step(sched)  # nothing waiting -> one decode step with everyone, oldest first
    assert not r.is_prefill and _ids(r.seqs) == [0, 1, 2]
    assert sched.peak_running == 3
    assert sched.has_unfinished()


def test_nothing_to_do_returns_none():
    sched, _ = _make(num_blocks=8)
    assert sched.schedule() is None
    assert not sched.has_unfinished()


def test_max_num_seqs_caps_running():
    sched, _ = _make(num_blocks=64, max_num_seqs=2)
    for i in range(3):
        sched.add(_seq(i, prompt_len=4))
    assert _ids(_step(sched).seqs) == [0]
    assert _ids(_step(sched).seqs) == [1]
    r = _step(sched)  # seq 2 must wait: 2 already running
    assert not r.is_prefill and _ids(r.seqs) == [0, 1]


# --- Blocks ------------------------------------------------------------------------------------


def test_blocks_grow_on_demand():
    sched, alloc = _make(num_blocks=8, block_size=16)
    s = _seq(0, prompt_len=16)
    sched.add(s)

    _step(sched)  # prefill: 16 tokens in the cache -> exactly 1 block
    assert len(s.block_table) == 1 and alloc.num_free == 7
    # now num_tokens = 17 (the sampled token isn't in the cache yet; the next decode writes it at position 16)

    _step(sched)  # decode: position 16 needs a 2nd block
    assert len(s.block_table) == 2 and s.num_tokens == 18

    for _ in range(14):  # decodes write positions 17 .. 30: still fits in 2 blocks
        _step(sched)
    assert s.num_tokens == 32 and len(s.block_table) == 2

    _step(sched)  # writes position 31 (last slot of block 2)
    assert len(s.block_table) == 2
    _step(sched)  # writes position 32 -> 3rd block
    assert len(s.block_table) == 3


def test_reserve_max_allocates_upfront():
    sched, alloc = _make(num_blocks=8, block_size=16, max_model_len=64, kv_policy="reserve_max")
    s = _seq(0, prompt_len=5)
    sched.add(s)
    _step(sched)
    assert len(s.block_table) == math.ceil(64 / 16) == 4 and alloc.num_free == 4
    for _ in range(20):
        _step(sched)
    assert len(s.block_table) == 4, "reserve_max never allocates after admission"


def test_waits_when_not_enough_blocks():
    sched, _ = _make(num_blocks=3, block_size=4)
    a, b = _seq(0, prompt_len=8), _seq(1, prompt_len=8)  # 2 blocks each
    sched.add(a)
    sched.add(b)
    _step(sched)  # prefill a (2 blocks, 1 left)
    r = _step(sched)  # b needs 2 blocks, only 1 free -> decode a instead
    assert not r.is_prefill and _ids(r.seqs) == [0]
    assert b.status == SeqStatus.WAITING and b.block_table == []


# --- Finishing ---------------------------------------------------------------------------------


def test_finishes_at_max_new_tokens():
    sched, alloc = _make(num_blocks=8)
    s = _seq(0, prompt_len=5, max_new_tokens=3)
    sched.add(s)
    _step(sched)
    _step(sched)
    assert s.status == SeqStatus.RUNNING
    r = sched.schedule()
    finished = sched.update(r.seqs, [TOK], EOS_IDS)
    assert finished == [s] and s.status == SeqStatus.FINISHED
    assert len(s.output_ids) == 3
    assert s.block_table == [] and alloc.num_free == 8, "a finished sequence must give its blocks back"
    assert sched.schedule() is None and not sched.has_unfinished()


def test_eos_first_token_finishes():
    """EOS as the very first generated token."""
    sched, alloc = _make(num_blocks=8)
    s = _seq(0, prompt_len=5, max_new_tokens=10)
    sched.add(s)
    r = sched.schedule()
    assert sched.update(r.seqs, [EOS], EOS_IDS) == [s]
    assert s.output_ids == [EOS] and s.status == SeqStatus.FINISHED
    assert alloc.num_free == 8 and sched.schedule() is None


def test_ignore_eos_keeps_going():
    sched, _ = _make(num_blocks=8)
    s = _seq(0, prompt_len=5, max_new_tokens=3, ignore_eos=True)
    sched.add(s)
    for _ in range(2):
        _step(sched, token=EOS)
        assert s.status == SeqStatus.RUNNING
    _step(sched, token=EOS)
    assert s.status == SeqStatus.FINISHED and len(s.output_ids) == 3


def test_max_new_tokens_one():
    """Finishes right after prefill and is never decoded."""
    sched, alloc = _make(num_blocks=8)
    s = _seq(0, prompt_len=5, max_new_tokens=1)
    sched.add(s)
    r = _step(sched)
    assert r.is_prefill
    assert s.status == SeqStatus.FINISHED and len(s.output_ids) == 1
    assert alloc.num_free == 8 and sched.schedule() is None


# --- Preemption --------------------------------------------------------------------------------


def test_preempts_newest_when_out_of_blocks():
    sched, alloc = _make(num_blocks=4, block_size=4)
    a, b, c = _seq(0, prompt_len=4), _seq(1, prompt_len=4), _seq(2, prompt_len=8)
    for s in (a, b, c):
        sched.add(s)
    for _ in range(3):  # prefill a (1 block), b (1 block), c (2 blocks): all 4 blocks used
        _step(sched)
    assert alloc.num_free == 0
    d = _seq(3, prompt_len=4)
    sched.add(d)  # can't be admitted: no free blocks

    # a and b each need a 2nd block for position 4. None free -> preempt the NEWEST running (c).
    r = _step(sched)
    assert not r.is_prefill and _ids(r.seqs) == [0, 1]
    assert sched.num_preemptions == 1
    assert c.status == SeqStatus.WAITING and c.block_table == []
    assert c.output_ids == [TOK], "preemption keeps the generated tokens (they get recomputed on re-prefill)"
    assert _ids(sched.waiting) == [2, 3], "the preempted sequence goes to the FRONT of the waiting queue"
    assert len(a.block_table) == 2 and len(b.block_table) == 2


def test_preempted_sequence_resumes_with_its_output():
    sched, _ = _make(num_blocks=4, block_size=4)
    a, c = _seq(0, prompt_len=4, max_new_tokens=2), _seq(1, prompt_len=11)
    sched.add(a)
    sched.add(c)
    _step(sched)  # prefill a (1 block)
    _step(sched)  # prefill c (3 blocks) -> 0 free
    _step(sched)  # a needs a block -> c preempted, a decodes and finishes (2 tokens)
    assert c.status == SeqStatus.WAITING and a.status == SeqStatus.FINISHED

    r = _step(sched)  # c comes back: its prefill covers prompt + the token it already generated
    assert r.is_prefill and r.seqs == [c]
    assert c.num_tokens == 11 + 2
    assert len(c.block_table) == math.ceil(12 / 4), "re-prefill holds prompt + old output (12 tokens) in the cache"


# --- Continuous vs static batching -------------------------------------------------------------


def _admission_scenario(batching):
    """max_num_seqs=2. a (2 new tokens) and b (5 new tokens) start together; c arrives too."""
    sched, _ = _make(num_blocks=64, max_num_seqs=2, batching=batching)
    a, b, c = _seq(0, 4, max_new_tokens=2), _seq(1, 4, max_new_tokens=5), _seq(2, 4)
    for s in (a, b, c):
        sched.add(s)
    assert _ids(_step(sched).seqs) == [0]  # prefill a
    assert _ids(_step(sched).seqs) == [1]  # prefill b (batch still forming, even in static mode)
    r = _step(sched)  # decode: a reaches 2 tokens and finishes
    assert _ids(r.seqs) == [0, 1] and a.status == SeqStatus.FINISHED
    return sched, b, c


def test_continuous_admits_while_running():
    sched, b, _ = _admission_scenario("continuous")
    r = _step(sched)  # a's slot is free -> c joins right away, while b is still running
    assert r.is_prefill and _ids(r.seqs) == [2]
    assert b.status == SeqStatus.RUNNING


def test_static_waits_for_empty():
    sched, b, _ = _admission_scenario("static")
    while b.status != SeqStatus.FINISHED:  # c must wait for the whole batch (just b now) to finish
        r = _step(sched)
        assert not r.is_prefill and _ids(r.seqs) == [1], "static batching admitted c while b was running"
    r = _step(sched)
    assert r.is_prefill and _ids(r.seqs) == [2]


def test_static_second_batch_forms():
    """After a static batch drains, the NEXT batch must be able to fill up again, not just admit one sequence."""
    sched, _ = _make(num_blocks=32, batching="static")
    for i in (0, 1):
        sched.add(_seq(i, 4, max_new_tokens=3))
    while sched.has_unfinished():  # batch 1 runs to completion
        _step(sched)
    for i in (2, 3):
        sched.add(_seq(i, 4, max_new_tokens=3))
    assert _ids(_step(sched).seqs) == [2]  # prefill
    r = _step(sched)
    assert r.is_prefill and _ids(r.seqs) == [3], "batch 2 stopped forming after its first sequence"


# --- The big one: random lifecycles -----------------------------------------------------------


def _check_invariants(sched, alloc, num_blocks, cfg, result):
    owned = [blk for s in sched.running for blk in s.block_table]
    assert len(owned) == len(set(owned)), "a block is owned by two running sequences"
    assert all(alloc.ref_count(blk) == 1 for blk in owned)
    assert alloc.num_free + len(owned) == num_blocks, "leak: some block is neither free nor owned"
    assert len(sched.running) <= cfg.max_num_seqs
    for s in sched.waiting:
        assert s.status == SeqStatus.WAITING and s.block_table == []
    if result.is_prefill:
        assert len(result.seqs) == 1
    else:
        assert result.seqs == sched.running, "a decode step runs every running sequence"
    for s in result.seqs:
        assert s.status == SeqStatus.RUNNING
        assert len(s.block_table) * cfg.block_size >= s.num_tokens, f"seq {s.seq_id}: no slot for its tokens"
        if cfg.kv_policy == "reserve_max":
            assert len(s.block_table) == math.ceil(cfg.max_model_len / cfg.block_size)


@pytest.mark.parametrize(
    "kv_policy, batching",
    [("on_demand", "continuous"), ("on_demand", "static"), ("reserve_max", "continuous")],
)
def test_no_leaks_randomized(kv_policy, batching):
    """1,000 requests arriving over time into a tiny cache (12 blocks of 4 tokens), random EOS.

    Checks the bookkeeping after EVERY step, and that everything finishes with every block returned.
    """
    rng = random.Random(0)
    num_blocks = 12
    sched, alloc = _make(num_blocks, block_size=4, max_num_seqs=4, max_model_len=32,
                         kv_policy=kv_policy, batching=batching)
    cfg = sched.cfg
    seqs = []

    for _ in range(200_000):
        if len(seqs) < 1000 and rng.random() < 0.5:
            s = Sequence(
                seq_id=len(seqs),
                prompt_ids=[rng.randrange(100) for _ in range(rng.randint(1, 16))],
                max_new_tokens=rng.randint(1, 16),  # prompt + output <= 32 = max_model_len
                ignore_eos=rng.random() < 0.2,
            )
            sched.add(s)
            seqs.append(s)
        result = sched.schedule()
        if result is None:
            if len(seqs) == 1000:
                break
            continue
        _check_invariants(sched, alloc, num_blocks, cfg, result)
        tokens = [EOS if rng.random() < 0.05 else rng.randrange(100) for _ in result.seqs]
        sched.update(result.seqs, tokens, EOS_IDS)
    else:
        pytest.fail("didn't finish in 200,000 steps: is something stuck (livelock)?")

    assert not sched.has_unfinished()
    assert alloc.num_free == num_blocks, "blocks leaked"
    for s in seqs:
        assert s.status == SeqStatus.FINISHED and s.block_table == []
        assert 1 <= len(s.output_ids) <= s.max_new_tokens
        stopped_on_eos = s.output_ids[-1] == EOS and not s.ignore_eos
        assert stopped_on_eos or len(s.output_ids) == s.max_new_tokens, f"seq {s.seq_id} stopped early"
        if not s.ignore_eos:
            assert EOS not in s.output_ids[:-1], f"seq {s.seq_id} kept going after EOS"
    if kv_policy == "on_demand" and batching == "continuous":
        assert sched.num_preemptions > 0, "this workload should have needed preemption"
