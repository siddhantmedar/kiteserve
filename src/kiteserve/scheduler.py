"""Scheduler: decides what the GPU runs each step. Pure Python bookkeeping: no model, no GPU.

Every engine step:
    result = scheduler.schedule()                 # ONE prefill of one sequence, or ONE decode of all running
    ... the engine runs the model on result.seqs and samples one token per sequence ...
    scheduler.update(result.seqs, tokens, eos)    # append tokens, finish sequences, free their blocks

The KV-cache rule that all the block math follows:
    The cache holds every token of a sequence EXCEPT the last sampled one.
    (The last sampled token's K/V gets written by the NEXT step, when it's fed in as input.)
    So before running a step, a sequence needs  len(block_table) * block_size >= num_tokens.
        prefill: writes all num_tokens tokens (prompt + any output it already had)
        decode:  writes 1 token, at position num_tokens - 1
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil

from kiteserve.config import EngineConfig
from kiteserve.kv_cache import BlockAllocator


class SeqStatus(Enum):
    WAITING = auto()  # in the waiting queue; owns no blocks
    RUNNING = auto()  # in the running list, owns blocks holding its K/V
    FINISHED = auto()  # done; its blocks have been freed


@dataclass(eq=False)  # eq=False: two Sequences are "equal" only if they're the same object (and hashable)
class Sequence:
    seq_id: int  # stable id from the engine. NEVER a batch row: rows change every step, ids don't.
    prompt_ids: list[int] # the input prompt token IDs
    max_new_tokens: int  # maximum number of new tokens to generate
    temperature: float = 0.0  # sampling temperature
    ignore_eos: bool = False  # whether to ignore the end-of-sequence token
    output_ids: list[int] = field(default_factory=list)  # tokens generated so far
    status: SeqStatus = SeqStatus.WAITING  # current status of the sequence
    block_table: list[int] = field(default_factory=list)  # blocks allocated for this sequence's K/V cache

    @property
    def num_tokens(self) -> int:
        """Prompt plus everything generated so far."""
        return len(self.prompt_ids) + len(self.output_ids)
    
    @property
    def all_ids(self) -> list[int]:
        """prompt_ids followed by output_ids (what a prefill feeds the model, e.g. after preemption)."""
        return self.prompt_ids + self.output_ids


@dataclass
class ScheduleResult:
    is_prefill: bool
    seqs: list[Sequence]  # prefill: exactly one; decode: all running, oldest first


class Scheduler:
    def __init__(self, cfg: EngineConfig, allocator: BlockAllocator):
        self.cfg = cfg
        self.allocator = allocator
        self.waiting: deque[Sequence] = deque()  # arrival order; preempted sequences go back to the FRONT
        self.running: list[Sequence] = []  # admission order: running[0] is the oldest, running[-1] the newest
        self.num_preemptions = 0  # how many times a running sequence was evicted
        self.peak_running = 0  # largest len(running) ever seen
        # Static batch: running=[A] -> [] when update() finishes A or schedule() preempts A.
        # Both cases should reopen admission (forming_batch=True); if B remains, keep it closed.
        self.forming_batch = True  # for static mode: True while the batch is still forming
    
    def add(self, seq: Sequence) -> None:
        """Add a sequence to the waiting queue."""
        self.waiting.append(seq)
    
    def has_unfinished(self) -> bool:
        """Anything waiting or running?"""
        return bool(self.waiting or self.running)

    def _free_seq(self, seq: Sequence) -> None:
        """Free all blocks of the sequence and clear its block table."""
        for block in seq.block_table:
            self.allocator.free(block)
        seq.block_table.clear()

    def _allocate_blocks(self, seq: Sequence, count: int = 1) -> None:
        """Allocate count blocks and add them to the sequence's table."""
        for _ in range(count):
            seq.block_table.append(self.allocator.allocate())

    def schedule(self) -> ScheduleResult | None:
        """Pick this step's work. Returns None when there's nothing to run."""
        # This only selects work and prepares cache capacity; the engine runs the model after schedule() returns.
        # PREFILL PATH: admit one sequence and return before reaching decode.
        request = self.waiting[0] if self.waiting else None
        may_prefill = (
            request is not None
            and len(self.running) < self.cfg.max_num_seqs
            and (self.cfg.batching != "static" or self.forming_batch)
        )
        if may_prefill:
            if self.cfg.kv_policy == "on_demand":
                num_blocks = ceil(request.num_tokens / self.cfg.block_size)
            else:
                num_blocks = ceil(self.cfg.max_model_len / self.cfg.block_size)

            if self.allocator.num_free >= num_blocks:
                self._allocate_blocks(request, num_blocks)
                request.status = SeqStatus.RUNNING
                self.running.append(request)
                self.waiting.popleft()
                self.peak_running = max(self.peak_running, len(self.running))
                return ScheduleResult(is_prefill=True, seqs=[request])

        if not self.running: # No sequences are currently running, so nothing to schedule.
            return None

        # DECODE PATH STARTS HERE: prepare one input token per running sequence.
        # The engine executes the model after schedule() returns this decision.
        # Ensure every running sequence has room for the input token it will
        # process. If necessary, evict the newest sequence and retry.
        while True:
            needs_preemption = False
            for seq in self.running:
                if len(seq.block_table) * self.cfg.block_size < seq.num_tokens:
                    if self.allocator.num_free:
                        self._allocate_blocks(seq)
                    else:
                        newest = self.running.pop()
                        self._free_seq(newest)
                        newest.status = SeqStatus.WAITING
                        self.waiting.appendleft(newest)
                        self.num_preemptions += 1
                        needs_preemption = True
                        break
            if not needs_preemption:
                break

        if not self.running: # All running sequences have been preempted, so we need to form a new batch.
            self.forming_batch = True
            return None

        self.forming_batch = False

        # Tell the engine to decode the remaining running sequences together.
        return ScheduleResult(is_prefill=False, seqs=list(self.running))

    def update(self, seqs: list[Sequence], token_ids: list[int], eos_ids: tuple[int, ...]) -> list[Sequence]:
        """Append token_ids[i] to seqs[i].output_ids, then finish the sequences that are done.

        Done = the token is in eos_ids (unless ignore_eos), or len(output_ids) == max_new_tokens.
        Finishing: status FINISHED, free its blocks, clear block_table, remove it from running.
        Returns the sequences that finished in this call.
        """
        finished_seqs = []
        for seq, token_id in zip(seqs, token_ids):
            seq.output_ids.append(token_id)
            done = (token_id in eos_ids and not seq.ignore_eos) or len(seq.output_ids) == seq.max_new_tokens
            if done:
                seq.status = SeqStatus.FINISHED
                self._free_seq(seq)
                self.running.remove(seq)
                finished_seqs.append(seq)

        if not self.running:
            self.forming_batch = True
        
        return finished_seqs
