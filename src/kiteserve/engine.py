"""The engine: wires model + KV cache + scheduler + sampling together.

    engine = LLMEngine(EngineConfig(model_path=...))
    outputs = engine.generate([prompt_ids_1, prompt_ids_2], SamplingParams(max_new_tokens=64))

One step() = one forward pass of the model:
    scheduler.schedule()  ->  build input_ids / positions / AttnMeta  ->  model(...)  ->  sample  ->  scheduler.update()
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer
from kiteserve.config import EngineConfig, ModelConfig
from kiteserve.kv_cache import AttnMeta, BlockAllocator, KVCache, slot_for
from kiteserve.loader import load_weights
from kiteserve.model import Qwen3ForCausalLM
from kiteserve.sampling import SamplingParams, sample
from kiteserve.scheduler import Scheduler, Sequence


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        """Size the KV cache (fail fast if it can't hold one max_model_len sequence), then load the model."""
        self.cfg = cfg
        self.model_config = ModelConfig.from_pretrained(cfg.model_path)
        self.num_blocks = KVCache.num_blocks_for(
            self.model_config,
            cfg.block_size,
            cfg.kv_memory_gb,
            cfg.dtype
        )

        if self.num_blocks * cfg.block_size < self.cfg.max_model_len:
            raise ValueError("KV cache cannot hold one sequence of max_model_len tokens")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model = Qwen3ForCausalLM(self.model_config).to(cfg.device, cfg.dtype)
        load_weights(self.model, cfg.model_path)
        self.model.eval()
        self.kv_cache = KVCache(self.model_config, self.num_blocks, cfg.block_size, cfg.dtype, cfg.device)
        self.allocator = BlockAllocator(self.num_blocks)
        self.scheduler = Scheduler(cfg, self.allocator)
        self.seq_counter = 0

    def add_request(self, prompt_ids: list[int], params: SamplingParams) -> int:
        """Queue one request; return its seq_id.

        Raise ValueError for an empty prompt, or when len(prompt_ids) + max_new_tokens > cfg.max_model_len
        (it could never fit in its block table / RoPE range).
        """
        if not prompt_ids:
            raise ValueError("Prompt cannot be empty")

        if len(prompt_ids) + params.max_new_tokens > self.cfg.max_model_len:
            raise ValueError("Prompt + max_new_tokens exceeds max_model_len")

        request_seq_id = self.seq_counter
        seq = Sequence(request_seq_id, prompt_ids.copy(), params.max_new_tokens, params.temperature, params.ignore_eos)
        self.scheduler.add(seq)
        self.seq_counter += 1
        return request_seq_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    @torch.inference_mode()
    def step(self) -> list[tuple[int, int, bool]]:
        """One forward pass: schedule -> build inputs + AttnMeta -> model -> sample -> update.

        Returns (seq_id, new_token, finished) for each sequence that ran.
        Returns [] only when there is no work left; raises RuntimeError if work remains but nothing fits.
        """
        result = self.scheduler.schedule()

        if result is None:
            if self.has_unfinished():
                raise RuntimeError("scheduler stuck: waiting but nothing fits")
            else:
                return []

        if result.is_prefill:
            # Build inputs for prefill
            prefill_seq = result.seqs[0] # only one sequence for prefill
            input_ids = prefill_seq.all_ids # prompt + any output from before a preemption
            positions = list(range(prefill_seq.num_tokens))
            # slot_mapping is needed to write the prompt’s new K/V into the cache.
            # block_tables and context_lens aren’t needed because prefill attends directly to the K/V it just computed, using a causal mask. It doesn’t gather history from the cache.
            attn_meta = AttnMeta(is_prefill=True, slot_mapping=torch.tensor([slot_for(prefill_seq.block_table,pos,self.cfg.block_size) for pos in positions], dtype=torch.long, device=self.cfg.device))
            input_ids = torch.tensor(input_ids, dtype=torch.long,device=self.cfg.device)
            positions = torch.tensor(positions, dtype=torch.long,device=self.cfg.device)
        else:
            # Build inputs for decode
            # one input token per running sequence: its last sampled token, at position num_tokens - 1
            input_ids = [seq.all_ids[-1] for seq in result.seqs]
            positions = [seq.num_tokens - 1 for seq in result.seqs]
            max_blocks_in_batch = max(len(seq.block_table) for seq in result.seqs)
            
            attn_meta = AttnMeta(
                is_prefill=False,
                slot_mapping=torch.tensor([slot_for(seq.block_table,pos,self.cfg.block_size) for seq,pos in zip(result.seqs, positions)], dtype=torch.long, device=self.cfg.device),
                block_tables=torch.tensor([seq.block_table + [0] * (max_blocks_in_batch - len(seq.block_table)) for seq in result.seqs], dtype=torch.long, device=self.cfg.device),
                context_lens=torch.tensor([seq.num_tokens for seq in result.seqs], dtype=torch.long, device=self.cfg.device)
            )

            input_ids = torch.tensor(input_ids, dtype=torch.long,device=self.cfg.device)
            positions = torch.tensor(positions, dtype=torch.long,device=self.cfg.device)

        # Forward pass through the model to get logits for the current step
        logits = self.model(input_ids, positions, attn_meta, self.kv_cache)
        logits = (logits[-1:] if result.is_prefill else logits).to(torch.float32)

        temperatures = torch.tensor([seq.temperature for seq in result.seqs], dtype=torch.float32, device=self.cfg.device)

        next_tokens = sample(logits, temperatures).tolist() # B Python ints, one per sequence (in result.seqs order)

        finished_seqs = set(self.scheduler.update(result.seqs, next_tokens, self.model_config.eos_token_ids))

        return [(seq.seq_id, token, seq in finished_seqs) for seq, token in zip(result.seqs, next_tokens)]

    
    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Add every prompt, step until nothing is unfinished, return the outputs IN INPUT ORDER.

        Sequences finish in any order, and step() events can arrive interleaved, so collect tokens
        by seq_id (a dict), not by position in the step's list. Events for requests added outside
        this call are skipped.
        """
        if not prompts:
            return []

        request_seqs = dict()

        # Add each prompt to the scheduler
        for prompt in prompts:
            request_seq_id = self.add_request(prompt, params) # store the returned sequence id for later tracking
            request_seqs[request_seq_id] = []

        while self.has_unfinished():
            # Step the scheduler and collect results
            step_results = self.step()  # get the results of the current step
            for seq_id, token, finished in step_results:
                if seq_id in request_seqs:
                    request_seqs[seq_id].append(token)

        return [request_seqs[seq_id] for seq_id in sorted(request_seqs)]
