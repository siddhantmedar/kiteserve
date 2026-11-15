"""Turning logits into the next token: greedy (temperature 0) or sampled from softmax(logits / T)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 = greedy (always the most likely token)
    ignore_eos: bool = False  # keep generating past EOS (for fixed-length runs)


def sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """One next token per row.

    logits:        [B, vocab]  (float32: the engine upcasts before calling)
    temperatures:  [B]         row b is greedy if temperatures[b] == 0
    Returns        [B] long

    temperature T > 0: sample from softmax(logits / T). T < 1 sharpens the distribution, T > 1 flattens it.
    Rows can mix greedy and sampled, so boolean masks pick the rows for each path. Only sampled rows
    are divided by their temperature, so a temperature of 0 never causes inf/nan (which
    torch.multinomial would reject).
    """
    # Compute the next token for each row, respecting the temperature.
    # Greedy rows (temperature == 0) pick the argmax.
    # Sampled rows (temperature > 0) draw from softmax(logits / temperature).

    greedy_mask = temperatures == 0 # [B] bool, True for rows that are greedy
    sampled_mask = ~greedy_mask # [B] bool, True for rows that are sampled

    next_tokens = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)  # [B] long, will hold the next token for each row

    if greedy_mask.any():
        next_tokens[greedy_mask] = logits[greedy_mask].argmax(dim=1) # [B_greedy] long, next token for each greedy row

    if sampled_mask.any():
        sampled_logits = logits[sampled_mask] / temperatures[sampled_mask, None]
        next_tokens[sampled_mask] = torch.multinomial(torch.softmax(sampled_logits, dim=1), num_samples=1).squeeze(1) # [B_sampled] long, next token for each sampled row
    
    return next_tokens # [B] long, next token for each row
