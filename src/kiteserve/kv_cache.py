"""Paged KV cache: a reference-counted block allocator plus the K/V pool (write and gather).

The big idea (like virtual memory in an OS):
    * GPU memory for K/V is ONE big pool, cut into fixed-size BLOCKS of `block_size` tokens.
    * Each sequence owns a list of block ids -- its BLOCK TABLE -- e.g. [7, 3, 9].
      Token at position p lives in block  block_table[p // block_size],  at offset  p % block_size.
    * With the default on_demand policy, blocks are handed out only when a sequence needs one,
      instead of reserving max_model_len for every request up front (the reserve_max policy).

Layout of the pool (one for K, one for V):
    [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
A "slot" is one token's place in the pool:  slot = block_id * block_size + offset.
Flattening dims (num_blocks, block_size) -> num_blocks * block_size makes a slot a plain
index into that flat dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kiteserve.config import ModelConfig


def slot_for(block_table: list[int], pos: int, block_size: int) -> int:
    """Physical slot of the token at logical position `pos` of a sequence.

    Example: block_table=[7, 3, 9], block_size=16, pos=17  ->  block 3, offset 1  ->  3*16 + 1 = 49
    """
    index = pos // block_size
    offset = pos % block_size
    return block_table[index] * block_size + offset

class BlockAllocator:
    """Hands out block ids 0 .. num_blocks-1 and takes them back.

    Every block has a REFERENCE COUNT (how many sequences use it):
        allocate()  -> a free block, count set to 1
        incref(b)   -> count += 1   (nothing shares blocks yet: groundwork for prefix caching)
        free(b)     -> count -= 1, and the block becomes free again when the count hits 0
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.ref_counts = [0] * num_blocks
        self.free_list = list(range(num_blocks-1, -1, -1))  # stack of free block ids
    
    def _check_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError("Invalid block id")
    
    def allocate(self) -> int:
        """Raise RuntimeError if no block is free."""
        if not self.free_list:
            raise RuntimeError("No free blocks available")
        block_id = self.free_list.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> None:
        """Raise ValueError if the block isn't currently allocated (count is 0)."""
        self._check_block_id(block_id)
        if self.ref_counts[block_id] == 0:
            raise ValueError("Block is not currently allocated")
        self.ref_counts[block_id] += 1

    def free(self, block_id: int) -> None:
        """Raise ValueError if the count is already 0 (double free = a scheduler bug)."""
        self._check_block_id(block_id)
        if self.ref_counts[block_id] == 0:
            raise ValueError("Double free detected")
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_list.append(block_id)

    def ref_count(self, block_id: int) -> int:
        self._check_block_id(block_id)
        return self.ref_counts[block_id]

    @property
    def num_free(self) -> int:
        return len(self.free_list)


class KVCache:
    """The two big pre-allocated tensors, plus write (store new K/V) and gather (read a batch back)."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.k = torch.zeros(cfg.num_layers, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
        self.v = torch.zeros(cfg.num_layers, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)

    @staticmethod
    def num_blocks_for(
        cfg: ModelConfig, block_size: int, kv_memory_gb: float, dtype: torch.dtype = torch.bfloat16
    ) -> int:
        """How many whole blocks fit in kv_memory_gb GiB (1 GiB = 1024**3 bytes)?

        One block holds block_size tokens of K *and* V for *every* layer.
        """        
        bytes_per_token = cfg.kv_bytes_per_token(dtype_bytes=torch.tensor([], dtype=dtype).element_size())
        bytes_per_block = block_size * bytes_per_token
        bytes_per_gb = 1024 ** 3
        return int(kv_memory_gb * bytes_per_gb // bytes_per_block)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Store new tokens' K/V for one layer.

        k, v:          [N, num_kv_heads, head_dim]   -- N new tokens (may come from many sequences)
        slot_mapping:  [N] long                      -- where token i goes
        Writes go through a flat [num_blocks * block_size, kv_heads, head_dim] view of self.k[layer];
        a view shares memory, so indexing into it writes into the pool.
        """
        k_cache = self.k[layer] # [num_blocks, block_size, kv_heads, head_dim]
        k_cache = k_cache.view(-1,self.k.size(-2),self.k.size(-1)) # [num_blocks * block_size, kv_heads, head_dim]
        
        v_cache = self.v[layer] # [num_blocks, block_size, kv_heads, head_dim]
        v_cache = v_cache.view(-1,self.v.size(-2),self.v.size(-1)) # [num_blocks * block_size, kv_heads, head_dim]

        k_cache[slot_mapping] = k
        v_cache[slot_mapping] = v
    
    def gather(
        self, layer: int, block_tables: torch.Tensor, context_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read back each sequence's K/V history for one layer, as padded contiguous tensors.

        block_tables:  [B, M] long  -- row b = sequence b's block table, padded with 0 to M blocks
        context_lens:  [B] long     -- how many tokens of sequence b are real
        Positions >= context_lens[b] are garbage (padding / other blocks); the CALLER masks them.
        Advanced indexing (self.k[layer][block_tables]) copies the blocks -> [B, M, block_size, kvh, hd].
        """
        block_tables = block_tables.to(torch.long) # [B, M]

        k_cache = self.k[layer] # [num_blocks, block_size, kv_heads, head_dim]
        v_cache = self.v[layer] # [num_blocks, block_size, kv_heads, head_dim]

        # Gather the K/V for each sequence according to its block table
        k_cache = k_cache[block_tables] # [B, M, block_size, kv_heads, head_dim]
        v_cache = v_cache[block_tables] # [B, M, block_size, kv_heads, head_dim]

        k_cache = k_cache.view(block_tables.size(0), -1, k_cache.size(-2), k_cache.size(-1)) # [B, M * block_size, kv_heads, head_dim]
        v_cache = v_cache.view(block_tables.size(0), -1, v_cache.size(-2), v_cache.size(-1)) # [B, M * block_size, kv_heads, head_dim]
        return k_cache, v_cache


@dataclass
class AttnMeta:
    """What attention needs to know about this step's batch (built by the engine each step).

    Prefill (one sequence, N = its length):
        is_prefill=True, slot_mapping=[N], block_tables=None, context_lens=None
    Decode (B sequences, one new token each, so N == B):
        is_prefill=False, slot_mapping=[B], block_tables=[B, M], context_lens=[B]
        context_lens[b] counts the new token too (it's written before attention reads).
    """

    is_prefill: bool
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
