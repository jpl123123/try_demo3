"""kvcore: device-agnostic pure logic for kvpress-ascend (L0 testable).

Two layout modes:

* ``view`` (default): block-granular pruning expressed purely as per-layer
  *view rows* over the shared true block table. Cache content, positions and
  slot mapping are never touched -> prefix caching stays valid, per-layer
  budgets are possible, MTP draft (separate KV groups) is unaffected.

* ``compact`` (opt-in): token-granular (head-uniform) top-k physically packed
  into the request's tail blocks. Requires `KVPRESS_ASCEND_PREFIX_CACHE=force`
  and a uniform per-request layout (shared slot mapping cannot express
  per-layer positions).

All functions here operate on numpy / plain python so L0 tests run without
torch, vllm or NPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Keep-set math
# ---------------------------------------------------------------------------


def head_uniform_keep_indices(
    scores: "object",
    n_kept: int,
) -> "object":
    """Head-uniform top-k: scores shape (num_kv_heads, seq) -> (seq,) indices.

    The block KV layout shares one block across all kv heads, so the keep set
    must be identical for every head. Averaging over heads first is the
    documented approximation of kvpress's per-head topk.
    """
    # numpy or torch agnostic: delegate to the array's own ops.
    import numpy as np

    if hasattr(scores, "mean") and not isinstance(scores, np.ndarray):
        # torch tensor path
        agg = scores.mean(dim=0)
        return agg.topk(n_kept, dim=-1).indices.sort().values
    agg = np.mean(scores, axis=0)
    return np.argsort(agg)[-n_kept:].astype(np.int64)


def block_keep_indices(
    scores: "object",
    num_blocks: int,
    block_size: int,
    n_kept_blocks: int,
) -> "object":
    """Block-granular keep: aggregate per-token scores per block (mean), then
    keep the top ``n_kept_blocks`` blocks. Returns block indices (ascending)."""
    import numpy as np

    if hasattr(scores, "reshape") and not isinstance(scores, np.ndarray):
        # torch tensor path: scores (seq,)
        blocks = _torch_block_agg(scores, num_blocks, block_size)
        return blocks.topk(n_kept_blocks, dim=-1).indices.sort().values
    seq = scores.shape[-1]
    padded = np.pad(np.asarray(scores, dtype=np.float64), (0, (-seq) % block_size))
    blocks = padded.reshape(num_blocks, block_size).mean(axis=1)
    return np.argsort(blocks)[-n_kept_blocks:].astype(np.int64)


def _torch_block_agg(scores, num_blocks: int, block_size: int):
    import torch

    seq = scores.shape[-1]
    pad = (-seq) % block_size
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad))
    return scores.view(num_blocks, block_size).mean(dim=1)


def n_kept_tokens(orig_len: int, compression_ratio: float, min_keep: int = 1) -> int:
    """Number of kept tokens for a given ratio. Always >= 1 and < orig_len
    (except ratio=0 which keeps everything)."""
    if not 0.0 <= compression_ratio < 1.0:
        raise ValueError(f"compression_ratio must be in [0, 1), got {compression_ratio}")
    if compression_ratio == 0.0 or orig_len <= 1:
        return orig_len
    return max(min_keep, min(orig_len - 1, int(orig_len * (1.0 - compression_ratio))))


def n_kept_blocks(orig_blocks: int, compression_ratio: float, min_keep: int = 1) -> int:
    if compression_ratio == 0.0 or orig_blocks <= 1:
        return orig_blocks
    return max(min_keep, min(orig_blocks - 1, int(orig_blocks * (1.0 - compression_ratio))))


# ---------------------------------------------------------------------------
# Layout state
# ---------------------------------------------------------------------------


@dataclass
class CompactLayout:
    """Per-request layout for compact mode (uniform across layers)."""

    req_id: str
    orig_len: int          # prompt length at compaction
    n_kept: int            # kept tokens (head-uniform)
    block_size: int
    m: int                 # blocks allocated at compaction: ceil(orig_len/bs)
    k: int                 # kept blocks (tail region): m - delta // bs

    @property
    def delta(self) -> int:
        return self.orig_len - self.n_kept

    @property
    def slack(self) -> int:
        """d' = k*bs - n_kept >= m*bs - orig_len (the skill's slack invariant)."""
        return self.k * self.block_size - self.n_kept

    def check_slack(self) -> bool:
        return self.slack >= self.m * self.block_size - self.orig_len

    @staticmethod
    def build(req_id: str, orig_len: int, n_kept: int, block_size: int) -> "CompactLayout":
        m = math.ceil(orig_len / block_size)
        delta = orig_len - n_kept
        k = m - delta // block_size
        assert k >= 1, f"k={k} < 1 for orig_len={orig_len} n_kept={n_kept} bs={block_size}"
        return CompactLayout(req_id=req_id, orig_len=orig_len, n_kept=n_kept,
                             block_size=block_size, m=m, k=k)

    def rewrite_row(self, true_row: "object") -> "object":
        """One-shot row permutation: [b_{m-k}..b_{m-1}] + [b_m..valid].

        Must be applied exactly once, at the first _prepare_inputs after
        compaction (on the ORIGINAL row). After that the row is stable:
        vllm's append_row keeps appending new blocks after the permuted
        content and the layout never changes again (until preemption clears
        the row). CaptureManager tracks this with `row_rewritten`.
        """
        import numpy as np

        row = np.asarray(true_row, dtype=np.int64)
        valid = len(row)
        kept = row[self.m - self.k:self.m]
        rest = row[self.m:valid]
        return np.concatenate([kept, rest]) if len(rest) else kept


@dataclass
class ViewLayout:
    """Per-request per-layer layout for view mode.

    View row = [kept blocks (ascending)] + [true row from index m .. valid].

    `kept_blocks` holds PHYSICAL block ids. The caller is responsible for
    ensuring the prompt's LAST block (index m-1) is part of the kept set when
    the prompt is not block-aligned: the first new decode tokens land in that
    block's padding slots (true vllm slot semantics) and must stay visible.
    """

    req_id: str
    layer_name: str
    orig_len: int                # prompt length at compaction
    block_size: int
    m: int                       # blocks at compaction: ceil(orig_len/bs)
    n_kept: int                  # kept TOKENS = sum of min(bs, orig - b*bs) over kept blocks
    kept_blocks: "object"        # list[int]: PHYSICAL block ids in ascending order
    n_kept_blocks: int           # len(kept_blocks)

    @staticmethod
    def build(req_id: str, layer_name: str, orig_len: int,
              kept_block_ids: Sequence[int], n_kept: int, block_size: int) -> "ViewLayout":
        kept = sorted(set(int(b) for b in kept_block_ids))
        m = math.ceil(orig_len / block_size)
        return ViewLayout(
            req_id=req_id, layer_name=layer_name, orig_len=orig_len,
            block_size=block_size, m=m, n_kept=max(1, int(n_kept)),
            kept_blocks=kept, n_kept_blocks=len(kept),
        )

    def view_row(self, true_row: "object") -> "object":
        """View row = [kept blocks] + [true row from index m .. valid]."""
        import numpy as np

        row = np.asarray(true_row, dtype=np.int64)
        valid = len(row)
        rest = row[self.m:valid]
        kept = np.asarray(self.kept_blocks, dtype=np.int64)
        return np.concatenate([kept, rest]) if len(rest) else kept

    def view_seq_len(self, true_seq_len: int) -> int:
        """Number of KV positions the FIA should attend for this layer.

        kept tokens (block-granular, partial-aware) + new tokens since
        compaction. Capping inside the last kept block guarantees zero-padding
        slots are never read.
        """
        return max(1, self.n_kept + (true_seq_len - self.orig_len))


# ---------------------------------------------------------------------------
# SqueezeAttention-style window layout (used by the squeeze package; kept here
# as shared math is not needed — the squeeze package has its own kvcore)
# ---------------------------------------------------------------------------


def window_block_ranges(
    true_seq_len: int,
    window: int,
    start_size: int,
    block_size: int,
) -> tuple[int, int] | None:
    """Block range [first, last) of the true row covered by the window
    [0, start_size) ∪ [true_seq_len - (window - start_size), true_seq_len).

    Returns None when the window covers the whole sequence (no rewrite needed).
    """
    recent = max(0, window - start_size)
    if true_seq_len <= window:
        return None
    sink_blocks = math.ceil(start_size / block_size) if start_size > 0 else 0
    recent_start = true_seq_len - recent
    first_recent_block = recent_start // block_size
    last_block = math.ceil(true_seq_len / block_size)
    if sink_blocks >= last_block:
        return None
    return sink_blocks, last_block


def window_view_row(
    true_row: "object",
    true_seq_len: int,
    window: int,
    start_size: int,
    block_size: int,
) -> tuple["object", int] | None:
    """Full view row + view seq len for a window layout, or None if the window
    covers the whole sequence."""
    import numpy as np

    row = np.asarray(true_row, dtype=np.int64)
    ranges = window_block_ranges(true_seq_len, window, start_size, block_size)
    if ranges is None:
        return None
    sink_blocks, last_block = ranges
    recent = max(0, window - start_size)
    view = np.concatenate([row[:sink_blocks], row[recent_start_block(true_seq_len, recent, block_size):last_block]])
    view_len = min(true_seq_len, start_size + recent)
    return view, view_len


def recent_start_block(true_seq_len: int, recent: int, block_size: int) -> int:
    start = true_seq_len - recent
    return max(0, start // block_size)


# ---------------------------------------------------------------------------
# Sequence-lens correction
# ---------------------------------------------------------------------------


def corrected_seq_lens(
    true_seq_lens: Sequence[int],
    per_req_delta: Sequence[int],
    pad: int = 0,
) -> list[int]:
    """seq_lens - delta per request, never below 1, padded with `pad`."""
    out = []
    for seq, d in zip(true_seq_lens, per_req_delta):
        out.append(max(1, int(seq) - int(d)))
    out.extend([pad] * (len(true_seq_lens) - len(out)))
    return out
