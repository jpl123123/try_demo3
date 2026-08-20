"""Ported kvpress press implementations for the block-KV world of vllm.

Adaptations vs upstream kvpress (documented, see README §limitations):

1. The block cache shares one physical block across all kv heads -> every
   press produces a HEAD-UNIFORM token score `(seq,)` (upstream: per-head
   `(kv_heads, seq)` topk). Scores that are naturally identical across heads
   (SnapKV/TOVA after group averaging) are unchanged in spirit.
2. `view` mode additionally aggregates token scores to block scores.
3. The queries captured from the Ascend backend are already RoPE-rotated, so
   window-attention presses no longer re-apply RoPE (upstream applies RoPE to
   pre-rope queries). ExpectedAttention needs pre-rope queries and re-projects
   them from captured hidden states (best-effort).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Layer context handed to every press
# ---------------------------------------------------------------------------


@dataclass
class LayerCtx:
    layer_name: str
    layer_idx: int
    num_hidden_layers: int
    num_heads: int
    num_kv_heads: int
    head_size: int

    @property
    def num_key_value_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@dataclass
class Press:
    name: str = "base"
    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0.0 <= self.compression_ratio < 1.0, "compression_ratio must be in [0, 1)"

    def score(self, ctx: LayerCtx, queries: torch.Tensor | None,
              keys: torch.Tensor, values: torch.Tensor | None,
              hidden: torch.Tensor | None) -> torch.Tensor:
        """Return head-uniform token scores of shape (k_len,). Higher = keep."""
        raise NotImplementedError

    def budget_tokens(self, ctx: LayerCtx, orig_len: int) -> int:
        """Number of tokens to keep (compact mode / reporting)."""
        from kvpress_ascend.kvcore import n_kept_tokens
        return n_kept_tokens(orig_len, self.compression_ratio)

    def budget_blocks(self, ctx: LayerCtx, orig_len: int, block_size: int) -> int:
        """Number of blocks to keep (view mode)."""
        from kvpress_ascend.kvcore import n_kept_blocks
        orig_blocks = (orig_len + block_size - 1) // block_size
        return n_kept_blocks(orig_blocks, self.compression_ratio)


@dataclass
class KnormPress(Press):
    name: str = "knorm"
    compression_ratio: float = 0.5

    def score(self, ctx, queries, keys, values, hidden):
        # keys: (k_len, kv_heads, hd) -> mean over heads -> (k_len,)
        return -keys.norm(dim=-1).mean(dim=-1)


@dataclass
class RandomPress(Press):
    name: str = "random"
    compression_ratio: float = 0.5
    seed: int | None = None

    def score(self, ctx, queries, keys, values, hidden):
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=keys.device)
            generator.manual_seed(self.seed)
        return torch.rand(keys.shape[0], generator=generator, device=keys.device, dtype=torch.float32)


@dataclass
class StreamingLLMPress(Press):
    """Sink + recent window. In view mode the block aggregation makes the
    boundary blocks partially included (documented approximation)."""
    name: str = "streamingllm"
    compression_ratio: float = 0.5
    n_sink: int = 4

    def score(self, ctx, queries, keys, values, hidden):
        k_len = keys.shape[0]
        n_pruned = k_len - self.budget_tokens(ctx, k_len)
        scores = torch.ones(k_len, device=keys.device, dtype=torch.float32)
        lo = min(self.n_sink, k_len)
        hi = min(self.n_sink + n_pruned, k_len)
        if lo < hi:
            scores[lo:hi] = 0.0
        return scores


def window_attention_scores(
    ctx: LayerCtx,
    queries: torch.Tensor,   # (window, num_heads, hd) post-rope
    keys: torch.Tensor,      # (k_len, kv_heads, hd)
    window: int,
    kernel: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Head-uniform SnapKV/TOVA-style window attention scores (k_len,)."""
    k_len = keys.shape[0]
    w = min(window, queries.shape[0], k_len)
    if w <= 0 or queries is None:
        return torch.ones(k_len, device=keys.device, dtype=torch.float32)
    if k_len - w < 1:
        # no keys left outside the window: nothing to score (upstream keeps all)
        return torch.ones(k_len, device=keys.device, dtype=torch.float32)
    q = queries[-w:].float()                       # (w, h, hd)
    num_heads, hd = q.shape[1], q.shape[2]
    kvh = ctx.num_kv_heads
    g = num_heads // kvh
    scale = scale if scale is not None else 1.0 / math.sqrt(hd)
    q = q.view(w, kvh, g, hd).transpose(0, 1)      # (kvh, w, g, hd)
    k = keys[: k_len - w].float()                  # only first k_len-w keys (upstream semantics)
    k = k.transpose(0, 1).unsqueeze(1)             # (kvh, 1, k_len-w, hd)
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale          # (kvh, w, g, k_len-w)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32)
    scores = attn.mean(dim=1)                      # (kvh, g, k_len-w)  mean over window queries
    scores = scores.mean(dim=1)                    # (kvh, k_len-w)     mean over groups
    scores = scores.mean(dim=0)                    # (k_len-w,)         mean over kv heads
    if kernel > 1 and scores.shape[0] > kernel:
        scores = F.avg_pool1d(scores.view(1, 1, -1), kernel_size=kernel,
                              padding=kernel // 2, stride=1).view(-1)
        if scores.shape[0] > k_len - w:
            scores = scores[: k_len - w]
    pad_val = scores.max().item() + 1.0
    scores = F.pad(scores, (0, w), value=pad_val)
    return scores[:k_len]


@dataclass
class SnapKVPress(Press):
    name: str = "snapkv"
    compression_ratio: float = 0.5
    window_size: int = 64
    kernel_size: int = 5

    def score(self, ctx, queries, keys, values, hidden):
        return window_attention_scores(ctx, queries, keys, self.window_size, self.kernel_size)


@dataclass
class TOVAPress(Press):
    name: str = "tova"
    compression_ratio: float = 0.5

    def score(self, ctx, queries, keys, values, hidden):
        return window_attention_scores(ctx, queries, keys, window=1, kernel=0)


@dataclass
class PyramidKVPress(SnapKVPress):
    """SnapKV scores + per-layer budget (PyramidKV formula). In compact mode
    the per-layer budgets are averaged into a uniform per-request budget
    (shared slot mapping cannot express per-layer positions)."""
    name: str = "pyramidkv"
    compression_ratio: float = 0.5
    window_size: int = 64
    kernel_size: int = 5
    beta: int = 20

    def layer_budget(self, ctx: LayerCtx, q_len: int) -> int:
        assert self.beta >= 1
        max_capacity = self.window_size + q_len * (1.0 - self.compression_ratio)
        min_num = (max_capacity - self.window_size) / self.beta
        max_num = (max_capacity - self.window_size) * 2 - min_num
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (max_capacity - self.window_size) * 2 - max_num
        if not (q_len >= max_num >= min_num >= self.window_size) or ctx.num_hidden_layers <= 1:
            return round(q_len * (1.0 - self.compression_ratio))
        steps = (max_num - min_num) / (ctx.num_hidden_layers - 1)
        return max(1, round(max_num - ctx.layer_idx * steps))

    def budget_tokens(self, ctx, orig_len):
        return max(1, min(orig_len - 1, self.layer_budget(ctx, orig_len)))


@dataclass
class ExpectedAttentionPress(Press):
    """Best-effort port: pre-rope queries are re-projected from captured hidden
    states using the attention module's qkv projection. Falls back to Knorm
    scores when the module lacks the required attributes (fail-soft)."""
    name: str = "expected_attention"
    compression_ratio: float = 0.5
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    epsilon: float = 0.0
    _fallback: bool = field(default=False)

    def _pre_rope_queries(self, module, hidden: torch.Tensor) -> torch.Tensor | None:
        # hidden: (T, H); returns (T, num_heads, hd) pre-rope queries or None.
        try:
            num_heads = ctx_heads(module)
            head_dim = ctx_head_size(module)
            q = None
            if hasattr(module, "q_proj"):
                q = module.q_proj(hidden)
            elif hasattr(module, "qkv_proj"):
                qkv = module.qkv_proj(hidden)
                q = qkv[..., : num_heads * head_dim]
            else:
                return None
            T = hidden.shape[0]
            q = q.view(T, num_heads, head_dim).transpose(0, 1)  # (h, T, hd)
            q_norm = getattr(module, "q_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            return q.transpose(0, 1)  # (T, h, hd)
        except Exception:
            return None

    def score(self, ctx, queries, keys, values, hidden):
        if hidden is None:
            return KnormPress(self.compression_ratio).score(ctx, queries, keys, values, hidden)
        module = getattr(hidden, "_attn_module", None)
        if module is None:
            return KnormPress(self.compression_ratio).score(ctx, queries, keys, values, hidden)
        q_pre = self._pre_rope_queries(module, hidden)
        if q_pre is None:
            return KnormPress(self.compression_ratio).score(ctx, queries, keys, values, hidden)
        k_len = keys.shape[0]
        hd = ctx.head_size
        mu = q_pre[self.n_sink:].mean(dim=0)  # (h, hd)
        cov = None
        if self.use_covariance:
            centered = q_pre[self.n_sink:] - mu
            cov = torch.einsum("th,ti->hi", centered, centered) / max(1, q_pre.shape[0] - self.n_sink)
        # average RoPE rotation over future positions
        rot = self._avg_rope_matrix(module, k_len, hd, q_pre.device)
        if rot is None:
            return KnormPress(self.compression_ratio).score(ctx, queries, keys, values, hidden)
        mu = mu @ rot  # (h, hd)
        keys_f = keys.float()
        logits = keys_f @ mu.t()  # (k_len, h)
        if cov is not None:
            quad = torch.einsum("kh,hi,ki->k", keys_f, cov, keys_f)  # k cov k
            logits = logits + 0.5 * quad.unsqueeze(-1)
        scores = torch.exp(logits / math.sqrt(hd)).mean(dim=-1)  # (k_len,)
        if self.use_vnorm and values is not None:
            scores = (scores + self.epsilon) * values.float().norm(dim=-1).mean(dim=-1)
        scores[self.n_sink:] = scores[self.n_sink:]  # keep sink via scores
        scores[: self.n_sink] = scores.max() + 1.0
        return scores

    def _avg_rope_matrix(self, module, q_len, hd, device):
        try:
            rotary = getattr(module, "rotary_emb", None)
            if rotary is None:
                return None
            positions = torch.arange(q_len, q_len + self.n_future_positions, device=device)
            cos_sin = None
            if hasattr(rotary, "get_cos_sin"):
                cos_sin = rotary.get_cos_sin(positions)
            elif hasattr(rotary, "forward") and hasattr(rotary, "cos_sin_cache"):
                cos_sin = rotary(positions)
            if cos_sin is None:
                return None
            cos, sin = cos_sin[0], cos_sin[1]
            cos = cos.view(-1, hd // 2).repeat(1, 2)
            sin = sin.view(-1, hd // 2).repeat(1, 2)
            Id = torch.eye(hd, device=device, dtype=cos.dtype)
            P = torch.zeros(hd, hd, device=device, dtype=cos.dtype)
            P[hd // 2:, :hd // 2] = torch.eye(hd // 2, device=device, dtype=cos.dtype)
            P[:hd // 2, hd // 2:] = -torch.eye(hd // 2, device=device, dtype=cos.dtype)
            R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P
            return R.mean(dim=0)
        except Exception:
            return None


@dataclass
class CriticalKVPress(Press):
    """Two-stage: base scores rescaled by ||Wo @ values||_1 (best-effort)."""
    name: str = "criticalkv"
    compression_ratio: float = 0.5
    first_stage_ratio: float = 0.5
    epsilon: float = 1e-4
    base: Press = field(default_factory=lambda: SnapKVPress())

    def __post_init__(self):
        assert 0.0 <= self.compression_ratio < 1.0

    def score(self, ctx, queries, keys, values, hidden):
        scores = self.base.score(ctx, queries, keys, values, hidden)
        if values is None:
            return scores
        module = getattr(values, "_attn_module", None)
        if module is None or not hasattr(module, "o_proj"):
            return scores
        try:
            k_len = keys.shape[0]
            o = module.o_proj.weight  # (H, H) local heads
            v = values.float()  # (k_len, kvh, hd)
            v = v.repeat_interleave(ctx.num_key_value_groups, dim=1)  # (k_len, h, hd)
            norm = torch.einsum("khi,hi->kh", v, o.t().view(-1, o.shape[1])).abs().sum(-1)  # approx
            norm = norm.mean(dim=-1)
            k_len2 = k_len
            budget = self.base.budget_tokens(ctx, k_len2)
            sel = int(k_len2 * self.first_stage_ratio)
            sel = max(1, min(sel, k_len2 - 1))
            top = scores.topk(sel).indices
            scores = (scores + self.epsilon) * norm
            scores[top] = torch.finfo(scores.dtype).max
            return scores
        except Exception:
            return scores


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESS_REGISTRY: dict[str, Callable[[], Press]] = {
    "knorm": lambda: KnormPress(),
    "random": lambda: RandomPress(),
    "streamingllm": lambda: StreamingLLMPress(),
    "snapkv": lambda: SnapKVPress(),
    "tova": lambda: TOVAPress(),
    "pyramidkv": lambda: PyramidKVPress(),
    "expected_attention": lambda: ExpectedAttentionPress(),
    "criticalkv": lambda: CriticalKVPress(),
}


def build_press(name: str, ratio: float, window: int, sink: int, kernel: int) -> Press:
    name = (name or "snapkv").strip().lower()
    if name not in PRESS_REGISTRY:
        name = "snapkv"
    press = PRESS_REGISTRY[name]()
    press.compression_ratio = ratio
    for attr, val in (("window_size", window), ("n_sink", sink), ("kernel_size", kernel)):
        if hasattr(press, attr):
            setattr(press, attr, val)
    if isinstance(press, CriticalKVPress):
        press.base.compression_ratio = ratio
        for attr, val in (("window_size", window), ("n_sink", sink), ("kernel_size", kernel)):
            if hasattr(press.base, attr):
                setattr(press.base, attr, val)
    return press


def ctx_heads(module) -> int:
    return int(getattr(module, "num_heads", getattr(module, "num_attention_heads", 0)))


def ctx_head_size(module) -> int:
    return int(getattr(module, "head_size", getattr(module, "head_dim", 0)))
