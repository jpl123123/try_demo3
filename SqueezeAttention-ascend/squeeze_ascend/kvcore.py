"""kvcore for SqueezeAttention-ascend: pure window math + 1D KMeans (L0).

The window layout (faithful to SqueezeAttention's streaming drop):
  view tokens = [0, start_size) ∪ [true_len - recent, true_len)
  recent = window - start_size
The block cache can only express whole blocks, so the view row is
  [sink blocks] + [recent blocks], and the FIA reads `view_len` positions:
  view_len = true_len - (recent_first - sink_blocks) * block_size
(block-boundary over-inclusion of at most one block on each side is the
documented approximation; no padding slots are ever read).
"""

from __future__ import annotations

import math


def window_block_ranges(
    true_seq_len: int,
    window: int,
    start_size: int,
    block_size: int,
) -> tuple[int, int, int] | None:
    """Return (sink_blocks, recent_first, last_block) or None when the window
    covers the whole sequence (no rewrite needed)."""
    window = max(1, window)
    if true_seq_len <= window:
        return None
    recent = max(0, window - start_size)
    sink_blocks = math.ceil(start_size / block_size) if start_size > 0 else 0
    recent_first = max(0, (true_seq_len - recent) // block_size)
    last_block = math.ceil(true_seq_len / block_size)
    if recent_first < sink_blocks:
        # the recent range reaches into the sink blocks: drop the overlap from
        # the recent part (those tokens are already covered by the sink part)
        recent_first = sink_blocks
    if sink_blocks >= last_block:
        return None
    return sink_blocks, recent_first, last_block


def window_view_len(true_seq_len: int, sink_blocks: int, recent_first: int,
                    block_size: int) -> int:
    """Number of view positions the FIA attends: caps inside the last block so
    zero padding is never read. Always >= 1."""
    return max(1, true_seq_len - (recent_first - sink_blocks) * block_size)


def window_view_row(
    true_row: "object",
    true_seq_len: int,
    window: int,
    start_size: int,
    block_size: int,
) -> tuple["object", int] | None:
    """Full view row (block ids) + view seq len, or None when no rewrite."""
    import numpy as np

    row = np.asarray(true_row, dtype=np.int64)
    ranges = window_block_ranges(true_seq_len, window, start_size, block_size)
    if ranges is None:
        return None
    sink_blocks, recent_first, last_block = ranges
    view = np.concatenate([row[:sink_blocks], row[recent_first:last_block]])
    view_len = window_view_len(true_seq_len, sink_blocks, recent_first, block_size)
    return view, view_len


# ---------------------------------------------------------------------------
# Layer importance clustering (SqueezeAttention's class 1/2/3 budgets)
# ---------------------------------------------------------------------------


def kmeans1d(values: "object", k: int = 3, max_iter: int = 50, seed: int = 42) -> "object":
    """Deterministic 1D k-means. Returns labels (same order as values).

    Initial centers are the min / mid / max (quantile-spread) so the result is
    deterministic across TP ranks that share the same input.
    """
    import numpy as np

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n <= k:
        return np.zeros(n, dtype=np.int64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros(n, dtype=np.int64)
    centers = np.array([lo + (hi - lo) * (i + 0.5) / k for i in range(k)])
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.abs(x[:, None] - centers[None, :])
        new_labels = np.argmin(dists, axis=1)
        moved = np.any(new_labels != labels)
        labels = new_labels
        for c in range(k):
            members = x[labels == c]
            if members.size:
                centers[c] = members.mean()
        if not moved:
            break
    return labels


def layer_windows(
    means: "object",
    num_layers: int,
    ini_size: float,
    class3_ratio: float,
    start_size: int,
    prompt_len: int,
) -> tuple[list[int], dict]:
    """SqueezeAttention budget formula.

    Class 3 = layers with the HIGHEST input/output cosine similarity (attention
    changes the representation least) -> smallest window (class3_ratio).
    Classes 1/2 share the remaining budget: a = (N*ini - n3*ratio) / n12.
    Returns (windows per layer in layer-index order, info dict).
    """
    import numpy as np

    info: dict = {}
    if prompt_len <= 1 or num_layers <= 0:
        return [prompt_len] * num_layers, info
    if ini_size <= 0.0:
        # no budget -> no compression (windows cover the whole prompt)
        info["disabled"] = True
        return [prompt_len] * num_layers, info
    labels = kmeans1d(means, k=3)
    centers = np.array([float(np.mean(np.asarray(means)[labels == c])) if np.any(labels == c) else 0.0
                        for c in range(3)])
    order = np.argsort(centers)  # ascending -> order[2] is the highest mean class
    class3 = int(order[2])
    n3 = int((labels == class3).sum())
    n12 = num_layers - n3
    ratio = min(max(class3_ratio, 0.0), 1.0)
    ini = max(ini_size, 0.0)
    if n12 <= 0:
        a = 0.0
    else:
        a = (num_layers * ini - n3 * ratio) / n12
    windows = []
    for lab in labels:
        if int(lab) == class3:
            w = int(ratio * prompt_len)
        else:
            w = int(a * prompt_len)
        w = min(max(w, start_size + 1), prompt_len)
        windows.append(w)
    info.update({
        "class3": int(class3),
        "n_class3": int(n3),
        "a_ratio": float(a),
        "centers": [round(float(c), 4) for c in centers],
        "labels": [int(l) for l in labels],
    })
    return windows, info


def window_stats(windows: list[int]) -> dict:
    if not windows:
        return {"min": 0, "max": 0}
    return {"min": min(windows), "max": max(windows)}
