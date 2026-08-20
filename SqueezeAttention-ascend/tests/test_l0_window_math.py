"""L0: pure-logic tests for squeeze_ascend.kvcore (window math + KMeans + budgets)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from squeeze_ascend.kvcore import (  # noqa: E402
    kmeans1d,
    layer_windows,
    window_block_ranges,
    window_view_len,
    window_view_row,
)


class TestWindowRanges:
    def test_no_rewrite_when_window_covers_all(self):
        assert window_block_ranges(100, 200, 4, 16) is None
        assert window_block_ranges(100, 100, 4, 16) is None

    def test_ranges_basic(self):
        # true=1000, window=100, start=4, bs=16
        sink, recent_first, last = window_block_ranges(1000, 100, 4, 16)
        assert sink == 1
        assert recent_first == (1000 - 96) // 16  # 56
        assert last == math.ceil(1000 / 16)  # 63

    def test_overlap_clamped(self):
        # window nearly covers the sequence: recent range reaches sink blocks
        sink, recent_first, last = window_block_ranges(105, 100, 4, 16)
        assert recent_first >= sink  # clamped: no duplicate blocks

    def test_view_len_caps_inside_last_block(self):
        # true=1000, window=100, start=4: view_len = 1000 - (56-1)*16 = 120
        sink, recent_first, last = window_block_ranges(1000, 100, 4, 16)
        assert window_view_len(1000, sink, recent_first, 16) == 120
        # last read position must be < true_len (no padding read)
        assert window_view_len(1000, sink, recent_first, 16) <= 1000

    def test_view_row(self):
        row = np.arange(1000, dtype=np.int64)
        view, view_len = window_view_row(row, 1000, 100, 4, 16)
        sink, recent_first, last = window_block_ranges(1000, 100, 4, 16)
        assert np.array_equal(view, np.concatenate([row[:sink], row[recent_first:last]]))
        assert view_len == window_view_len(1000, sink, recent_first, 16)


class TestKMeans:
    def test_three_clear_clusters(self):
        x = np.array([0.1, 0.2, 0.15, 0.5, 0.6, 0.55, 0.9, 0.95, 0.85])
        labels = kmeans1d(x, k=3)
        assert len(set(labels.tolist())) == 3
        # cluster order is deterministic
        labels2 = kmeans1d(x, k=3)
        assert np.array_equal(labels, labels2)

    def test_deterministic(self):
        rng = np.random.default_rng(7)
        x = rng.random(40)
        assert np.array_equal(kmeans1d(x, k=3), kmeans1d(x, k=3))


class TestBudgets:
    def test_layer_windows_budget_conservation(self):
        means = [0.9, 0.5, 0.1, 0.85]  # 4 layers
        windows, info = layer_windows(means, 4, ini_size=0.3, class3_ratio=0.1,
                                      start_size=4, prompt_len=1000)
        assert len(windows) == 4
        for w in windows:
            assert 4 + 1 <= w <= 1000
        # total budget ≈ num_layers * ini_size * prompt_len (within tolerance)
        total = sum(windows)
        assert abs(total - 4 * 0.3 * 1000) <= 2 * 1000  # slack from clamping

    def test_class3_gets_smallest_window(self):
        # highest cos sim (least-changing) layers -> class 3 -> smallest window
        means = [0.95, 0.1, 0.2, 0.9]
        windows, info = layer_windows(means, 4, ini_size=0.4, class3_ratio=0.05,
                                      start_size=4, prompt_len=2000)
        c3 = info["class3"]
        c3_windows = [windows[i] for i in range(4) if info["labels"][i] == c3]
        other_windows = [windows[i] for i in range(4) if info["labels"][i] != c3]
        assert c3_windows and other_windows
        assert max(c3_windows) <= min(other_windows)

    def test_all_layers_full_when_no_budget(self):
        windows, info = layer_windows([0.5, 0.5, 0.5], 3, ini_size=0.0, class3_ratio=0.0,
                                      start_size=4, prompt_len=100)
        assert all(w == 100 for w in windows)
        assert info.get("disabled") is True
