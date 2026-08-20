"""L0: pure-logic tests for kvpress_ascend.kvcore (no torch required beyond numpy)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvpress_ascend.kvcore import (  # noqa: E402
    CompactLayout,
    ViewLayout,
    block_keep_indices,
    corrected_seq_lens,
    head_uniform_keep_indices,
    n_kept_blocks,
    n_kept_tokens,
)


class TestKeepMath:
    def test_n_kept_tokens(self):
        assert n_kept_tokens(1000, 0.5) == 500
        assert n_kept_tokens(1000, 0.0) == 1000
        assert n_kept_tokens(1000, 0.999) == 1
        assert n_kept_tokens(3, 0.5) == 1
        with pytest.raises(ValueError):
            n_kept_tokens(100, 1.0)

    def test_head_uniform_keep_numpy(self):
        rng = np.random.default_rng(0)
        scores = rng.random((4, 100))  # 4 kv heads
        idx = head_uniform_keep_indices(scores, 40)
        assert idx.shape == (40,)
        # head-uniform: same set for every head (by construction of the agg)
        agg = scores.mean(axis=0)
        assert set(idx.tolist()) == set(np.argsort(agg)[-40:].tolist())

    def test_block_keep_indices_numpy(self):
        rng = np.random.default_rng(1)
        scores = rng.random(100)
        blocks = block_keep_indices(scores, 7, 16, 3)
        assert blocks.shape == (3,)
        # aggregate mean per block, take top-3
        padded = np.pad(scores, (0, 12))
        agg = padded.reshape(7, 16).mean(axis=1)
        assert set(blocks.tolist()) == set(np.argsort(agg)[-3:].tolist())


class TestCompactLayout:
    def test_slack_invariant(self):
        for orig, n_kept, bs in [(1000, 500, 128), (260, 130, 16), (100, 10, 16),
                                 (1024, 512, 128), (17, 8, 16), (1000, 999, 128)]:
            lay = CompactLayout.build("r", orig, n_kept, bs)
            assert lay.m == math.ceil(orig / bs)
            assert lay.delta == orig - n_kept
            assert lay.k == lay.m - lay.delta // bs
            assert lay.k >= 1
            assert lay.check_slack(), f"slack violated for {orig=} {n_kept=} {bs=}"

    def test_rewrite_row_once(self):
        lay = CompactLayout.build("r", 1000, 500, 128)
        m = lay.m  # 8
        row = np.arange(200, dtype=np.int64)  # fake block ids
        row[:m] = np.arange(10, 10 + m)  # prefill blocks 10..17
        # append 3 decode blocks
        valid = m + 3
        row[m:valid] = np.arange(100, 103)
        r1 = lay.rewrite_row(row[:valid])
        # tail-k blocks first, then the appended blocks
        expected = np.concatenate([np.arange(10, 10 + m)[m - lay.k:], np.arange(100, 103)])
        assert np.array_equal(r1, expected)

    def test_rewrite_row_keeps_new_blocks_order(self):
        lay = CompactLayout.build("r", 260, 130, 16)  # m=17, delta=130, k=17-8=9
        row = np.arange(500, dtype=np.int64)
        row[:17] = np.arange(50, 67)
        row[17:20] = np.arange(200, 203)
        r = lay.rewrite_row(row[:20])
        assert np.array_equal(r[:9], np.arange(58, 67))
        assert np.array_equal(r[9:], np.arange(200, 203))


class TestViewLayout:
    def test_build_keeps_physical_ids(self):
        # orig=1000, bs=128 -> m=8; kept PHYSICAL ids 10,12,15
        lay = ViewLayout.build("r", "l0", 1000, [10, 12, 15], 384, 128)
        assert lay.kept_blocks == [10, 12, 15]
        assert lay.n_kept == 384
        assert lay.n_kept_blocks == 3

    def test_view_seq_len(self):
        lay = ViewLayout.build("r", "l0", 1000, [10, 12, 15], 384, 128)
        assert lay.view_seq_len(1001) == 385
        assert lay.view_seq_len(1010) == 394

    def test_view_row(self):
        lay = ViewLayout.build("r", "l0", 1000, [10, 12, 15], 384, 128)
        row = np.arange(40, dtype=np.int64)
        row[:8] = np.arange(10, 18)  # prefill blocks
        row[8:12] = np.arange(100, 104)  # decode blocks
        view = lay.view_row(row[:12])
        assert np.array_equal(view[:3], [10, 12, 15])
        assert np.array_equal(view[3:], [100, 101, 102, 103])


class TestSeqLens:
    def test_corrected_seq_lens(self):
        out = corrected_seq_lens([100, 50, 30], [20, 0, 5])
        assert out == [80, 50, 25]
        out = corrected_seq_lens([10, 5], [9, 4])
        assert out == [1, 1]
