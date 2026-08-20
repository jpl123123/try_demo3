"""Fail-soft injection tests (D3): broken inputs must downgrade, never crash
the serving flow. Every injected failure leaves the manager in a consistent
state and records a `skipped_error` / `skipped_*` counter."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvpress_ascend import registry  # noqa: E402
from kvpress_ascend.capture import CaptureManager  # noqa: E402
from kvpress_ascend.presses import build_press  # noqa: E402
from kvpress_ascend.simulate import (  # noqa: E402
    FakeInputBatch,
    FakeRunner,
    FakeSchedulerOutput,
)


@pytest.fixture(autouse=True)
def _min_prompt_zero(monkeypatch):
    monkeypatch.setenv("KVPRESS_ASCEND_MIN_PROMPT", "0")


def make_mgr(mode="view"):
    mgr = CaptureManager()
    mgr.mode = mode
    mgr.press = build_press("knorm", ratio=0.5, window=8, sink=2, kernel=3)
    mgr.capture_w = 64
    return mgr


class TestFailSoft:
    def test_missing_kv_cache_skips_layer(self):
        runner = FakeRunner().build(num_layers=2, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        # corrupt one module: no kv_cache
        mod0 = runner.compilation_config.static_forward_context["model.layers.0.self_attn.attn"]
        del mod0.kv_cache
        mgr = make_mgr()
        before = registry.stats_snapshot().get("compressed", 0)
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))
        # layer 1 still compressed; layer 0 skipped silently
        assert registry.stats_snapshot().get("compressed", 0) >= before + 1
        assert "r0" in mgr.layouts

    def test_none_tuple_kv_cache_skips_layer(self):
        """(None, None) kv_cache tuples are NOT caught by `not kc` - the guard
        must check entries (real machine: 'NoneType' object has no attribute
        'shape' in the scorer). Other layers must still compress."""
        runner = FakeRunner().build(num_layers=2, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        mod0 = runner.compilation_config.static_forward_context["model.layers.0.self_attn.attn"]
        mod0.kv_cache = (None, None)  # the exact real-machine failure shape
        mgr = make_mgr()
        before = registry.stats_snapshot().get("compressed", 0)
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))  # must not raise
        assert registry.stats_snapshot().get("compressed", 0) >= before + 1  # layer 1 still done
        assert registry.stats_snapshot().get("skipped_no_kv", 0) >= 1
        assert "r0" in mgr.layouts

    def test_missing_static_forward_context_no_crash(self):
        runner = FakeRunner().build(num_layers=2, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        runner.compilation_config.static_forward_context = {}
        runner.vllm_config.compilation_config.static_forward_context = {}
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))  # must not raise
        assert mgr.layouts == {}

    def test_zero_row_blocks_skipped(self):
        runner = FakeRunner().build(num_layers=2, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        bt.block_table.np[0, 6] = 0  # corrupt last block id
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))
        assert "r0" not in mgr.layouts  # skipped, no crash
        assert registry.stats_snapshot().get("skipped_error", 0) >= 1

    def test_metadata_rewrite_with_mismatched_shapes(self):
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [100], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        from kvpress_ascend.kvcore import ViewLayout
        mgr = make_mgr()
        mgr.layouts["r0"] = {"model.layers.0.self_attn.attn": ViewLayout.build(
            "r0", "model.layers.0.self_attn.attn", 100, [100, 102], 30, 16)}
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([101], dtype=torch.int64)
        from kvpress_ascend.simulate import ATTN_STATES, FakeAscendMetadata
        meta = {"model.layers.0.self_attn.attn": FakeAscendMetadata(
            1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
            bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))}
        mgr.on_metadata_built(runner, meta, None)  # must not raise
        m = meta["model.layers.0.self_attn.attn"]
        assert int(m.seq_lens_list[0]) == 31  # 30 kept + 1 new

    def test_bad_row_skipped_before_device_gather(self):
        """G11 guard: a block id beyond the KV cache must be rejected on CPU
        before any device gather (real machine: AIV index out of range)."""
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.block_table.np[0, 3] = 10000  # corrupt: beyond num cache blocks (256)
        bt.num_blocks_per_row[0] = 7
        mgr = make_mgr()
        before = registry.stats_snapshot().get("skipped_bad_row", 0)
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))  # must not raise
        assert registry.stats_snapshot().get("skipped_bad_row", 0) > before
        assert "r0" not in mgr.layouts

    def test_view_rewrite_drops_out_of_range_kept(self):
        """G11 guard in the view rewrite: kept block ids outside the cache are
        dropped instead of poisoning the FIA block-table gather."""
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [100], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        from kvpress_ascend.kvcore import ViewLayout
        mgr = make_mgr()
        mgr.layouts["r0"] = {"model.layers.0.self_attn.attn": ViewLayout.build(
            "r0", "model.layers.0.self_attn.attn", 100, [9999, 100], 30, 16)}
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([101], dtype=torch.int64)
        from kvpress_ascend.simulate import ATTN_STATES, FakeAscendMetadata
        meta = {"model.layers.0.self_attn.attn": FakeAscendMetadata(
            1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
            bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))}
        mgr.on_metadata_built(runner, meta, None)  # must not raise
        assert "r0" not in mgr.layouts  # layout dropped
        assert int(meta["model.layers.0.self_attn.attn"].seq_lens_list[0]) == 101  # untouched


class TestDraftLayerExclusion:
    def test_is_draft_layer_name(self):
        from kvpress_ascend.capture import _is_draft_layer_name
        assert _is_draft_layer_name("mtp.layers.0.self_attn.attn")
        assert _is_draft_layer_name("model.layers.0.self_attn.attn") is False

    def test_mtp_layer_excluded_from_compression(self):
        """Real machine: 'mtp.layers.0.self_attn.attn' sits in the iterated
        layer list with an unbound kv_cache - exclude structurally instead of
        failing per attempt."""
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        # add an MTP layer into group 0's layer list with an unbound kv_cache
        group0 = runner.kv_cache_config.kv_cache_groups[0]
        group0.layer_names = ["model.layers.0.self_attn.attn", "mtp.layers.0.self_attn.attn"]
        no_kv_before = registry.stats_snapshot().get("skipped_no_kv", 0)
        exc_before = registry.stats_snapshot().get("layers_excluded_draft", 0)
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))
        # base layer compressed, MTP layer never scored
        assert "r0" in mgr.layouts
        assert "model.layers.0.self_attn.attn" in mgr.layouts["r0"]
        assert registry.stats_snapshot().get("skipped_no_kv", 0) == no_kv_before
        assert registry.stats_snapshot().get("layers_excluded_draft", 0) > exc_before

    def test_drafter_attn_layer_names_excluded(self):
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [0], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        class _Drafter:
            attn_layer_names = ["model.layers.0.self_attn.attn"]
        runner.drafter = _Drafter()
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 100}, 0))
        # the only base layer is excluded via the drafter list -> nothing compresses
        assert "r0" not in mgr.layouts


class TestBadViewGuard:
    def test_view_len_out_of_range_drops_layout(self):
        """A corrupted layout whose view_len exceeds the true length must be
        dropped before the FIA sees it (real machine: AIV index out of range
        after ~400 steps of healthy compression)."""
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [100], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        from kvpress_ascend.kvcore import ViewLayout
        mgr = make_mgr()
        # corrupt layout: n_kept huge -> view_len(101) = 1000 > true_len
        mgr.layouts["r0"] = {"model.layers.0.self_attn.attn": ViewLayout.build(
            "r0", "model.layers.0.self_attn.attn", 100, [100, 102], 999, 16)}
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([101], dtype=torch.int64)
        from kvpress_ascend.simulate import ATTN_STATES, FakeAscendMetadata
        meta = {"model.layers.0.self_attn.attn": FakeAscendMetadata(
            1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
            bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))}
        mgr.on_metadata_built(runner, meta, None)  # must not raise
        assert "r0" not in mgr.layouts  # dropped
        # the metadata stays untouched (true view) - nothing bad reaches the FIA
        assert int(meta["model.layers.0.self_attn.attn"].seq_lens_list[0]) == 101
        assert registry.stats_snapshot().get("skipped_bad_view", 0) >= 1

    def test_rewrite_failure_restores_metadata(self, monkeypatch):
        """An exception mid-rewrite must restore the original metadata (no
        half-rewritten state ever reaches the FIA)."""
        runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                    block_size=16, max_blocks=256, max_reqs=8)
        ib = FakeInputBatch(["r0"], [100], [100], 16, 256, 8)
        runner.input_batch = ib
        bt = ib.block_table[0]
        for b in range(7):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = 7
        from kvpress_ascend.kvcore import ViewLayout
        mgr = make_mgr()
        mgr.layouts["r0"] = {"model.layers.0.self_attn.attn": ViewLayout.build(
            "r0", "model.layers.0.self_attn.attn", 100, [100, 102], 30, 16)}
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([101], dtype=torch.int64)
        from kvpress_ascend.simulate import ATTN_STATES, FakeAscendMetadata
        meta = {"model.layers.0.self_attn.attn": FakeAscendMetadata(
            1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
            bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))}
        # make the layer buffer allocation fail on purpose
        import kvpress_ascend.capture as cap
        orig_alloc = cap.CaptureManager._layer_buffer
        def boom(self, layer_name, shape, device):
            raise RuntimeError("simulated alloc failure")
        cap.CaptureManager._layer_buffer = boom
        try:
            mgr.on_metadata_built(runner, meta, None)  # must not raise
        finally:
            cap.CaptureManager._layer_buffer = orig_alloc
        m = meta["model.layers.0.self_attn.attn"]
        assert int(m.seq_lens_list[0]) == 101  # restored, not rewritten
