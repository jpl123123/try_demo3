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
