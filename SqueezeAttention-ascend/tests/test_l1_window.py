"""L1: WindowManager capture + clustering + window rewrite against fakes."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvpress_ascend.simulate import (  # noqa: E402
    ATTN_STATES,
    FakeAscendMetadata,
    FakeInputBatch,
    FakeRunner,
    FakeSchedulerOutput,
)
from squeeze_ascend.capture import WindowManager  # noqa: E402


def make_world():
    runner = FakeRunner().build(num_layers=4, kv_heads=2, head_size=8, num_heads=8,
                                block_size=16, max_blocks=256, max_reqs=8)
    ib = FakeInputBatch(["r0"], [0], [260], 16, 256, 8)
    runner.input_batch = ib
    bt = ib.block_table[0]
    m = (260 + 15) // 16
    for b in range(m):
        bt.block_table.np[0, b] = 100 + b
    bt.num_blocks_per_row[0] = m
    return runner, ib, bt


class TestImportanceCapture:
    def test_cos_sim_accumulation(self):
        runner, ib, bt = make_world()
        mgr = WindowManager()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 260}, 0))
        layer = "model.layers.1.self_attn.attn"
        vec1 = torch.randn(100, 32)
        mgr.on_layer_input(layer, vec1, is_draft=False)
        mgr.on_attn_output(layer, vec1, vec1 * 1.0, is_draft=False)  # cos sim = 1
        acc = mgr.importance["r0"][layer]
        assert acc[1] == 100
        assert abs(acc[0] / 100 - 1.0) < 1e-3

    def test_importance_ignores_multi_prefill_steps(self):
        runner, ib, bt = make_world()
        ib2 = FakeInputBatch(["r0", "r1"], [0, 0], [260, 80], 16, 256, 8)
        runner.input_batch = ib2
        mgr = WindowManager()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100, "r1": 80}, 0))
        layer = "model.layers.1.self_attn.attn"
        vec1 = torch.randn(180, 32)
        mgr.on_layer_input(layer, vec1, is_draft=False)
        mgr.on_attn_output(layer, vec1, vec1, is_draft=False)
        assert mgr.importance == {}


class TestClusterPass:
    def test_windows_recorded(self):
        runner, ib, bt = make_world()
        mgr = WindowManager()
        layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 260}, 0))
        T = 260
        for i, ln in enumerate(layer_names):
            vec1 = torch.randn(T, 32)
            mgr.on_layer_input(ln, vec1, is_draft=False)
            mgr.on_attn_output(ln, vec1, vec1 * (0.9 - 0.2 * i) - vec1, is_draft=False)
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 260}, 0))
        w = mgr.windows["r0"]
        assert len(w) == len(layer_names)
        for v in w.values():
            assert 5 <= v <= 260

    def test_skipped_short_prompt(self):
        runner, ib, bt = make_world()
        ib.num_prompt_tokens[0] = 50
        ib.num_computed_tokens_cpu[0] = 0
        mgr = WindowManager()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 50}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 50}, 0))
        assert mgr.windows == {}


class TestWindowRewrite:
    def test_metadata_rewrite_applies_windows(self):
        runner, ib, bt = make_world()
        mgr = WindowManager()
        layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
        mgr.windows["r0"] = {ln: 100 for ln in layer_names}
        # true len 261, window 100, start 4 -> rewrite expected
        bt.num_blocks_per_row[0] = 17  # 261 tokens
        for b in range(16, 17):
            bt.block_table.np[0, b] = 900 + b
        bt.copy_to_gpu(17)
        ib.num_computed_tokens_cpu[0] = 260
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([261], dtype=torch.int64)
        meta = {}
        for ln in layer_names:
            meta[ln] = FakeAscendMetadata(1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
                                          bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))
        mgr.on_metadata_built(runner, meta, None)
        from squeeze_ascend.kvcore import window_view_len, window_block_ranges
        for ln in layer_names:
            m = meta[ln]
            assert int(m.seq_lens_list[0]) == window_view_len(261, *window_block_ranges(261, 100, 4, 16)[:2], 16)
            row = m.block_tables[0].numpy()
            sink, recent_first, last = window_block_ranges(261, 100, 4, 16)
            assert row[0] == 100  # sink block id
            assert row[sink] == bt.block_table.np[0, recent_first]  # recent blocks
            assert int(m.seq_lens[0]) == int(m.seq_lens_list[0])

    def test_no_rewrite_when_window_covers(self):
        runner, ib, bt = make_world()
        mgr = WindowManager()
        layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
        mgr.windows["r0"] = {ln: 500 for ln in layer_names}  # window > true len
        ib.num_computed_tokens_cpu[0] = 260
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([261], dtype=torch.int64)
        meta = {}
        for ln in layer_names:
            meta[ln] = FakeAscendMetadata(1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
                                          bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))
        mgr.on_metadata_built(runner, meta, None)
        for ln in layer_names:
            assert int(meta[ln].seq_lens_list[0]) == 261  # untouched
