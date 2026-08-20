"""L1: CaptureManager against fakes that mirror the vllm-ascend attribute
surface (completion detection, capture buffers, layout recording, metadata
rewrite)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvpress_ascend.capture import CaptureManager  # noqa: E402
from kvpress_ascend.presses import build_press  # noqa: E402
from kvpress_ascend.simulate import (  # noqa: E402
    ATTN_STATES,
    FakeAscendMetadata,
    FakeInputBatch,
    FakeRunner,
    FakeSchedulerOutput,
)


def make_runner(num_layers=4, block_size=16, max_blocks=256):
    return FakeRunner().build(num_layers=num_layers, kv_heads=2, head_size=8,
                              num_heads=8, block_size=block_size, max_blocks=max_blocks,
                              max_reqs=16)


def make_mgr(mode="view", press="knorm"):
    mgr = CaptureManager()
    mgr.mode = mode
    mgr.press = build_press(press, ratio=0.5, window=8, sink=2, kernel=3)
    mgr.capture_w = 64
    return mgr


@pytest.fixture(autouse=True)
def _min_prompt_zero(monkeypatch):
    monkeypatch.setenv("KVPRESS_ASCEND_MIN_PROMPT", "0")


class TestStepBegin:
    def test_completion_detection(self):
        runner = make_runner()
        ib = FakeInputBatch(["r0", "r1"], [0, 0], [260, 80], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        sched = FakeSchedulerOutput({"r0": 200, "r1": 80}, 0)
        mgr.on_step_begin(runner, sched)
        assert mgr.step.num_reqs == 2
        assert mgr.step.completed_prefill == ["r1"]  # r0: 0+200<260; r1: 0+80>=80

    def test_completion_uses_before_plus_scheduled(self):
        """The vLLM v1 timing trap: num_computed updates only in sample_tokens,
        so the check must be before + scheduled."""
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [200], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 60}, 0))
        assert mgr.step.completed_prefill == ["r0"]  # 200+60 >= 260

    def test_layout_dropped_on_recompute(self):
        """Preemption detection is now regression-based (before < last_seen):
        a request may legitimately carry a mid-prefill layout while still
        prefilling, so 'before < prompt' alone is no longer a drop signal."""
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [100], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.layouts["r0"] = {"l0": object()}
        mgr._last_before["r0"] = 200  # num_computed fell back -> preemption
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 50}, 0))
        assert "r0" not in mgr.layouts

    def test_layout_survives_still_prefilling(self):
        """A mid-prefill layout must NOT be dropped while the request keeps
        prefilling normally (monotonic num_computed)."""
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [150], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.layouts["r0"] = {"l0": object()}
        mgr._last_before["r0"] = 100  # progressed 100 -> 150: normal
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        assert "r0" in mgr.layouts


class TestCapture:
    def test_query_capture_rolling(self):
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [0], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        layer = "model.layers.0.self_attn.attn"
        meta = FakeAscendMetadata(100, [100], ATTN_STATES["ChunkedPrefill"],
                                  torch.tensor([100]), torch.zeros(1, 64, dtype=torch.int32),
                                  torch.zeros(100, dtype=torch.int32))
        q = torch.randn(100, 8, 8)
        mgr.on_backend_forward(layer, q, meta, is_draft=False)
        rc = mgr.requests["r0"]
        assert rc.captured_q[layer] == 64  # capped at capture_w
        # second step overwrites with the new chunk
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 100}, 0))
        q2 = torch.randn(100, 8, 8)
        mgr.on_backend_forward(layer, q2, meta, is_draft=False)
        assert torch.allclose(rc.queries[layer][:64], q2[-64:])

    def test_no_capture_for_decode(self):
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [260], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        layer = "model.layers.0.self_attn.attn"
        meta = FakeAscendMetadata(1, [1], ATTN_STATES["DecodeOnly"],
                                  torch.tensor([261]), torch.zeros(1, 64, dtype=torch.int32),
                                  torch.zeros(1, dtype=torch.int32))
        mgr.on_backend_forward(layer, torch.randn(1, 8, 8), meta, is_draft=False)
        assert "r0" not in mgr.requests


class TestViewCompressionPass:
    def test_view_layout_recorded(self):
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [0], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr(press="streamingllm")
        # grow the row (scheduler mirror)
        bt = ib.block_table[0]
        m = (260 + 15) // 16
        for b in range(m):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = m
        # capture queries for the last chunk
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 260}, 0))
        layer = "model.layers.0.self_attn.attn"
        meta = FakeAscendMetadata(260, [260], ATTN_STATES["PrefillNoCache"],
                                  torch.tensor([260]), torch.zeros(1, 64, dtype=torch.int32),
                                  torch.zeros(260, dtype=torch.int32))
        mgr.on_backend_forward(layer, torch.randn(260, 8, 8), meta, is_draft=False)
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 260}, 0))
        layouts = mgr.layouts["r0"]
        assert layer in layouts
        lay = layouts[layer]
        assert lay.orig_len == 260
        # partial last block (260 % 16 = 4) must be forced into the view
        # (physical id: fake row uses 100 + block_index)
        assert (100 + m - 1) in lay.kept_blocks
        assert lay.n_kept_blocks <= m

    def test_skipped_short_prompt(self):
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [0], [50], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 50}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 50}, 0))
        assert "r0" not in mgr.layouts

    def test_dry_run_records_nothing(self, monkeypatch):
        import kvpress_ascend.envs as envs
        monkeypatch.setenv("KVPRESS_ASCEND_DRY_RUN", "1")
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [0], [260], 16, 256, 16)
        runner.input_batch = ib
        bt = ib.block_table[0]
        m = (260 + 15) // 16
        for b in range(m):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = m
        mgr = make_mgr(press="streamingllm")
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 260}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 260}, 0))
        assert "r0" not in mgr.layouts


class TestCompletionCatchUp:
    def test_catch_up_missed_completion(self):
        """The final prefill chunk can cross the boundary without the strict
        check firing (MTP async scheduling). The next step must catch up:
        last_before < prompt <= before -> compress now."""
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [100], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        # step A: 100 + 50 < 260 -> not completed
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 50}, 0))
        assert mgr.step.completed_prefill == []
        # sample_tokens moves num_computed past the prompt (e.g. 300)
        ib.num_computed_tokens_cpu[0] = 300
        # step B: before=300 >= prompt, prev=100 < prompt -> catch-up fires
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 0}, 0))
        assert mgr.step.completed_prefill == ["r0"]
        # step C: no re-trigger on later decode steps
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        assert mgr.step.completed_prefill == []

    def test_no_catch_up_for_normal_completion(self):
        """The strict path handles the normal completion step; the catch-up
        must not fire on it (before < prompt there)."""
        runner = make_runner()
        ib = FakeInputBatch(["r0"], [200], [260], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr()
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 60}, 0))
        assert mgr.step.completed_prefill == ["r0"]
        ib.num_computed_tokens_cpu[0] = 260
        mgr._compressed_done.add("r0")
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        assert mgr.step.completed_prefill == []


class TestDeviceSafeNumpy:
    def test_as_numpy(self):
        from kvpress_ascend.capture import _as_numpy
        t = torch.tensor([1, 2, 3])
        out = _as_numpy(t)
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [1, 2, 3]
        assert _as_numpy(np.array([4, 5])).tolist() == [4, 5]


class TestMetadataRewrite:
    def test_view_metadata_rewrite(self):
        runner = make_runner()
        ib = FakeInputBatch(["r0", "r1"], [260, 80], [260, 80], 16, 256, 16)
        runner.input_batch = ib
        mgr = make_mgr(press="streamingllm")
        # compacted layout for r0
        bt = ib.block_table[0]
        m0 = (260 + 15) // 16
        for b in range(m0):
            bt.block_table.np[0, b] = 100 + b
        bt.num_blocks_per_row[0] = m0 + 2  # 2 decode blocks appended
        for b in range(m0, m0 + 2):
            bt.block_table.np[0, b] = 900 + b
        from kvpress_ascend.kvcore import ViewLayout
        kept = [100, 102, 105]  # physical ids of blocks 0, 2, 5
        if (100 + m0 - 1) not in kept:
            kept.append(100 + m0 - 1)  # force the partial last block
        n_kept = 16 * 3 + (260 - 16 * 16)
        mgr.layouts["r0"] = {
            "model.layers.0.self_attn.attn": ViewLayout.build(
                "r0", "model.layers.0.self_attn.attn", 260, kept, n_kept, 16),
        }
        bt.copy_to_gpu(3)
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1, "r1": 1}, 0))
        seq_lens = torch.tensor([261, 81], dtype=torch.int64)
        meta = {}
        for ln in runner.kv_cache_config.kv_cache_groups[0].layer_names:
            meta[ln] = FakeAscendMetadata(2, [1, 1], ATTN_STATES["DecodeOnly"], seq_lens,
                                          bt.get_device_tensor()[:2], torch.zeros(2, dtype=torch.int32))
        mgr.on_metadata_built(runner, meta, None)
        m = meta["model.layers.0.self_attn.attn"]
        # seq lens of r0 = n_kept + 1
        lay = mgr.layouts["r0"]["model.layers.0.self_attn.attn"]
        assert int(m.seq_lens[0]) == lay.view_seq_len(261)
        assert int(m.seq_lens[1]) == 81  # r1 untouched
        assert m.seq_lens_list[0] == int(m.seq_lens[0])
        # block row of r0 starts with kept blocks
        row0 = m.block_tables[0].numpy()
        assert row0[0] == 100  # block id of kept block 0
        # r1's row untouched
        row1 = m.block_tables[1].numpy()
        assert row1[0] == bt.block_table.np[1, 0]
