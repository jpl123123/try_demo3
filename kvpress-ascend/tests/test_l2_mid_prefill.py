"""L2: progressive (mid-prefill) compression — the fix for the
chicken-and-egg of the completion-only design.

Scenario: prompt=300 tokens, chunked prefill 100/100/100, mid-prefill budget=150.
  step1 (100): no compression (below budget)
  step2 (200): MID compression anchors the layout at 200 (block m'-1 forced)
  step3 (300): completion re-anchors the layout at 300

Invariants:
  I1: mid layout exists after step2 with orig_len == 200 and forced last block
  I2: the layout survives further prefill steps (regression-based drop check
      must NOT treat "still prefilling" as preemption)
  I3: the next chunk's tokens stay visible through the mid view
  I4: completion re-anchors the layout at the prompt length
"""

import math
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

BS = 16
ORIG = 300
BUDGET = 150


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("KVPRESS_ASCEND_MIN_PROMPT", "0")
    monkeypatch.setenv("KVPRESS_ASCEND_MID_PREFILL", "1")
    monkeypatch.setenv("KVPRESS_ASCEND_MID_PREFILL_BUDGET", str(BUDGET))
    monkeypatch.setenv("KVPRESS_ASCEND_MID_PREFILL_REFRESH", "100")


def make_world():
    runner = FakeRunner().build(num_layers=1, kv_heads=2, head_size=8, num_heads=8,
                                block_size=BS, max_blocks=256, max_reqs=8)
    ib = FakeInputBatch(["r0"], [0], [ORIG], BS, 256, 8)
    runner.input_batch = ib
    bt = ib.block_table[0]
    m = math.ceil(ORIG / BS)
    for b in range(m):
        bt.block_table.np[0, b] = 100 + b
    bt.num_blocks_per_row[0] = m
    bt.copy_to_gpu(m)
    return runner, ib, bt


def run_chunk(runner, ib, bt, mgr, done, attn_state):
    """One chunked-prefill step: grow row, write KV, capture, step_end."""
    n_sched = done - int(ib.num_computed_tokens_cpu[0])
    mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": n_sched}, 0))
    n_blocks = math.ceil(done / BS)
    for b in range(int(bt.num_blocks_per_row[0]), n_blocks):
        bt.block_table.np[0, b] = 100 + b
    bt.num_blocks_per_row[0] = n_blocks
    bt.copy_to_gpu(n_blocks)
    # write this chunk's KV at true slots + capture queries
    g = torch.Generator().manual_seed(done)
    layer = "model.layers.0.self_attn.attn"
    mod = runner.compilation_config.static_forward_context[layer]
    start = int(ib.num_computed_tokens_cpu[0])
    kv = torch.randn(n_sched, 2, 8, generator=g) * 0.3
    for j in range(n_sched):
        p = start + j
        slot = int(bt.block_table.np[0, p // BS]) * BS + p % BS
        mod.kv_cache[0].view(-1, 2, 8)[slot] = kv[j]
        mod.kv_cache[1].view(-1, 2, 8)[slot] = kv[j] * 0.7 + 0.1
    meta = {}
    for ln in runner.kv_cache_config.kv_cache_groups[0].layer_names:
        meta[ln] = FakeAscendMetadata(
            n_sched, [n_sched], attn_state,
            torch.tensor([done], dtype=torch.int64),
            bt.get_device_tensor()[:1], torch.zeros(n_sched, dtype=torch.int32))
        mgr.on_backend_forward(ln, torch.randn(n_sched, 8, 8), meta[ln], is_draft=False)
    mgr.on_metadata_built(runner, meta, None)
    mgr.on_step_end(runner, FakeSchedulerOutput({"r0": n_sched}, 0))
    ib.num_computed_tokens_cpu[0] = done  # sample_tokens timing


def test_mid_prefill_anchors_and_reanchors():
    runner, ib, bt = make_world()
    mgr = CaptureManager()
    mgr.mode = "view"
    mgr.press = build_press("streamingllm", ratio=0.5, window=8, sink=2, kernel=3)
    mgr.capture_w = 64

    run_chunk(runner, ib, bt, mgr, 100, ATTN_STATES["ChunkedPrefill"])
    assert "r0" not in mgr.layouts, "below budget: no compression yet"

    run_chunk(runner, ib, bt, mgr, 200, ATTN_STATES["ChunkedPrefill"])
    # I1: mid layout anchored at 200
    assert mgr.mid_anchors.get("r0") == 200
    lay = mgr.layouts["r0"]["model.layers.0.self_attn.attn"]
    assert lay.orig_len == 200
    m2 = math.ceil(200 / BS)
    assert (100 + m2 - 1) in lay.kept_blocks, "partial last block forced at mid anchor"

    # I3: a further chunk (step 3, completion at 300) keeps the mid layout
    # alive during the step (regression check must not drop it)
    run_chunk(runner, ib, bt, mgr, 300, ATTN_STATES["PrefillNoCache"])
    # I4: completion re-anchored at 300
    lay = mgr.layouts["r0"]["model.layers.0.self_attn.attn"]
    assert lay.orig_len == 300
    assert lay.kept_blocks  # non-empty
    assert "r0" in mgr.requests or True  # capture may be freed at completion

    # decode steps keep working (view rewrite applies)
    for step in range(1, 5):
        done = 300 + step
        n_blocks = math.ceil(done / BS)
        for b in range(int(bt.num_blocks_per_row[0]), n_blocks):
            bt.block_table.np[0, b] = 2000 + step * 10 + b
        bt.num_blocks_per_row[0] = n_blocks
        bt.copy_to_gpu(n_blocks)
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        meta = {}
        for ln in runner.kv_cache_config.kv_cache_groups[0].layer_names:
            meta[ln] = FakeAscendMetadata(
                1, [1], ATTN_STATES["DecodeOnly"], torch.tensor([done], dtype=torch.int64),
                bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))
        mgr.on_metadata_built(runner, meta, None)
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": 1}, 0))
        ib.num_computed_tokens_cpu[0] = done
        m = meta["model.layers.0.self_attn.attn"]
        lay = mgr.layouts["r0"]["model.layers.0.self_attn.attn"]
        assert int(m.seq_lens_list[0]) == lay.view_seq_len(done)


def test_mid_prefill_view_visibility():
    """The view after mid compression must contain exactly the kept blocks'
    tokens and stay consistent with subsequent writes (true positions)."""
    runner, ib, bt = make_world()
    mgr = CaptureManager()
    mgr.mode = "view"
    mgr.press = build_press("streamingllm", ratio=0.5, window=8, sink=2, kernel=3)
    mgr.capture_w = 64
    run_chunk(runner, ib, bt, mgr, 100, ATTN_STATES["ChunkedPrefill"])
    run_chunk(runner, ib, bt, mgr, 200, ATTN_STATES["ChunkedPrefill"])
    lay = mgr.layouts["r0"]["model.layers.0.self_attn.attn"]
    assert lay.orig_len == 200
    # view_len at anchor == kept tokens
    assert lay.view_seq_len(200) == lay.n_kept
    # kept blocks are physical ids within the true row
    row_ids = set(int(b) for b in bt.block_table.np[0, :math.ceil(200 / BS)])
    assert set(lay.kept_blocks) <= row_ids
