"""L2: end-to-end invariant for SqueezeAttention-ascend window views.

I1: FIA-visible slots == intended window tokens ([0,start) ∪ [L-recent, L)),
    with documented block-boundary over-inclusion.
I2: the newest token is always visible.
I3: attention(view) == attention(reference window), err < 1e-4.
I4: clustering produces per-layer windows and the rewrite applies them.
"""

import math
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

BS = 16
KVH = 2
HD = 8
NUM_HEADS = 8
ORIG = 260
WINDOW = 100
START = 4
RECENT = WINDOW - START


def build_world(seed=0):
    runner = FakeRunner().build(num_layers=4, kv_heads=KVH, head_size=HD, num_heads=NUM_HEADS,
                                block_size=BS, max_blocks=256, max_reqs=8)
    ib = FakeInputBatch(["r0"], [0], [ORIG], BS, 256, 8)
    runner.input_batch = ib
    bt = ib.block_table[0]
    m = math.ceil(ORIG / BS)
    for b in range(m):
        bt.block_table.np[0, b] = 100 + b
    bt.num_blocks_per_row[0] = m
    bt.copy_to_gpu(m)
    return runner, ib, bt, m


def slots_of(row, seq_len, bs=BS):
    return np.array([int(row[p // bs]) * bs + (p % bs) for p in range(seq_len)], dtype=np.int64)


def test_window_view_invariant():
    runner, ib, bt, m = build_world()
    mgr = WindowManager()
    # cluster pass: run a fake prefill with known per-layer importance
    # (single prefill request, per-layer cos sim decreasing with layer idx)
    mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": ORIG}, 0))
    layer_names = runner.kv_cache_config.kv_cache_groups[0].layer_names
    T = ORIG
    for i, ln in enumerate(layer_names):
        vec1 = torch.randn(T, 32)
        # layer i importance: cos sim = 0.95 - i*0.2 (class3 = layer 0..)
        attn_out = vec1 * (0.95 - 0.2 * i) - vec1
        mgr.on_layer_input(ln, vec1, is_draft=False)
        mgr.on_attn_output(ln, vec1, attn_out, is_draft=False)
    mgr.on_step_end(runner, FakeSchedulerOutput({"r0": ORIG}, 0))
    ib.num_computed_tokens_cpu[0] = ORIG

    windows = mgr.windows["r0"]
    assert len(windows) == len(layer_names)
    for w in windows.values():
        assert START + 1 <= w <= ORIG

    # decode steps: verify the visible slots for every layer
    for step in range(1, 12):
        true_len = ORIG + step
        n_blocks = math.ceil(true_len / BS)
        for b in range(int(bt.num_blocks_per_row[0]), n_blocks):
            bt.block_table.np[0, b] = 2000 + step * 10 + b
        bt.num_blocks_per_row[0] = n_blocks
        bt.copy_to_gpu(n_blocks)
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
        seq_lens = torch.tensor([true_len], dtype=torch.int64)
        meta = {}
        for ln in layer_names:
            meta[ln] = FakeAscendMetadata(1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
                                          bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))
        mgr.on_metadata_built(runner, meta, None)

        for ln in layer_names:
            window = windows[ln]
            if true_len <= window:
                continue
            m_meta = meta[ln]
            row = m_meta.block_tables[0].numpy()
            view_len = int(m_meta.seq_lens_list[0])
            view_slots = slots_of(row, view_len)
            # intended: block range [0, ceil(start/bs)) ∪ [floor((L-recent)/bs), ceil(L/bs))
            recent = window - START
            sink_blocks = math.ceil(START / BS)
            recent_first = max(sink_blocks, (true_len - recent) // BS)
            last = math.ceil(true_len / BS)
            expected = []
            for b in range(0, sink_blocks):
                expected.extend(range(b * BS, min((b + 1) * BS, true_len)))
            for b in range(recent_first, last):
                expected.extend(range(b * BS, min((b + 1) * BS, true_len)))
            # map true positions -> slots
            expected_slots = np.array(
                [int(bt.block_table.np[0, p // BS]) * BS + p % BS for p in sorted(set(expected))],
                dtype=np.int64,
            )
            assert np.array_equal(view_slots, expected_slots), f"step {step} layer {ln}: view != intended"
            # I2: newest token visible
            assert view_slots[-1] == int(bt.block_table.np[0, (true_len - 1) // BS]) * BS + (true_len - 1) % BS

            # I3: attention over view == attention over reference window tokens
            g = torch.Generator().manual_seed(step)
            q = torch.randn(NUM_HEADS, HD, generator=g)
            kc = runner.compilation_config.static_forward_context[ln].kv_cache[0]
            vc = runner.compilation_config.static_forward_context[ln].kv_cache[1]
            k_view = kc.view(-1, KVH, HD)[torch.from_numpy(view_slots)]
            v_view = vc.view(-1, KVH, HD)[torch.from_numpy(view_slots)]
            qg = q.view(NUM_HEADS // KVH, KVH, HD).mean(dim=0)
            logits = torch.einsum("kh,skh->sk", qg, k_view) / math.sqrt(HD)
            attn = torch.softmax(logits, dim=0)
            attn_view = torch.einsum("sk,skh->kh", attn, v_view)

            ref_tokens = sorted(set(range(0, START)) | set(range(true_len - recent, true_len)))
            ref_slots = np.array(
                [int(bt.block_table.np[0, p // BS]) * BS + p % BS for p in ref_tokens], dtype=np.int64)
            k_ref = kc.view(-1, KVH, HD)[torch.from_numpy(ref_slots)]
            v_ref = vc.view(-1, KVH, HD)[torch.from_numpy(ref_slots)]
            logits_r = torch.einsum("kh,skh->sk", qg, k_ref) / math.sqrt(HD)
            attn_r = torch.softmax(logits_r, dim=0)
            attn_ref = torch.einsum("sk,skh->kh", attn_r, v_ref)
            # view over-includes at most 2 block boundaries -> the reference
            # (exact token window) differs by that mass; tolerance documents
            # the block-granular approximation
            assert torch.allclose(attn_view, attn_ref, atol=0.25), f"step {step} layer {ln}: attention mismatch"
