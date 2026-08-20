"""L2: end-to-end invariants with the CaptureManager driving fakes in the
vLLM v1 step order. Multi-step decode across block boundaries.

Invariants (one-vote-veto):
  I1 view mode:  FIA-visible slots == (kept blocks' tokens ∪ new tokens)
  I2 view mode:  the newest token is always visible (never dropped by the view)
  I3 view mode:  attention(view) == attention(reference set), err < 1e-4
  I4 compact:    tail slots hold exactly the kept reference KV after packing
  I5 compact:    new tokens land at compressed slots (n_kept + j)
  I6 compact:    attention(compacted view) == attention(kept + new), err < 1e-4
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
KVH = 2
HD = 8
NUM_HEADS = 8
ORIG = 100  # prompt tokens (block 6 partial: 100 - 96 = 4 tokens)


@pytest.fixture(autouse=True)
def _min_prompt_zero(monkeypatch):
    monkeypatch.setenv("KVPRESS_ASCEND_MIN_PROMPT", "0")


def build_world(seed=0):
    runner = FakeRunner().build(num_layers=2, kv_heads=KVH, head_size=HD, num_heads=NUM_HEADS,
                                block_size=BS, max_blocks=256, max_reqs=8)
    ib = FakeInputBatch(["r0"], [0], [ORIG], BS, 256, 8)
    runner.input_batch = ib
    bt = ib.block_table[0]
    m = math.ceil(ORIG / BS)
    for b in range(m):
        bt.block_table.np[0, b] = 100 + b
    bt.num_blocks_per_row[0] = m
    bt.copy_to_gpu(m)
    # prefill KV snapshot (reference ground truth) written at true slots
    g = torch.Generator().manual_seed(seed)
    pref_k = torch.randn(ORIG, KVH, HD, generator=g) * 0.3
    pref_v = torch.randn(ORIG, KVH, HD, generator=g) * 0.3
    for ln, mod in runner.compilation_config.static_forward_context.items():
        kc, vc = mod.kv_cache
        slots = torch.tensor([(100 + b) * BS + off for b in range(m) for off in range(BS)], dtype=torch.int64)
        kc.view(-1, KVH, HD)[slots[:ORIG]] = pref_k
        vc.view(-1, KVH, HD)[slots[:ORIG]] = pref_v
    return runner, ib, bt, m, pref_k, pref_v


def slots_of(row, seq_len, bs=BS):
    """FIA semantics: slot = row[p//bs]*bs + p%bs for p in [0, seq_len)."""
    out = []
    for p in range(seq_len):
        out.append(int(row[p // bs]) * bs + (p % bs))
    return np.array(out, dtype=np.int64)


def attention_at(slots, key_cache, value_cache, q, scale):
    k = key_cache.view(-1, KVH, HD)[torch.from_numpy(slots)]
    v = value_cache.view(-1, KVH, HD)[torch.from_numpy(slots)]
    # q: (NUM_HEADS, HD); average over kv heads via GQA repeat
    qg = q.view(NUM_HEADS // KVH, KVH, HD).mean(dim=0)  # (KVH, HD)
    logits = torch.einsum("kh,skh->sk", qg, k) / math.sqrt(HD)  # (S, KVH)
    attn = torch.softmax(logits, dim=0)
    return torch.einsum("sk,skh->kh", attn, v)


class TestViewModeInvariants:
    def test_view_slots_equal_intended_set(self):
        runner, ib, bt, m, pref_k, pref_v = build_world()
        mgr = CaptureManager()
        mgr.mode = "view"
        mgr.press = build_press("streamingllm", ratio=0.5, window=8, sink=2, kernel=3)
        mgr.capture_w = 64

        # prefill completion step
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": ORIG}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": ORIG}, 0))
        ib.num_computed_tokens_cpu[0] = ORIG  # sample_tokens updates it (vLLM v1 timing)
        lay = mgr.layouts["r0"]["model.layers.0.self_attn.attn"]
        kept = {int(b) - 100 for b in lay.kept_blocks}  # physical -> index (row ids are 100+idx)
        assert (m - 1) in kept  # partial last block forced

        # multiple decode steps crossing block boundaries
        for step in range(1, 12):
            true_len = ORIG + step
            n_blocks = math.ceil(true_len / BS)
            for b in range(int(bt.num_blocks_per_row[0]), n_blocks):
                bt.block_table.np[0, b] = 1000 + step * 10 + b
            bt.num_blocks_per_row[0] = n_blocks
            bt.copy_to_gpu(n_blocks)
            # write the new token's KV at its true slot
            g = torch.Generator().manual_seed(step)
            new_k = torch.randn(1, KVH, HD, generator=g) * 0.3
            new_v = torch.randn(1, KVH, HD, generator=g) * 0.3
            true_pos = true_len - 1
            slot = int(bt.block_table.np[0, true_pos // BS]) * BS + true_pos % BS
            for ln, mod in runner.compilation_config.static_forward_context.items():
                mod.kv_cache[0].view(-1, KVH, HD)[slot] = new_k
                mod.kv_cache[1].view(-1, KVH, HD)[slot] = new_v
            # build metadata + rewrite
            mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
            seq_lens = torch.tensor([true_len], dtype=torch.int64)
            meta = {}
            for ln in runner.kv_cache_config.kv_cache_groups[0].layer_names:
                meta[ln] = FakeAscendMetadata(1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
                                              bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))
            mgr.on_metadata_built(runner, meta, None)
            m_meta = meta["model.layers.0.self_attn.attn"]
            row = m_meta.block_tables[0].numpy()
            view_len = int(m_meta.seq_lens_list[0])
            view_slots = slots_of(row, view_len)

            # intended set: kept blocks' tokens (block-granular, partial-aware) + new tokens
            expected_pos = []
            for b in sorted(kept):
                first = b * BS
                last = min(first + BS, ORIG)
                expected_pos.extend(range(first, last))
            expected_pos.extend(range(ORIG, ORIG + step))
            expected = np.array(
                [int(bt.block_table.np[0, p // BS]) * BS + p % BS for p in expected_pos],
                dtype=np.int64,
            )

            assert np.array_equal(view_slots, expected), f"step {step}: view != intended"
            newest_slot = int(bt.block_table.np[0, (true_len - 1) // BS]) * BS + (true_len - 1) % BS
            assert newest_slot in view_slots.tolist(), "new token must be visible"
            # I3: attention over view == attention over reference gather
            q = torch.randn(NUM_HEADS, HD, generator=torch.Generator().manual_seed(step + 100))
            kc = runner.compilation_config.static_forward_context[
                "model.layers.0.self_attn.attn"].kv_cache[0]
            vc = runner.compilation_config.static_forward_context[
                "model.layers.0.self_attn.attn"].kv_cache[1]
            attn_view = attention_at(view_slots, kc, vc, q, 1.0)
            # reference: same slots but gathered through the true cache (identical)
            attn_ref = attention_at(expected, kc, vc, q, 1.0)
            assert torch.allclose(attn_view, attn_ref, atol=1e-5)


class TestCompactModeInvariants:
    def test_compact_packing_and_decode(self):
        runner, ib, bt, m, pref_k, pref_v = build_world()
        mgr = CaptureManager()
        mgr.mode = "compact"
        mgr.press = build_press("streamingllm", ratio=0.5, window=8, sink=2, kernel=3)
        mgr.capture_w = 64

        # prefill completion step -> compaction pass
        mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": ORIG}, 0))
        mgr.on_step_end(runner, FakeSchedulerOutput({"r0": ORIG}, 0))
        ib.num_computed_tokens_cpu[0] = ORIG  # sample_tokens updates it
        lay = mgr.compact["r0"]
        n_kept, delta, k = lay.n_kept, lay.delta, lay.k
        assert lay.check_slack()

        # I4: tail slots hold exactly the kept reference KV
        # streamingllm kept set: [0, n_sink) + [n_sink + n_pruned, ORIG)
        n_pruned = ORIG - n_kept
        n_sink = 2
        keep = sorted(set(range(0, min(n_sink, ORIG)))
                      | set(range(min(n_sink + n_pruned, ORIG), ORIG)))
        keep = keep[:n_kept]
        assert len(keep) == n_kept
        rew = lay.rewrite_row(bt.block_table.np[0, :m])
        for ln, mod in runner.compilation_config.static_forward_context.items():
            kc, vc = mod.kv_cache
            for j, pos in enumerate(keep):
                slot = int(rew[j // BS]) * BS + (j % BS)
                assert torch.allclose(kc.view(-1, KVH, HD)[slot], pref_k[pos], atol=1e-6), \
                    f"key mismatch at kept pos {pos} (slot {slot})"
                assert torch.allclose(vc.view(-1, KVH, HD)[slot], pref_v[pos], atol=1e-6)

        # decode steps: row rewrite + shifted positions + slot mapping
        prev_n_blocks = m
        all_new_k: list = []
        all_new_v: list = []
        for step in range(1, 8):
            true_len = ORIG + step
            n_blocks = math.ceil(true_len / BS)
            # scheduler appends only the NEW true blocks, at the permuted row's
            # end (the row count was reduced by the rewrite)
            count = int(bt.num_blocks_per_row[0])
            for b in range(count, count + (n_blocks - prev_n_blocks)):
                bt.block_table.np[0, b] = 2000 + step * 10 + b
            bt.num_blocks_per_row[0] = count + (n_blocks - prev_n_blocks)
            prev_n_blocks = n_blocks
            mgr.on_step_begin(runner, FakeSchedulerOutput({"r0": 1}, 0))
            mgr.on_prepare_inputs_entry(runner)  # row rewrite (S3, once)
            # positions (true) then shifted by delta (S7 simulation)
            from kvpress_ascend import engine
            engine.ctx = mgr  # _shift_positions reads the engine-global ctx
            positions = torch.tensor([true_len - 1], dtype=torch.int64)
            _shift_positions = engine._shift_positions
            _shift_positions(1, torch.tensor([0, 1], dtype=torch.int64), positions)
            assert int(positions[0]) == true_len - 1 - delta
            # slot mapping with the rewritten row
            ib.block_table.compute_slot_mapping(1, torch.tensor([0, 1], dtype=torch.int64), positions)
            slot = int(bt.slot_mapping.np[0])
            # I5: the new token lands at compressed position n_kept + step - 1
            comp_pos = n_kept + step - 1
            view_row = bt.block_table.np[0, :int(bt.num_blocks_per_row[0])]  # stored permuted row
            exp_slot = int(view_row[comp_pos // BS]) * BS + comp_pos % BS
            assert slot == exp_slot, f"step {step}: slot {slot} != {exp_slot}"

            # write the new token KV at the compressed slot (as reshape_and_cache would)
            g = torch.Generator().manual_seed(step)
            new_k = torch.randn(1, KVH, HD, generator=g) * 0.3
            new_v = torch.randn(1, KVH, HD, generator=g) * 0.3
            all_new_k.append(new_k)
            all_new_v.append(new_v)
            for ln, mod in runner.compilation_config.static_forward_context.items():
                mod.kv_cache[0].view(-1, KVH, HD)[slot] = new_k
                mod.kv_cache[1].view(-1, KVH, HD)[slot] = new_v
            # metadata rewrite: seq_lens -= delta
            seq_lens = torch.tensor([true_len], dtype=torch.int64)
            meta = {}
            for ln in runner.kv_cache_config.kv_cache_groups[0].layer_names:
                meta[ln] = FakeAscendMetadata(1, [1], ATTN_STATES["DecodeOnly"], seq_lens,
                                              bt.get_device_tensor()[:1], torch.zeros(1, dtype=torch.int32))
            mgr.on_metadata_built(runner, meta, None)
            m_meta = meta["model.layers.0.self_attn.attn"]
            assert int(m_meta.seq_lens_list[0]) == n_kept + step
            # I6: attention over the compacted view == reference (kept + new)
            view_row = bt.block_table.np[0, :int(bt.num_blocks_per_row[0])]
            view_len = n_kept + step
            view_slots = slots_of(view_row, view_len)
            q = torch.randn(NUM_HEADS, HD, generator=torch.Generator().manual_seed(step + 50))
            kc = runner.compilation_config.static_forward_context[
                "model.layers.0.self_attn.attn"].kv_cache[0]
            vc = runner.compilation_config.static_forward_context[
                "model.layers.0.self_attn.attn"].kv_cache[1]
            attn_view = attention_at(view_slots, kc, vc, q, 1.0)
            # reference: kept KV (original prefill snapshot) + ALL new tokens so far
            ref_k = torch.cat([pref_k[keep]] + all_new_k, dim=0)
            ref_v = torch.cat([pref_v[keep]] + all_new_v, dim=0)
            qg = q.view(NUM_HEADS // KVH, KVH, HD).mean(dim=0)
            logits = torch.einsum("kh,skh->sk", qg, ref_k) / math.sqrt(HD)
            attn = torch.softmax(logits, dim=0)
            attn_ref = torch.einsum("sk,skh->kh", attn, ref_v)
            assert torch.allclose(attn_view, attn_ref, atol=1e-4), f"step {step}: attention mismatch"
