"""L1/L2 offline simulator for kvpress-ascend.

No NPU, no vllm needed: the fakes below mirror the exact attribute surface of
the vllm-ascend v0.23.0 objects the patches touch (verified against source),
and the step driver reproduces the vLLM v1 timing order (see PLAN.md §2.3).

Usage:
    python -m kvpress_ascend.simulate            # run the built-in scenario
    python -m kvpress_ascend.simulate --steps 8  # more decode steps
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

# torch is required for the device-side pass; the sim runs on CPU tensors.
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes (attribute surface copied from vllm-ascend v0.23.0 source)
# ---------------------------------------------------------------------------


class FakeEnum:
    """Mimics AscendAttentionState Enum member: has .name and .value(int)."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __str__(self):
        return self.name


ATTN_STATES = {
    "PrefillNoCache": FakeEnum("PrefillNoCache", 0),
    "PrefillCacheHit": FakeEnum("PrefillCacheHit", 1),
    "DecodeOnly": FakeEnum("DecodeOnly", 2),
    "ChunkedPrefill": FakeEnum("ChunkedPrefill", 3),
    "SpecDecoding": FakeEnum("SpecDecoding", 4),
}


class FakeBlockTable:
    """Mirrors BlockTable: .block_table (CpuGpuBuffer-like), .num_blocks_per_row."""

    def __init__(self, block_size: int, max_blocks: int, num_reqs: int):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.num_reqs = num_reqs
        self.block_table = _FakeBuffer((num_reqs, max_blocks), dtype=torch.int32)
        self.num_blocks_per_row = np.zeros(num_reqs, dtype=np.int32)
        self.slot_mapping = _FakeBuffer((8192,), dtype=torch.int32)

    def get_device_tensor(self):
        return self.block_table.gpu

    def copy_to_gpu(self, n=None):
        self.block_table.gpu.copy_(torch.from_numpy(self.block_table.np))


class _FakeBuffer:
    """Mirrors CpuGpuBuffer: .np (numpy view), .gpu (torch tensor)."""

    def __init__(self, shape, dtype=torch.int64):
        self.np = np.zeros(shape, dtype=_np_dtype(dtype))
        self.gpu = torch.from_numpy(self.np).to(dtype=dtype)

    def copy_to_gpu(self, n=None):
        self.gpu.copy_(torch.from_numpy(self.np))


def _np_dtype(dtype):
    if dtype == torch.int32:
        return np.int32
    if dtype == torch.int64:
        return np.int64
    return np.float32


class FakeMultiGroupBlockTable:
    """Mirrors MultiGroupBlockTable: __getitem__(gid) -> FakeBlockTable."""

    def __init__(self, block_size: int, max_blocks: int, num_reqs: int):
        self.block_tables = [FakeBlockTable(block_size, max_blocks, num_reqs)]

    def __getitem__(self, idx):
        return self.block_tables[idx]

    def commit_block_table(self, num_reqs):
        for bt in self.block_tables:
            bt.copy_to_gpu(num_reqs)

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions, *a, **k):
        # kernel semantics: slot = row[pos//bs]*bs + pos%bs
        bt = self.block_tables[0]
        rows = bt.block_table.np
        bs = bt.block_size
        pos = positions.numpy() if torch.is_tensor(positions) else np.asarray(positions)
        qsl = query_start_loc.numpy() if torch.is_tensor(query_start_loc) else np.asarray(query_start_loc)
        slots = np.zeros(pos.shape[0], dtype=np.int64)
        for r in range(num_reqs):
            lo, hi = int(qsl[r]), int(qsl[r + 1])
            for t in range(lo, hi):
                p = int(pos[t])
                slots[t] = rows[r, p // bs] * bs + (p % bs)
        bt.slot_mapping.np[: pos.shape[0]] = slots
        bt.slot_mapping.copy_to_gpu(pos.shape[0])


class FakeInputBatch:
    def __init__(self, req_ids, num_computed, num_prompt, block_size, max_blocks, max_reqs=16):
        self.req_ids = list(req_ids)
        self.req_id_to_index = {r: i for i, r in enumerate(self.req_ids)}
        self.num_computed_tokens_cpu = np.array(num_computed, dtype=np.int32)
        self.num_prompt_tokens = np.array(num_prompt, dtype=np.int32)
        self.num_computed_tokens_cpu_tensor = torch.from_numpy(self.num_computed_tokens_cpu)
        self.num_prompt_tokens_cpu_tensor = torch.from_numpy(self.num_prompt_tokens)
        self.block_table = FakeMultiGroupBlockTable(block_size, max_blocks, max_reqs)


class FakeKVSpec:
    def __init__(self, block_size, num_kv_heads, head_size):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size


class FakeKVGroup:
    def __init__(self, layer_names, spec):
        self.layer_names = layer_names
        self.kv_cache_spec = spec


class FakeKVCacheConfig:
    def __init__(self, groups):
        self.kv_cache_groups = groups


class FakeAttentionModule:
    """Mirrors the Attention modules in static_forward_context."""

    def __init__(self, layer_name, num_blocks, block_size, kv_heads, head_size, num_heads, seed=0):
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        g = torch.Generator().manual_seed(seed + hash(layer_name) % 1000)
        k = torch.randn(num_blocks, block_size, kv_heads, head_size, generator=g) * 0.1
        v = torch.randn(num_blocks, block_size, kv_heads, head_size, generator=g) * 0.1
        self.kv_cache = (k, v)
        # vllm Attention module surface used by presses
        self.num_kv_heads = kv_heads
        self.q_proj = None  # ExpectedAttention falls back to knorm


class FakeCompilationConfig:
    def __init__(self, static_forward_context):
        self.static_forward_context = static_forward_context


class FakeVllmConfig:
    def __init__(self, sfc):
        self.compilation_config = FakeCompilationConfig(sfc)
        self.model_config = FakeModelConfig()


class FakeModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(num_hidden_layers=4)


class SimpleNamespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSchedulerOutput:
    def __init__(self, num_scheduled_tokens: dict, total: int):
        self.num_scheduled_tokens = num_scheduled_tokens
        self.total_num_scheduled_tokens = total


class FakeAscendMetadata:
    """Mirrors AscendMetadata fields the patches touch."""

    def __init__(self, num_actual_tokens, actual_seq_lengths_q, attn_state, seq_lens,
                 block_tables, slot_mapping):
        self.num_actual_tokens = num_actual_tokens
        self.actual_seq_lengths_q = actual_seq_lengths_q
        self.attn_state = attn_state
        self.seq_lens = seq_lens            # CPU tensor
        self.seq_lens_cpu = seq_lens
        self.seq_lens_list = seq_lens.tolist()
        self.block_tables = block_tables    # GPU tensor
        self.slot_mapping = slot_mapping


class FakeRunner:
    """Mirrors the NPUModelRunner attribute surface used by the patches."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.attn_state = ATTN_STATES["ChunkedPrefill"]
        self.input_batch = None
        self.kv_cache_config = None
        self.compilation_config = None
        self.vllm_config = None
        self.model_config = None
        self.requests = {}
        self.speculative_config = None

    def build(self, num_layers=4, kv_heads=2, head_size=8, num_heads=8, block_size=16,
              max_blocks=64, max_reqs=16, seed=0):
        layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]
        spec = FakeKVSpec(block_size, kv_heads, head_size)
        self.kv_cache_config = FakeKVCacheConfig([FakeKVGroup(layer_names, spec)])
        sfc = {}
        for i, ln in enumerate(layer_names):
            sfc[ln] = FakeAttentionModule(ln, max_blocks, block_size, kv_heads, head_size,
                                          num_heads, seed=seed + i)
        self.compilation_config = FakeCompilationConfig(sfc)
        self.vllm_config = FakeVllmConfig(sfc)
        self.model_config = self.vllm_config.model_config
        return self


# ---------------------------------------------------------------------------
# Step driver (reproduces vLLM v1 timing order)
# ---------------------------------------------------------------------------


class SimDriver:
    def __init__(self, runner: FakeRunner, mgr):
        self.runner = runner
        self.mgr = mgr
        self.true_seq_lens: dict[str, int] = {}

    def _grow_rows(self, sched: FakeSchedulerOutput) -> None:
        """Scheduler-side block growth (engine-core mirror): append blocks for
        the newly scheduled tokens of each request."""
        bt = self.runner.input_batch.block_table[0]
        bs = bt.block_size
        ib = self.runner.input_batch
        for r, req_id in enumerate(ib.req_ids):
            n_sched = int(sched.num_scheduled_tokens.get(req_id, 0))
            cur_len = int(ib.num_computed_tokens_cpu[r])
            new_len = cur_len + n_sched
            n_blocks = (new_len + bs - 1) // bs
            have = int(bt.num_blocks_per_row[r])
            # physical ids must stay within the fake cache tensor
            # (num_blocks = max_blocks): use r*100 + b
            for b in range(have, n_blocks):
                bt.block_table.np[r, b] = r * 100 + b
            bt.num_blocks_per_row[r] = n_blocks
            self.true_seq_lens[req_id] = new_len

    def _build_metadata(self, sched: FakeSchedulerOutput, attn_state) -> dict:
        """Replicates AscendAttentionMetadataBuilder.build semantics."""
        ib = self.runner.input_batch
        bt = ib.block_table[0]
        num_reqs = len(ib.req_ids)
        qsl = [int(sched.num_scheduled_tokens.get(r, 0)) for r in ib.req_ids]
        actual_q = [max(1, q) for q in qsl]
        seq_lens = torch.tensor([self.true_seq_lens[r] for r in ib.req_ids], dtype=torch.int64)
        block_tables = bt.get_device_tensor()[:num_reqs]
        slot_mapping = bt.slot_mapping.gpu
        meta = {}
        for ln in self.runner.kv_cache_config.kv_cache_groups[0].layer_names:
            meta[ln] = FakeAscendMetadata(
                num_actual_tokens=sum(actual_q),
                actual_seq_lengths_q=actual_q,
                attn_state=attn_state,
                seq_lens=seq_lens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
            )
        return meta

    def _simulate_forward(self, sched: FakeSchedulerOutput, meta: dict, attn_state) -> None:
        """Backend forward: write this step's tokens' KV at true slots and
        report the TND query tensor for capture."""
        ib = self.runner.input_batch
        bt = ib.block_table[0]
        bs = bt.block_size
        num_reqs = len(ib.req_ids)
        qsl = [int(sched.num_scheduled_tokens.get(r, 0)) for r in ib.req_ids]
        offsets = [0]
        for q in qsl:
            offsets.append(offsets[-1] + q)
        total = offsets[-1]
        if total == 0:
            return
        positions = torch.zeros(total, dtype=torch.int64)
        for r in range(num_reqs):
            base = int(ib.num_computed_tokens_cpu[r])
            positions[offsets[r]:offsets[r + 1]] = torch.arange(base, base + qsl[r])
        ib.block_table.compute_slot_mapping(num_reqs, torch.tensor(offsets, dtype=torch.int64), positions)
        slots = bt.slot_mapping.np[:total]
        layer0 = self.runner.kv_cache_config.kv_cache_groups[0].layer_names[0]
        num_heads = self.runner.compilation_config.static_forward_context[layer0].num_heads
        hd = self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.head_size
        # fake TND queries: token-dependent so scores differ
        query = torch.randn(total, num_heads, hd) * 0.5
        g = torch.Generator().manual_seed(1234)
        kv = torch.randn(total, self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.num_kv_heads,
                         hd, generator=g) * 0.1
        for ln, mod in self.runner.compilation_config.static_forward_context.items():
            kc, vc = mod.kv_cache
            flat_k = kc.view(-1, kc.shape[2], kc.shape[3])
            flat_v = vc.view(-1, vc.shape[2], vc.shape[3])
            slots_t = torch.from_numpy(slots.astype(np.int64))
            flat_k[slots_t] = kv
            flat_v[slots_t] = kv * 0.7 + 0.1
        return query, qsl, positions, slots, offsets

    def run_step(self, sched: FakeSchedulerOutput, attn_state) -> dict:
        runner = self.runner
        mgr = self.mgr
        runner.attn_state = attn_state
        mgr.on_step_begin(runner, sched)
        self._grow_rows(sched)
        mgr.on_prepare_inputs_entry(runner)
        meta = self._build_metadata(sched, attn_state)
        mgr.on_metadata_built(runner, meta, None)
        fwd = self._simulate_forward(sched, meta, attn_state)
        if fwd:
            query, qsl, positions, slots, offsets = fwd
            num_reqs = len(runner.input_batch.req_ids)
            layer0 = runner.kv_cache_config.kv_cache_groups[0].layer_names[0]
            for ln in runner.kv_cache_config.kv_cache_groups[0].layer_names:
                mod = runner.compilation_config.static_forward_context[ln]
                fake_meta = meta[ln]
                fake_meta.num_actual_tokens = query.shape[0]
                mgr.on_backend_forward(ln, query, fake_meta, is_draft=False)
                mgr.on_attn_module(ln, torch.randn(query.shape[0], 32), is_draft=False)
        mgr.on_step_end(runner, sched)
        # sample_tokens updates num_computed (vLLM v1 timing)
        for r, req_id in enumerate(runner.input_batch.req_ids):
            runner.input_batch.num_computed_tokens_cpu[r] += int(sched.num_scheduled_tokens.get(req_id, 0))
        return meta

    def attention_of(self, meta, layer_name: str, slots_flat, kv_ref) -> torch.Tensor:
        """Reference attention over the VISIBLE view (block rows + seq lens)."""
        bt = self.runner.input_batch.block_table[0]
        m = meta[layer_name]
        rows = m.block_tables
        seq_lens = m.seq_lens_list
        outs = []
        num_reqs = len(self.runner.input_batch.req_ids)
        for r in range(num_reqs):
            row = rows[r].numpy() if torch.is_tensor(rows) else np.asarray(rows[r])
            L = int(seq_lens[r])
            blocks = row[: (L + bt.block_size - 1) // bt.block_size]
            idx = []
            for b in blocks:
                idx.extend(range(int(b) * bt.block_size, int(b) * bt.block_size + bt.block_size))
            idx = np.array(idx[:L], dtype=np.int64)
            outs.append(idx)
        return outs


def run_scenario(steps: int = 8, mode: str = "view", press_name: str = "snapkv",
                 ratio: float = 0.5, verbose: bool = True) -> bool:
    from kvpress_ascend import envs
    from kvpress_ascend.capture import CaptureManager
    from kvpress_ascend.presses import build_press

    os_env_backup = {}
    for k, v in list(__import__("os").environ.items()):
        os_env_backup[k] = v
    __import__("os").environ["KVPRESS_ASCEND_PRESS"] = press_name
    __import__("os").environ["KVPRESS_ASCEND_RATIO"] = str(ratio)
    __import__("os").environ["KVPRESS_ASCEND_MODE"] = mode
    __import__("os").environ["KVPRESS_ASCEND_MIN_PROMPT"] = "0"

    runner = FakeRunner().build(num_layers=4, kv_heads=2, head_size=8, num_heads=8,
                                block_size=16, max_blocks=256, max_reqs=16)
    mgr = CaptureManager()
    mgr.mode = mode
    mgr.press = build_press(press_name, ratio, window=8, sink=2, kernel=3)
    mgr.capture_w = 64
    driver = SimDriver(runner, mgr)

    # prefill: 3 chunked steps for req0 (prompt 260 tokens), 1 step for req1 (prompt 80)
    prompt0 = 260
    chunk = 100
    req_ids = ["r0", "r1"]
    ib = FakeInputBatch(req_ids, [0, 0], [prompt0, 80], 16, 256, 16)
    runner.input_batch = ib

    step = 0
    for done in (100, 200, prompt0):
        step += 1
        sched = FakeSchedulerOutput({"r0": done - int(ib.num_computed_tokens_cpu[0])}, 0)
        if done < prompt0:
            state = ATTN_STATES["ChunkedPrefill"]
        else:
            state = ATTN_STATES["DecodeOnly"] if step >= 3 else ATTN_STATES["ChunkedPrefill"]
        driver.run_step(sched, state)
    # req1 prefill
    step += 1
    sched = FakeSchedulerOutput({"r0": 1, "r1": 80}, 0)
    driver.run_step(sched, ATTN_STATES["ChunkedPrefill"])
    # decode steps (mixed batch, r0 already compacted)
    for _ in range(steps):
        step += 1
        sched = FakeSchedulerOutput({"r0": 1, "r1": 1}, 0)
        driver.run_step(sched, ATTN_STATES["DecodeOnly"])

    ok = True
    if verbose:
        state = {k: len(v) for k, v in mgr.layouts.items()} if mode == "view" \
            else {k: f"kept={v.n_kept}" for k, v in mgr.compact.items()}
        print(f"scenario OK: mode={mode} press={press_name} steps={steps} layouts={state}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="kvpress-ascend offline simulator")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mode", choices=["view", "compact"], default="view")
    ap.add_argument("--press", default="snapkv")
    ap.add_argument("--ratio", type=float, default=0.5)
    args = ap.parse_args(argv)
    try:
        ok = run_scenario(args.steps, args.mode, args.press, args.ratio)
    except Exception as exc:  # noqa: BLE001
        print(f"SCENARIO FAILED: {exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
