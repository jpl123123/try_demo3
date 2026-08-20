"""L1/L2 offline simulator for SqueezeAttention-ascend.

Usage:
    python -m squeeze_ascend.simulate            # default scenario
    python -m squeeze_ascend.simulate --steps 8
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch  # noqa: E402

try:
    from kvpress_ascend.simulate import (  # reuse the fakes (same attribute surface)
        ATTN_STATES,
        FakeAttentionModule,
        FakeCompilationConfig,
        FakeInputBatch,
        FakeKVCacheConfig,
        FakeKVGroup,
        FakeKVSpec,
        FakeModelConfig,
        FakeMultiGroupBlockTable,
        FakeRunner as _BaseFakeRunner,
        FakeSchedulerOutput,
        FakeVllmConfig,
        SimpleNamespace,
    )
except ModuleNotFoundError:  # pragma: no cover - CLI run from this package dir
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kvpress-ascend"))
    from kvpress_ascend.simulate import (  # noqa: E402
        ATTN_STATES,
        FakeAttentionModule,
        FakeCompilationConfig,
        FakeInputBatch,
        FakeKVCacheConfig,
        FakeKVGroup,
        FakeKVSpec,
        FakeModelConfig,
        FakeMultiGroupBlockTable,
        FakeRunner as _BaseFakeRunner,
        FakeSchedulerOutput,
        FakeVllmConfig,
        SimpleNamespace,
    )


class FakeDecoderLayer:
    def __init__(self, self_attn: FakeAttentionModule):
        self.self_attn = self_attn
        self.layer_idx = int(self_attn.layer_name.split(".")[2])
        self.forward = self._forward

    def _forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class FakeModel:
    def __init__(self, layers):
        self.layers = layers


class FakeRunner(_BaseFakeRunner):
    def get_model(self):
        if self._model is None:
            layers = []
            for ln in self.kv_cache_config.kv_cache_groups[0].layer_names:
                layers.append(FakeDecoderLayer(self.compilation_config.static_forward_context[ln]))
            self._model = FakeModel(layers)
        return self._model

    def build(self, *a, **k):
        self._model = None
        return super().build(*a, **k)


class SimDriver:
    """Same timing order as the kvpress simulator; adds the layer forward +
    attention output capture steps."""

    def __init__(self, runner: FakeRunner, mgr):
        self.runner = runner
        self.mgr = mgr
        self.true_seq_lens: dict[str, int] = {}

    def _grow_rows(self, sched) -> None:
        bt = self.runner.input_batch.block_table[0]
        bs = bt.block_size
        ib = self.runner.input_batch
        for r, req_id in enumerate(ib.req_ids):
            n_sched = int(sched.num_scheduled_tokens.get(req_id, 0))
            cur_len = int(ib.num_computed_tokens_cpu[r])
            new_len = cur_len + n_sched
            n_blocks = (new_len + bs - 1) // bs
            have = int(bt.num_blocks_per_row[r])
            for b in range(have, n_blocks):
                bt.block_table.np[r, b] = r * 100 + b
            bt.num_blocks_per_row[r] = n_blocks
            self.true_seq_lens[req_id] = new_len

    def _build_metadata(self, sched, attn_state) -> dict:
        ib = self.runner.input_batch
        bt = ib.block_table[0]
        num_reqs = len(ib.req_ids)
        qsl = [int(sched.num_scheduled_tokens.get(r, 0)) for r in ib.req_ids]
        actual_q = [max(1, q) for q in qsl]
        seq_lens = torch.tensor([self.true_seq_lens[r] for r in ib.req_ids], dtype=torch.int64)
        block_tables = bt.get_device_tensor()[:num_reqs]
        from kvpress_ascend.simulate import FakeAscendMetadata
        meta = {}
        for ln in self.runner.kv_cache_config.kv_cache_groups[0].layer_names:
            meta[ln] = FakeAscendMetadata(
                num_actual_tokens=sum(actual_q),
                actual_seq_lengths_q=actual_q,
                attn_state=attn_state,
                seq_lens=seq_lens,
                block_tables=block_tables,
                slot_mapping=bt.slot_mapping.gpu,
            )
        return meta

    def _simulate_forward(self, sched, meta) -> tuple:
        ib = self.runner.input_batch
        bt = ib.block_table[0]
        num_reqs = len(ib.req_ids)
        qsl = [int(sched.num_scheduled_tokens.get(r, 0)) for r in ib.req_ids]
        offsets = [0]
        for q in qsl:
            offsets.append(offsets[-1] + q)
        total = offsets[-1]
        if total == 0:
            return None
        positions = torch.zeros(total, dtype=torch.int64)
        for r in range(num_reqs):
            base = int(ib.num_computed_tokens_cpu[r])
            positions[offsets[r]:offsets[r + 1]] = torch.arange(base, base + qsl[r])
        ib.block_table.compute_slot_mapping(num_reqs, torch.tensor(offsets, dtype=torch.int64), positions)
        slots = bt.slot_mapping.np[:total]
        kvh = self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.num_kv_heads
        hd = self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.head_size
        g = torch.Generator().manual_seed(99)
        kv = torch.randn(total, kvh, hd, generator=g) * 0.1
        for mod in self.runner.compilation_config.static_forward_context.values():
            kc, vc = mod.kv_cache
            slots_t = torch.from_numpy(slots.astype(np.int64))
            kc.view(-1, kc.shape[2], kc.shape[3])[slots_t] = kv
            vc.view(-1, vc.shape[2], vc.shape[3])[slots_t] = kv * 0.6 + 0.2
        return total, offsets, slots

    def run_step(self, sched, attn_state) -> dict:
        runner = self.runner
        mgr = self.mgr
        runner.attn_state = attn_state
        mgr.on_step_begin(runner, sched)
        # decoder layer forwards + attention modules run inside _model_forward;
        # emulate with a deterministic hidden input per layer
        self._grow_rows(sched)
        meta = self._build_metadata(sched, attn_state)
        hidden_in = {}
        g = torch.Generator().manual_seed(7)
        T = sum(max(1, int(sched.num_scheduled_tokens.get(r, 0))) for r in runner.input_batch.req_ids)
        for i, ln in enumerate(runner.kv_cache_config.kv_cache_groups[0].layer_names):
            hidden_in[ln] = torch.randn(T, 32, generator=g)
            mgr.on_layer_input(ln, hidden_in[ln], is_draft=False)
        mgr.on_metadata_built(runner, meta, None)
        fwd = self._simulate_forward(sched, meta)
        if fwd:
            for i, ln in enumerate(runner.kv_cache_config.kv_cache_groups[0].layer_names):
                attn_out = hidden_in[ln] * 0.9 + torch.randn_like(hidden_in[ln]) * (0.05 * (i + 1))
                mgr.on_attn_output(ln, hidden_in[ln], attn_out, is_draft=False)
        mgr.on_step_end(runner, sched)
        # sample_tokens updates num_computed (vLLM v1 timing)
        for r, req_id in enumerate(runner.input_batch.req_ids):
            runner.input_batch.num_computed_tokens_cpu[r] += int(sched.num_scheduled_tokens.get(req_id, 0))
        return meta


def run_scenario(steps: int = 8, ini: float = 0.3, class3: float = 0.1,
                 start: int = 4, verbose: bool = True) -> bool:
    import os

    os.environ["SQUEEZE_ASCEND_INI_SIZE"] = str(ini)
    os.environ["SQUEEZE_ASCEND_CLASS3_RATIO"] = str(class3)
    os.environ["SQUEEZE_ASCEND_START_SIZE"] = str(start)
    os.environ["SQUEEZE_ASCEND_MIN_PROMPT"] = "0"

    from squeeze_ascend.capture import WindowManager

    runner = FakeRunner().build(num_layers=4, kv_heads=2, head_size=8, num_heads=8,
                                block_size=16, max_blocks=256, max_reqs=16)
    mgr = WindowManager()
    driver = SimDriver(runner, mgr)

    prompt0 = 260
    req_ids = ["r0", "r1"]
    ib = FakeInputBatch(req_ids, [0, 0], [prompt0, 80], 16, 256, 16)
    runner.input_batch = ib

    for done in (100, 200, prompt0):
        sched = FakeSchedulerOutput({"r0": done - int(ib.num_computed_tokens_cpu[0])}, 0)
        driver.run_step(sched, ATTN_STATES["ChunkedPrefill"])
    driver.run_step(FakeSchedulerOutput({"r0": 1, "r1": 80}, 0), ATTN_STATES["ChunkedPrefill"])
    for _ in range(steps):
        driver.run_step(FakeSchedulerOutput({"r0": 1, "r1": 1}, 0), ATTN_STATES["DecodeOnly"])

    ok = bool(mgr.windows.get("r0"))
    if verbose:
        w = mgr.windows.get("r0", {})
        print(f"scenario OK={ok}: windows[r0]={ {k.split('.')[2]: v for k, v in w.items()} }")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SqueezeAttention-ascend offline simulator")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--ini", type=float, default=0.3)
    ap.add_argument("--class3", type=float, default=0.1)
    ap.add_argument("--start", type=int, default=4)
    args = ap.parse_args(argv)
    try:
        ok = run_scenario(args.steps, args.ini, args.class3, args.start)
    except Exception as exc:  # noqa: BLE001
        print(f"SCENARIO FAILED: {exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
