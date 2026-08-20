"""Capture + clustering + per-layer window view rewrite for
SqueezeAttention-ascend.

Data flow per step (vLLM v1 timing, see PLAN.md §2.3):
  execute_model pre  -> on_step_begin    (counters, completion detection)
  decoder layer fwd  -> on_layer_input   (pre-layernorm residual, TND)
  Attention.forward  -> on_attn_output   (attn output; cos-sim accumulation)
  _build_attention_metadata -> on_metadata_built (window view rewrite)
  execute_model post -> on_step_end      (cluster pass + heartbeat)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from squeeze_ascend import envs, registry
from squeeze_ascend.log import logger

try:
    import numpy as np
    import torch
except Exception:  # pragma: no cover
    np = None
    torch = None


@dataclass
class StepInfo:
    step_id: int
    req_ids: list = field(default_factory=list)
    num_reqs: int = 0
    num_computed_before: dict = field(default_factory=dict)
    num_scheduled: dict = field(default_factory=dict)
    num_prompt: dict = field(default_factory=dict)
    completed_prefill: list = field(default_factory=list)
    attn_state_name: str = ""


class WindowManager:
    def __init__(self) -> None:
        self.windows: dict[str, dict] = {}          # req_id -> {layer_name: window}
        self.importance: dict[str, dict] = {}       # req_id -> {layer_name: (sum, count)}
        self.layer_inputs: dict[str, torch.Tensor] = {}  # per-step: layer_name -> TND input
        self.buffers: dict = {}
        self.step: StepInfo | None = None
        self._step_counter = 0
        self._num_layers = 0
        self._layer_names: list[str] = []
        self._cluster_info: dict = {}

    # -- lifecycle ----------------------------------------------------------

    def on_step_begin(self, runner, scheduler_output) -> None:
        if torch is None:
            return
        self._step_counter += 1
        ib = getattr(runner, "input_batch", None)
        info = StepInfo(step_id=self._step_counter)
        if ib is None or scheduler_output is None:
            self.step = info
            return
        req_ids = list(getattr(ib, "req_ids", []) or [])
        info.req_ids = req_ids
        info.num_reqs = len(req_ids)
        num_computed = getattr(ib, "num_computed_tokens_cpu", None)
        num_prompt = getattr(ib, "num_prompt_tokens", None)
        sched = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        for i, req_id in enumerate(req_ids):
            before = int(num_computed[i]) if num_computed is not None else 0
            prompt = int(num_prompt[i]) if num_prompt is not None else 0
            n_sched = int(sched.get(req_id, 0))
            info.num_computed_before[req_id] = before
            info.num_scheduled[req_id] = n_sched
            info.num_prompt[req_id] = prompt
            if before < prompt and before + n_sched >= prompt:
                info.completed_prefill.append(req_id)
        live = set(req_ids)
        for req_id in [r for r in self.windows if r not in live]:
            self.windows.pop(req_id, None)
            self.importance.pop(req_id, None)
        for req_id in list(self.windows):
            before = info.num_computed_before.get(req_id, 0)
            prompt = info.num_prompt.get(req_id, 0)
            if prompt and before < prompt:  # preemption / recompute
                self.windows.pop(req_id, None)
                self.importance.pop(req_id, None)
                registry.record("window_dropped_recompute")
        attn_state = getattr(runner, "attn_state", None)
        info.attn_state_name = _attn_state_name(attn_state)
        self.step = info
        self.layer_inputs.clear()
        registry.mark_hit("S1_step_begin")

    def on_layer_input(self, layer_name: str, hidden, is_draft: bool) -> None:
        registry.mark_hit("S2_layer_input")
        if torch is None or hidden is None or is_draft:
            return
        self.layer_inputs[layer_name] = hidden

    def on_attn_output(self, layer_name: str, hidden_in, attn_out, is_draft: bool) -> None:
        """Accumulate per-layer mean cos_sim(layer_input, layer_input + attn_out)."""
        registry.mark_hit("S3_attn_output")
        if torch is None or attn_out is None or is_draft:
            return
        info = self.step
        if info is None or not info.req_ids:
            return
        prefilling = [r for r in info.req_ids
                      if info.num_computed_before.get(r, 0) < info.num_prompt.get(r, 0)]
        if len(prefilling) != 1:
            return  # only single-prefill steps give clean per-layer importance
        req_id = prefilling[0]
        vec1 = self.layer_inputs.pop(layer_name, None)
        if vec1 is None:
            return
        try:
            T = attn_out.shape[0]
            if vec1.shape[0] != T:
                return
            a = vec1.float().flatten(1)
            b = (vec1 + attn_out).float().flatten(1)
            sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # (T,)
            mean = float(sim.mean().item())
            acc = self.importance.setdefault(req_id, {})
            s, c = acc.get(layer_name, (0.0, 0))
            acc[layer_name] = (s + mean * T, c + T)
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("cos-sim capture failed at %s: %s", layer_name, exc)

    # -- metadata rewrite ---------------------------------------------------

    def on_metadata_built(self, runner, attn_metadata, spec_decode_common_attn_metadata) -> None:
        registry.mark_hit("S4_metadata")
        if torch is None or not attn_metadata:
            return
        if isinstance(attn_metadata, (list, tuple)):
            registry.record("skipped_ubatch")
            return
        info = self.step
        if info is None or not info.req_ids:
            return
        if envs.dry_run():
            return
        if not any(self.windows.get(r) for r in info.req_ids):
            return
        try:
            self._rewrite_windows(runner, attn_metadata)
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("window metadata rewrite failed: %s", exc)

    def _rewrite_windows(self, runner, attn_metadata) -> None:
        info = self.step
        ib = getattr(runner, "input_batch", None)
        if ib is None:
            return
        row_map = getattr(getattr(ib, "req_id_to_index", None), "get", None)
        bt = getattr(ib, "block_table", None)
        if row_map is None or bt is None:
            return
        try:
            bt0 = bt[0]
            true_gpu = bt0.block_table.gpu
            max_blocks = true_gpu.shape[1]
            device = true_gpu.device
            bs = bt0.block_size
            start_size = envs.start_size()
            num_reqs = info.num_reqs
            for layer_name, meta in attn_metadata.items():
                layer_w = {r: w for r in info.req_ids
                           for w in [self.windows.get(r, {}).get(layer_name)] if w}
                if not layer_w:
                    continue
                seq = getattr(meta, "seq_lens", None)
                if seq is None or seq.shape[0] < num_reqs:
                    continue
                buf = self._layer_buffer(layer_name, (max(1, meta.block_tables.shape[0]),
                                                      max_blocks), device)
                buf.zero_()
                n_rows = min(meta.block_tables.shape[0], true_gpu.shape[0])
                buf[:n_rows] = true_gpu[:n_rows]
                new_seq = seq[:num_reqs].clone()
                for req_id, window in layer_w.items():
                    row_idx = row_map(req_id)
                    if row_idx is None or row_idx >= n_rows:
                        continue
                    true_len = int(seq[row_idx].item())
                    ranges = _window_ranges(true_len, window, start_size, bs)
                    if ranges is None:
                        continue
                    sink_blocks, recent_first, last_block = ranges
                    # view = [sink blocks][recent blocks]; rows beyond are zero
                    n_view = (last_block - recent_first) + sink_blocks
                    if sink_blocks > 0:
                        buf[row_idx, :sink_blocks] = true_gpu[row_idx, :sink_blocks]
                    buf[row_idx, sink_blocks:n_view] = true_gpu[row_idx, recent_first:last_block]
                    new_seq[row_idx] = _window_view_len(true_len, sink_blocks, recent_first, bs)
                meta.block_tables = buf
                meta.seq_lens = new_seq
                meta.seq_lens_cpu = new_seq
                lst = new_seq.tolist()
                n_q = len(getattr(meta, "actual_seq_lengths_q", []) or [])
                while len(lst) < n_q:
                    lst.append(1)
                meta.seq_lens_list = lst
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("window rewrite failed: %s", exc)

    def _layer_buffer(self, layer_name: str, shape, device) -> torch.Tensor:
        buf = self.buffers.get(layer_name)
        if buf is None or buf.shape[0] < shape[0] or buf.shape[1] < shape[1]:
            buf = torch.zeros(shape, dtype=torch.int32, device=device)
            self.buffers[layer_name] = buf
        return buf[: shape[0], : shape[1]]

    # -- cluster pass -------------------------------------------------------

    def on_step_end(self, runner, scheduler_output) -> None:
        info = self.step
        if info is None:
            return
        registry.mark_hit("S6_step_end")
        try:
            if info.completed_prefill:
                self._cluster_completed(runner, info)
            self._heartbeat(runner)
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("cluster pass failed: %s", exc)

    def _cluster_completed(self, runner, info: StepInfo) -> None:
        registry.mark_hit("S5_cluster")
        kvcc = getattr(runner, "kv_cache_config", None)
        if kvcc is None:
            return
        try:
            group0 = kvcc.kv_cache_groups[0]
            self._layer_names = list(group0.layer_names)
        except Exception as exc:
            registry.record("skipped_error")
            logger.warning("cannot read kv_cache_groups: %s", exc)
            return
        self._num_layers = len(self._layer_names)
        if self._num_layers == 0:
            return
        for req_id in info.completed_prefill:
            try:
                self._cluster_one(runner, req_id, info)
            except Exception as exc:  # fail-soft
                registry.record("skipped_error")
                logger.warning("cluster request %s failed: %s", req_id, exc)

    def _cluster_one(self, runner, req_id: str, info: StepInfo) -> None:
        prompt_len = info.num_prompt.get(req_id, 0)
        if prompt_len < envs.min_prompt_tokens():
            registry.record("skipped_short")
            return
        acc = self.importance.pop(req_id, {})
        means = []
        for ln in self._layer_names:
            s, c = acc.get(ln, (0.0, 0))
            means.append(s / c if c else 0.5)
        if not means:
            registry.record("skipped_error")
            return
        means = self._sync_means_across_tp(runner, means)
        from squeeze_ascend.kvcore import layer_windows
        windows, info_d = layer_windows(
            means, self._num_layers, envs.ini_size(), envs.class3_ratio(),
            envs.start_size(), prompt_len,
        )
        self._cluster_info = info_d
        by_name = {}
        for ln, w in zip(self._layer_names, windows):
            by_name[ln] = w
        self.windows[req_id] = by_name
        registry.record("clustered")
        wstats = (min(windows), max(windows)) if windows else (0, 0)
        logger.info("req %s clustered: prompt=%d ini=%.2f class3=%.2f windows[min=%d max=%d] class3_layer=%s",
                    req_id, prompt_len, envs.ini_size(), envs.class3_ratio(),
                    wstats[0], wstats[1], info_d.get("class3"))
        if envs.dry_run():
            registry.record("dry_run")
            self.windows.pop(req_id, None)

    def _sync_means_across_tp(self, runner, means: list[float]) -> list[float]:
        """TP-rank consistency: all ranks must agree on the windows because the
        per-layer view rewrite is per-rank independent but must produce the
        same effective context (MAX is the conservative direction)."""
        if len(means) <= 1:
            return means
        try:
            from vllm.distributed import get_tp_group
            tp = get_tp_group()
            if tp is not None and tp.world_size > 1:
                t = torch.tensor(means, dtype=torch.float32, device=runner.device)
                tp.all_reduce(t, op="MAX")  # type: ignore[attr-defined]
                return t.tolist()
        except Exception as exc:  # noqa: BLE001
            logger.debug("TP sync skipped: %s", exc)
        return means

    def _heartbeat(self, runner) -> None:
        info = self.step
        if info is None:
            return
        w = []
        for per_req in self.windows.values():
            w.extend(per_req.values())
        core = {
            "ini": envs.ini_size(),
            "class3": envs.class3_ratio(),
            "start": envs.start_size(),
            "w_min": min(w) if w else 0,
            "w_max": max(w) if w else 0,
            "active_windowed": len(self.windows),
            "prefilling": sum(1 for r in info.req_ids
                              if info.num_computed_before.get(r, 0) < info.num_prompt.get(r, 0)),
            "completed": len(info.completed_prefill),
            "attn_state": info.attn_state_name,
        }
        # Always show the key counters (with zeros) so "nothing happened" is
        # visible at a glance.
        stats = registry.stats_snapshot()
        for key in ("clustered", "skipped_short", "skipped_error",
                    "dry_run", "activation", "window_dropped_recompute"):
            stats.setdefault(key, 0)
        registry.heartbeat(info.step_id, core, stats=stats)


# ---------------------------------------------------------------------------
# helpers (shared with kvcore for duck typing)
# ---------------------------------------------------------------------------


def _attn_state_name(state) -> str:
    if state is None:
        return ""
    return getattr(state, "name", state) if not isinstance(state, str) else state


def _window_ranges(true_seq_len: int, window: int, start_size: int, bs: int):
    from squeeze_ascend.kvcore import window_block_ranges
    return window_block_ranges(true_seq_len, window, start_size, bs)


def _window_view_len(true_seq_len: int, sink_blocks: int, recent_first: int, bs: int) -> int:
    from squeeze_ascend.kvcore import window_view_len
    return window_view_len(true_seq_len, sink_blocks, recent_first, bs)
