"""Capture buffers and the compression pass (device side, duck-typed against
the vllm-ascend runner so L1/L2 fakes can drive it offline).

Data flow per step (see PLAN.md §2.3):
  execute_model pre  -> on_step_begin  (snapshot counters, detect completion)
  _prepare_inputs    -> on_prepare_inputs_entry (compact mode row rewrite)
  backend forward    -> on_backend_forward (query capture, prefill only)
  Attention.forward  -> on_attn_module (hidden capture, prefill only)
  _build_attention_metadata -> on_metadata_built (per-layer view rewrite)
  execute_model post -> on_step_end   (compression pass for completed prefills,
                                       layout recording, heartbeat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from kvpress_ascend import envs, registry
from kvpress_ascend.log import logger

try:  # torch is only imported when the package is activated
    import numpy as np
    import torch
except Exception:  # pragma: no cover - offline import safety
    np = None
    torch = None


# ---------------------------------------------------------------------------
# Per-step context
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    step_id: int
    req_ids: list = field(default_factory=list)
    num_reqs: int = 0
    num_computed_before: dict = field(default_factory=dict)   # req_id -> int
    num_scheduled: dict = field(default_factory=dict)         # req_id -> int
    num_prompt: dict = field(default_factory=dict)            # req_id -> int
    completed_prefill: list = field(default_factory=list)     # req_ids done THIS step
    with_prefill: bool = False
    attn_state_name: str = ""


@dataclass
class RequestCapture:
    req_id: str
    queries: dict = field(default_factory=dict)   # layer_name -> (w, h, hd) device
    hidden: dict = field(default_factory=dict)    # layer_name -> (w, H) device
    captured_q: dict = field(default_factory=dict)  # layer_name -> int (tokens captured)


class CaptureManager:
    """Owns per-request capture buffers and per-request layouts."""

    def __init__(self) -> None:
        self.requests: dict[str, RequestCapture] = {}
        self.layouts: dict[str, dict] = {}          # req_id -> {layer_name: ViewLayout|CompactLayout}
        self.compact: dict[str, "object"] = {}      # req_id -> CompactLayout (compact mode)
        self.row_rewritten: set = set()             # req_ids whose np row is already permuted
        self.mid_anchors: dict[str, int] = {}       # req_id -> current mid-prefill anchor (view mode)
        self._last_before: dict[str, int] = {}      # req_id -> num_computed seen last step (drop detection)
        self.capture_w = 0
        self.buffers: dict = {}                     # per-layer view-row device buffers
        self.press = None
        self.mode = "view"
        self.step: StepInfo | None = None
        self._step_counter = 0
        self._active_prefills = 0
        self._completed_total = 0
        self._compressed_done: set = set()          # req_ids compressed at completion (no re-trigger)

    # -- lifecycle ----------------------------------------------------------

    def on_step_begin(self, runner, scheduler_output) -> None:
        if torch is None:
            return
        self._step_counter += 1
        step_id = self._step_counter
        ib = getattr(runner, "input_batch", None)
        info = StepInfo(step_id=step_id)
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
            if before == 0:
                info.with_prefill = True
        # drop state of requests no longer in the batch
        live = set(req_ids)
        for req_id in [r for r in self.requests if r not in live]:
            self.requests.pop(req_id, None)
            self.layouts.pop(req_id, None)
            self.compact.pop(req_id, None)
            self.row_rewritten.discard(req_id)
            self.mid_anchors.pop(req_id, None)
            self._last_before.pop(req_id, None)
            self._compressed_done.discard(req_id)
        # recompute/preemption detection by num_computed REGRESSION (a request
        # may legitimately still be prefilling while carrying a mid-prefill
        # layout, so "before < prompt" is no longer a valid drop signal)
        for req_id in list(self.layouts):
            before = info.num_computed_before.get(req_id, 0)
            prev = self._last_before.get(req_id, 0)
            if before < prev:
                self.layouts.pop(req_id, None)
                self.compact.pop(req_id, None)
                self.row_rewritten.discard(req_id)
                self.mid_anchors.pop(req_id, None)
                self._compressed_done.discard(req_id)
                registry.record("layout_dropped_recompute")
        # catch-up: a request that crossed the prefill boundary between steps
        # WITHOUT the strict check firing (e.g. final-chunk under-count with
        # MTP async scheduling). last_before < prompt <= before now -> it
        # completed in the previous step; compress it this step.
        for req_id in req_ids:
            before = info.num_computed_before.get(req_id, 0)
            prompt = info.num_prompt.get(req_id, 0)
            prev = self._last_before.get(req_id, 0)
            if (req_id not in info.completed_prefill
                    and req_id not in self._compressed_done
                    and prev < prompt <= before):
                info.completed_prefill.append(req_id)
                registry.record("completion_caught_up")
        for i, req_id in enumerate(req_ids):
            self._last_before[req_id] = info.num_computed_before.get(req_id, 0)
        attn_state = getattr(runner, "attn_state", None)
        info.attn_state_name = _attn_state_name(attn_state)
        self._active_prefills = sum(
            1 for r in req_ids
            if info.num_computed_before.get(r, 0) < info.num_prompt.get(r, 0)
        )
        self.step = info
        registry.mark_hit("S5_execute_model")
        if logger.isEnabledFor(logging.DEBUG):
            dbg = [(r, info.num_computed_before.get(r, 0),
                    info.num_scheduled.get(r, 0), info.num_prompt.get(r, 0))
                   for r in req_ids[:3]]
            logger.debug("step %d reqs=%d prefill=%d completed=%s sample=%s",
                         step_id, len(req_ids), self._active_prefills,
                         info.completed_prefill, dbg)

    def on_prepare_inputs_entry(self, runner) -> None:
        """Compact mode: idempotent row permutation before commit_block_table."""
        registry.mark_hit("S3_prepare_inputs")
        if self.mode != "compact" or not self.compact:
            return
        if torch is None:
            return
        ib = getattr(runner, "input_batch", None)
        if ib is None:
            return
        row_map = getattr(getattr(ib, "req_id_to_index", None), "get", None)
        if row_map is None:
            return
        bt = getattr(ib, "block_table", None)
        if bt is None:
            return
        try:
            bt0 = bt[0]
            np_buf = bt0.block_table.np
            nbp = bt0.num_blocks_per_row
            for req_id, layout in self.compact.items():
                if req_id in self.row_rewritten:
                    continue  # row already permuted; append_row keeps it stable
                row_idx = row_map(req_id)
                if row_idx is None or row_idx >= np_buf.shape[0]:
                    continue
                valid = int(nbp[row_idx])
                if valid < layout.m:
                    continue
                row = np_buf[row_idx, :valid]
                new_row = layout.rewrite_row(row)
                # The logical row shrinks to k + (valid - m) blocks; the count
                # must shrink too so subsequent append_row calls (which append
                # at num_blocks_per_row) keep landing in the permuted layout.
                new_valid = layout.k + (valid - layout.m)
                np_buf[row_idx, :new_valid] = new_row
                nbp[row_idx] = new_valid
                self.row_rewritten.add(req_id)
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("compact row rewrite failed: %s", exc)

    def on_backend_forward(self, layer_name: str, query, attn_metadata, is_draft: bool) -> None:
        """Capture post-rope queries of prefill steps (per request, rolling)."""
        registry.mark_hit("S1_backend_forward")
        if torch is None or query is None or is_draft:
            return
        state_name = _attn_state_name(getattr(attn_metadata, "attn_state", None))
        if state_name not in ("PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"):
            return
        info = self.step
        if info is None or not info.req_ids:
            return
        num_actual = int(getattr(attn_metadata, "num_actual_tokens", query.shape[0]) or query.shape[0])
        q = query[:num_actual]
        qsl = getattr(attn_metadata, "actual_seq_lengths_q", None) or _qsl_from_meta(attn_metadata)
        if not qsl or len(qsl) != len(info.req_ids):
            return
        if self._active_prefills >= envs.max_tracked_prefills():
            registry.record("capture_dropped_cap")
            return
        cap_w = self.capture_w
        offsets = _cumsum(qsl)
        prefilling = [r for r in info.req_ids if info.num_computed_before.get(r, 0) < info.num_prompt.get(r, 0)]
        for i, req_id in enumerate(info.req_ids):
            if req_id not in prefilling:
                continue
            start, end = offsets[i], offsets[i + 1]
            seg = q[start:end]
            if seg.shape[0] == 0:
                continue
            rc = self.requests.setdefault(req_id, RequestCapture(req_id=req_id))
            prev = rc.queries.get(layer_name)
            if prev is None or prev.shape[0] < seg.shape[0]:
                prev = torch.empty((cap_w,) + tuple(seg.shape[1:]), dtype=seg.dtype, device=seg.device)
                rc.queries[layer_name] = prev
            take = min(seg.shape[0], cap_w)
            prev[:take] = seg[-take:]
            rc.captured_q[layer_name] = take

    def on_attn_module(self, layer_name: str, hidden, is_draft: bool) -> None:
        """Capture hidden states of the last chunk (ExpectedAttention needs
        pre-rope queries re-projected from them)."""
        registry.mark_hit("S2_attn_module")
        if torch is None or hidden is None or is_draft:
            return
        info = self.step
        if info is None or not info.req_ids:
            return
        if self._active_prefills >= envs.max_tracked_prefills():
            return
        prefilling = [r for r in info.req_ids if info.num_computed_before.get(r, 0) < info.num_prompt.get(r, 0)]
        if len(prefilling) != 1:
            return  # only capture when exactly one request is prefilling
        req_id = prefilling[0]
        rc = self.requests.setdefault(req_id, RequestCapture(req_id=req_id))
        cap_w = self.capture_w
        take = min(hidden.shape[0], cap_w)
        if layer_name not in rc.hidden or rc.hidden[layer_name].shape[0] < hidden.shape[0]:
            rc.hidden[layer_name] = torch.empty((cap_w,) + tuple(hidden.shape[1:]),
                                                dtype=hidden.dtype, device=hidden.device)
        rc.hidden[layer_name][:take] = hidden[-take:]

    # -- metadata rewrite ---------------------------------------------------

    def on_metadata_built(self, runner, attn_metadata, spec_decode_common_attn_metadata) -> None:
        registry.mark_hit("S4_attn_metadata")
        if torch is None or not attn_metadata:
            return
        if isinstance(attn_metadata, (list, tuple)):
            # ubatching (multiple ubatch metadata dicts) is not supported yet;
            # fail-soft: leave the step uncompressed
            registry.record("skipped_ubatch")
            return
        try:
            if self.mode == "view":
                self._rewrite_view_metadata(runner, attn_metadata)
            else:
                self._rewrite_compact_metadata(runner, attn_metadata, spec_decode_common_attn_metadata)
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("metadata rewrite failed: %s", exc)

    def _req_delta(self, info: StepInfo, runner) -> torch.Tensor | None:
        """(num_reqs,) int64 delta per request (compact mode, uniform)."""
        if not self.compact:
            return None
        req_ids = info.req_ids
        deltas = np.zeros(len(req_ids), dtype=np.int64)
        for i, req_id in enumerate(req_ids):
            lay = self.compact.get(req_id)
            if lay is not None:
                deltas[i] = lay.delta
        return torch.from_numpy(deltas).to(runner.device)

    def _rewrite_compact_metadata(self, runner, attn_metadata, spec_decode_common_attn_metadata) -> None:
        info = self.step
        if info is None or not info.req_ids:
            return
        delta_t = self._req_delta(info, runner)
        if delta_t is None or not bool(delta_t.any()):
            return
        num_reqs = info.num_reqs
        # per-layer metadata seq_lens is a CPU tensor; delta must be CPU too
        delta_cpu = delta_t.cpu()
        for layer_name, meta in attn_metadata.items():
            seq = getattr(meta, "seq_lens", None)
            if seq is None or seq.shape[0] < num_reqs:
                continue
            new_seq = seq[:num_reqs] - delta_cpu
            new_seq = new_seq.clamp(min=1)
            meta.seq_lens = new_seq
            meta.seq_lens_cpu = new_seq
            lst = new_seq.tolist()
            # keep FIA padding semantics: pad with 1 like the builder does
            n_q = len(getattr(meta, "actual_seq_lengths_q", []) or [])
            while len(lst) < n_q:
                lst.append(1)
            meta.seq_lens_list = lst
        # MTP: give the draft the same compressed view via the common metadata
        if spec_decode_common_attn_metadata is not None:
            cm = spec_decode_common_attn_metadata
            seq = getattr(cm, "seq_lens", None)
            if seq is not None and seq.shape[0] >= num_reqs:
                cm.seq_lens = (seq[:num_reqs] - delta_t).clamp(min=1)
            for attr in ("_seq_lens_cpu", "seq_lens_cpu"):
                cpu = getattr(cm, attr, None)
                if cpu is not None and cpu.shape[0] >= num_reqs:
                    setattr(cm, attr, (cpu[:num_reqs] - delta_t.cpu()).clamp(min=1))

    def _rewrite_view_metadata(self, runner, attn_metadata) -> None:
        info = self.step
        if info is None or not info.req_ids:
            return
        # fast path: nothing to rewrite
        has_any = any(self.layouts.get(r) for r in info.req_ids)
        if not has_any:
            return
        ib = getattr(runner, "input_batch", None)
        if ib is None:
            return
        row_map = getattr(getattr(ib, "req_id_to_index", None), "get", None)
        if row_map is None:
            return
        bt = getattr(ib, "block_table", None)
        if bt is None:
            return
        try:
            bt0 = bt[0]
            true_gpu = bt0.block_table.gpu
            max_blocks = true_gpu.shape[1]
            device = true_gpu.device
            num_reqs = info.num_reqs
            # G11 guard: a kept block id outside the KV cache would poison the
            # FIA's block-table gather on NPU (AIV index out of range).
            num_cache_blocks = self._num_cache_blocks_from_runner(runner)
            for layer_name, meta in attn_metadata.items():
                layer_layouts = {r: l for r in info.req_ids
                                 for l in [self.layouts.get(r, {}).get(layer_name)] if l is not None}
                if not layer_layouts:
                    continue
                seq = getattr(meta, "seq_lens", None)
                if seq is None or seq.shape[0] < num_reqs:
                    continue
                buf = self._layer_buffer(layer_name, (max(1, meta.block_tables.shape[0]),
                                                      max_blocks), device)
                buf.zero_()
                # copy the true rows (all requests) then overwrite compacted ones
                n_rows = min(meta.block_tables.shape[0], true_gpu.shape[0])
                buf[:n_rows] = true_gpu[:n_rows]
                new_seq = seq[:num_reqs].clone()
                for req_id, layout in layer_layouts.items():
                    row_idx = row_map(req_id)
                    if row_idx is None or row_idx >= n_rows:
                        continue
                    if num_cache_blocks and layout.kept_blocks and (
                            min(layout.kept_blocks) < 0 or max(layout.kept_blocks) >= num_cache_blocks):
                        registry.record("skipped_bad_row")
                        logger.warning("req %s kept blocks out of range layer=%s "
                                       "kept[min=%d max=%d] num_cache_blocks=%d - layout dropped",
                                       req_id, layer_name, min(layout.kept_blocks),
                                       max(layout.kept_blocks), num_cache_blocks)
                        self.layouts.pop(req_id, None)
                        self.mid_anchors.pop(req_id, None)
                        continue
                    kept = torch.as_tensor(layout.kept_blocks, dtype=torch.int32, device=device)
                    klen = kept.shape[0]
                    m = layout.m
                    rest_len = max_blocks - m
                    if klen > 0:
                        buf[row_idx, :klen] = kept
                    if rest_len > 0:
                        buf[row_idx, klen:klen + rest_len] = true_gpu[row_idx, m:max_blocks]
                    # seq lens: view len = n_kept + (true - orig)
                    true_len = int(seq[row_idx].item())
                    new_seq[row_idx] = layout.view_seq_len(true_len)
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
            logger.warning("view metadata rewrite failed: %s", exc)

    def _layer_buffer(self, layer_name: str, shape, device) -> torch.Tensor:
        key = layer_name
        buf = self.buffers.get(key)
        if buf is None or buf.shape[0] < shape[0] or buf.shape[1] < shape[1]:
            buf = torch.zeros(shape, dtype=torch.int32, device=device)
            self.buffers[key] = buf
        return buf[: shape[0], : shape[1]]

    # -- compression pass ---------------------------------------------------

    def on_step_end(self, runner, scheduler_output) -> None:
        info = self.step
        if info is None:
            return
        registry.mark_hit("S6_compress_pass")
        try:
            if info.completed_prefill:
                self._compress_completed_prefills(runner, info)
            else:
                self._maybe_mid_prefill(runner, info)
            self._progress_summary(info)
            self._heartbeat(runner)
        except Exception as exc:  # fail-soft
            registry.record("skipped_error")
            logger.warning("compression pass failed: %s", exc)

    def _compress_completed_prefills(self, runner, info: StepInfo) -> None:
        press = self.press
        if press is None:
            return
        if envs.dry_run():
            registry.record("dry_run")
        sfc = getattr(runner, "compilation_config", None)
        sfc = getattr(getattr(runner, "vllm_config", None), "compilation_config", None) or sfc
        sfc = getattr(sfc, "static_forward_context", None) if sfc is not None else None
        if sfc is None:
            sfc = getattr(runner, "static_forward_context", None)
        kvcc = getattr(runner, "kv_cache_config", None)
        if kvcc is None or sfc is None:
            registry.record("skipped_error")
            logger.warning("compression pass skipped: runner has no kv_cache_config/static_forward_context")
            return
        bs = None
        kv_heads = 0
        head_size = 0
        layer_names: list[str] = []
        try:
            group0 = kvcc.kv_cache_groups[0]
            spec = group0.kv_cache_spec
            bs = int(getattr(spec, "block_size", 128) or 128)
            kv_heads = int(getattr(spec, "num_kv_heads", 0) or 0)
            head_size = int(getattr(spec, "head_size", 0) or 0)
            layer_names = list(group0.layer_names)
        except Exception as exc:
            registry.record("skipped_error")
            logger.warning("cannot read kv_cache_spec: %s", exc)
            return
        if bs is None or not layer_names:
            return
        ib = getattr(runner, "input_batch", None)
        if ib is None:
            return
        row_map = getattr(getattr(ib, "req_id_to_index", None), "get", None)
        bt = getattr(ib, "block_table", None)
        if row_map is None or bt is None:
            return
        num_heads = self._num_heads(runner, layer_names[0], kv_heads)
        num_hidden_layers = self._num_hidden_layers(runner)
        for req_id in info.completed_prefill:
            try:
                self._compress_at_length(runner, sfc, ib, bt, row_map, layer_names, bs,
                                         kv_heads, head_size, num_heads, num_hidden_layers,
                                         req_id, info.num_prompt.get(req_id, 0), press,
                                         keep_capture=False, kind="completion")
                self._completed_total += 1
                self._compressed_done.add(req_id)
            except Exception as exc:  # fail-soft per request
                registry.record("skipped_error")
                logger.warning("compress request %s failed: %s", req_id, exc)

    def _maybe_mid_prefill(self, runner, info: StepInfo) -> None:
        """Progressive compression during chunked prefill (view mode only).

        Fixes the chicken-and-egg of the completion-only design: with very long
        prompts the KV cache can fill up before ANY request finishes prefilling
        (preemption loop, `completed=0` forever). Mid-prefill compression
        anchors the layout at the current true length and re-compresses every
        `refresh` tokens.
        """
        if self.mode != "view" or not envs.mid_prefill() or not info.req_ids:
            return
        if self.press is None or envs.dry_run():
            return
        sfc = getattr(runner, "compilation_config", None)
        sfc = getattr(getattr(runner, "vllm_config", None), "compilation_config", None) or sfc
        sfc = getattr(sfc, "static_forward_context", None) if sfc is not None else None
        if sfc is None:
            sfc = getattr(runner, "static_forward_context", None)
        kvcc = getattr(runner, "kv_cache_config", None)
        ib = getattr(runner, "input_batch", None)
        if kvcc is None or sfc is None or ib is None:
            return
        try:
            group0 = kvcc.kv_cache_groups[0]
            spec = group0.kv_cache_spec
            bs = int(getattr(spec, "block_size", 128) or 128)
            kv_heads = int(getattr(spec, "num_kv_heads", 0) or 0)
            head_size = int(getattr(spec, "head_size", 0) or 0)
            layer_names = list(group0.layer_names)
        except Exception as exc:
            registry.record("skipped_error")
            logger.warning("cannot read kv_cache_spec: %s", exc)
            return
        row_map = getattr(getattr(ib, "req_id_to_index", None), "get", None)
        bt = getattr(ib, "block_table", None)
        if row_map is None or bt is None or not layer_names:
            return
        num_heads = self._num_heads(runner, layer_names[0], kv_heads)
        num_hidden_layers = self._num_hidden_layers(runner)
        budget = max(1, envs.mid_prefill_budget())
        refresh = max(1, envs.mid_prefill_refresh())
        for req_id in info.req_ids:
            before = info.num_computed_before.get(req_id, 0)
            prompt = info.num_prompt.get(req_id, 0)
            n_sched = info.num_scheduled.get(req_id, 0)
            if not (before < prompt and before + n_sched >= budget):
                continue
            true_len = before + n_sched
            anchor = self.mid_anchors.get(req_id, 0)
            if anchor > 0 and true_len - anchor < refresh:
                continue
            if true_len < envs.min_prompt_tokens():
                continue
            try:
                self._compress_at_length(runner, sfc, ib, bt, row_map, layer_names, bs,
                                         kv_heads, head_size, num_heads, num_hidden_layers,
                                         req_id, true_len, self.press,
                                         keep_capture=True, kind="mid")
                self.mid_anchors[req_id] = true_len
                registry.record("mid_prefilled")
            except Exception as exc:  # fail-soft per request
                registry.record("skipped_error")
                logger.warning("mid-prefill compress %s failed: %s", req_id, exc)

    def _num_hidden_layers(self, runner) -> int:
        try:
            mc = getattr(runner, "model_config", None)
            if mc is None:
                mc = getattr(getattr(runner, "vllm_config", None), "model_config", None)
            hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
            n = getattr(hf, "num_hidden_layers", 0)
            if n:
                return int(n)
        except Exception:
            pass
        return 0

    def _num_cache_blocks(self, sfc, layer_names) -> int:
        """Number of KV cache blocks per layer (from the bound KV cache)."""
        try:
            for layer_name in layer_names:
                mod = sfc.get(layer_name)
                if mod is not None:
                    kc = getattr(mod, "kv_cache", None)
                    if kc and kc[0] is not None:
                        return int(kc[0].shape[0])
        except Exception:
            pass
        return 0

    def _num_cache_blocks_from_runner(self, runner) -> int:
        try:
            sfc = getattr(runner, "compilation_config", None)
            sfc = getattr(sfc, "static_forward_context", None)
            if sfc:
                for mod in sfc.values():
                    kc = getattr(mod, "kv_cache", None)
                    if kc and kc[0] is not None:
                        return int(kc[0].shape[0])
        except Exception:
            pass
        return 0

    def _num_heads(self, runner, layer_name: str, kv_heads: int) -> int:
        try:
            mod = runner.compilation_config.static_forward_context.get(layer_name)
            if mod is not None:
                h = getattr(mod, "num_heads", 0) or getattr(mod, "num_attention_heads", 0)
                if h:
                    return int(h)
        except Exception:
            pass
        return kv_heads * 4  # unknown fallback (GQA=4 default guess)

    def _compress_at_length(self, runner, sfc, ib, bt, row_map, layer_names, bs,
                            kv_heads, head_size, num_heads, num_hidden_layers,
                            req_id, anchor_len, press, keep_capture=False,
                            kind="completion") -> None:
        """Score + record layout anchored at `anchor_len` true tokens.

        kind="completion": anchor == prompt length (request done prefilling).
        kind="mid": anchor < prompt length (progressive compression, view mode).
        """
        row_idx = row_map(req_id)
        if row_idx is None:
            return
        orig_len = int(anchor_len)
        if orig_len < envs.min_prompt_tokens():
            registry.record("skipped_short")
            return
        m = (orig_len + bs - 1) // bs
        row = bt[0].block_table.np[row_idx, :m].copy()
        if row.shape[0] < m or int(row[m - 1]) == 0:
            registry.record("skipped_error")
            return
        # G11 guard: validate the row and the derived slots on CPU BEFORE any
        # device gather. A bad block id would reach the NPU gather kernel and
        # poison the device stream (Python try/except cannot recover; the
        # worker crashes at the next sync point). Real-machine: AIV
        # "Index out of range ... index value 341748 exceeds bounds 923136".
        num_blocks = self._num_cache_blocks(sfc, layer_names)
        if num_blocks:
            if int(row.min()) < 0 or int(row.max()) >= num_blocks:
                registry.record("skipped_bad_row")
                logger.warning("req %s bad block row anchor=%d m=%d ids[min=%d max=%d] "
                               "num_blocks=%d - skipped (G11 guard)",
                               req_id, orig_len, m, int(row.min()), int(row.max()), num_blocks)
                return
        device = runner.device
        # slots of the TRUE positions 0..orig_len-1 ONLY (never include the
        # last block's padding: scores must not be polluted by padding KV)
        slots = np.arange(orig_len, dtype=np.int64)
        block_ids = np.repeat(row, bs)[:orig_len]
        slots = block_ids * bs + (slots % bs)
        if num_blocks and int(slots.max()) >= num_blocks * bs:
            registry.record("skipped_bad_row")
            logger.warning("req %s slot overflow anchor=%d max_slot=%d num_slots=%d - skipped (G11 guard)",
                           req_id, orig_len, int(slots.max()), num_blocks * bs)
            return
        slot_t = torch.from_numpy(slots.astype(np.int64)).to(device)
        rc = self.requests.get(req_id)
        if self.mode == "view":
            self._score_and_record_view(sfc, layer_names, bs, kv_heads, head_size, num_heads,
                                        num_hidden_layers, req_id, orig_len, m, slot_t, rc,
                                        press, device, row, keep_capture=keep_capture, kind=kind)
        else:
            self._score_and_record_compact(sfc, layer_names, bs, kv_heads, head_size, num_heads,
                                           num_hidden_layers, req_id, orig_len, m, slot_t, rc,
                                           press, device, row)

    def _score_and_record_view(self, sfc, layer_names, bs, kv_heads, head_size, num_heads,
                               num_hidden_layers, req_id, orig_len, m, slot_t, rc, press,
                               device, row, keep_capture=False, kind="completion") -> None:
        per_layer: dict = {}
        for layer_name in layer_names:
            mod = sfc.get(layer_name)
            if mod is None:
                continue
            kc = getattr(mod, "kv_cache", None)
            if kc is None or not kc:
                continue
            key_cache = kc[0]
            flat = key_cache.view(-1, kv_heads, head_size)
            keys = flat[slot_t]  # (orig_len, kvh, hd)
            qbuf = rc.queries.get(layer_name) if rc else None
            queries = qbuf[:rc.captured_q.get(layer_name, 0)] if qbuf is not None else None
            hidden = rc.hidden.get(layer_name) if rc else None
            if hidden is not None:
                hidden._attn_module = mod  # for ExpectedAttention/CriticalKV
            ctx = _layer_ctx(layer_name, num_heads, kv_heads, head_size, num_hidden_layers)
            scores = press.score(ctx, queries, keys, None, hidden)
            n_kept_blocks = press.budget_blocks(ctx, orig_len, bs)
            n_blocks = m
            if n_kept_blocks >= n_blocks:
                continue
            block_scores, keep_blocks = _block_keep(scores, n_blocks, bs, n_kept_blocks)
            # keep_blocks/block_scores are DEVICE tensors on the real NPU:
            # never call .numpy() on them directly (raises "can't convert npu
            # device type tensor to numpy" - real-machine bug); use _as_numpy.
            bl = _as_numpy(keep_blocks).astype(np.int64)
            # The last block must stay visible when partial (new tokens land
            # in its padding slots). Make room by dropping the lowest-scored
            # KEPT block when at budget.
            if orig_len % bs != 0 and (m - 1) not in bl.tolist():
                bscores = _as_numpy(block_scores)
                drop_idx = int(np.argmin(bscores[bl]))
                bl = np.delete(bl, drop_idx)
                bl = np.append(bl, m - 1)
                bl.sort()
            kept_ids = row[bl]
            n_kept = sum(min(bs, orig_len - int(b) * bs) for b in bl.tolist())
            per_layer[layer_name] = {
                "kept_ids": kept_ids.astype(np.int64),
                "n_kept_blocks": int(len(bl)),
                "n_kept": n_kept,
            }
            registry.record("compressed")
        if not per_layer:
            return
        if envs.dry_run():
            registry.record("dry_run")
            return
        from kvpress_ascend.kvcore import ViewLayout
        layouts = self.layouts.setdefault(req_id, {})
        for layer_name, pl in per_layer.items():
            layouts[layer_name] = ViewLayout.build(
                req_id, layer_name, orig_len, pl["kept_ids"], pl["n_kept"], bs)
        logger.debug("req %s %s-compressed: layers=%d anchor=%d (view mode)",
                     req_id, kind, len(per_layer), orig_len)
        # free captured buffers of this request unless mid-prefill needs them
        if not keep_capture:
            self.requests.pop(req_id, None)

    def _score_and_record_compact(self, sfc, layer_names, bs, kv_heads, head_size, num_heads,
                                  num_hidden_layers, req_id, orig_len, m, slot_t, rc, press, device, row) -> None:
        # uniform layout: budgets are averaged across layers (shared slot mapping)
        budgets = []
        for layer_name in layer_names:
            mod = sfc.get(layer_name)
            if mod is None:
                continue
            ctx = _layer_ctx(layer_name, num_heads, kv_heads, head_size, num_hidden_layers)
            budgets.append(press.budget_tokens(ctx, orig_len))
        if not budgets:
            return
        n_kept = max(1, min(orig_len - 1, round(sum(budgets) / len(budgets))))
        from kvpress_ascend.kvcore import CompactLayout
        layout = CompactLayout.build(req_id, orig_len, n_kept, bs)
        if not layout.check_slack():
            registry.record("skipped_error")
            return
        if envs.dry_run():
            registry.record("dry_run")
            return
        # physical packing: for each layer write kept (K, V) into tail blocks
        target_slots = np.arange(n_kept, dtype=np.int64)
        rew = layout.rewrite_row(row)  # (m,) tail-k first
        t_slots = np.repeat(rew[: layout.k], bs)[:n_kept] * bs + (target_slots % bs)
        t_slot_t = torch.from_numpy(t_slots.astype(np.int64)).to(device)
        for layer_name in layer_names:
            mod = sfc.get(layer_name)
            if mod is None:
                continue
            kc = getattr(mod, "kv_cache", None)
            if kc is None or not kc:
                continue
            key_cache, value_cache = kc[0], kc[1]
            kflat = key_cache.view(-1, kv_heads, head_size)
            vflat = value_cache.view(-1, kv_heads, head_size)
            keys = kflat[slot_t]
            values = vflat[slot_t]
            qbuf = rc.queries.get(layer_name) if rc else None
            queries = qbuf[:rc.captured_q.get(layer_name, 0)] if qbuf is not None else None
            hidden = rc.hidden.get(layer_name) if rc else None
            if hidden is not None:
                hidden._attn_module = mod
            ctx = _layer_ctx(layer_name, num_heads, kv_heads, head_size, num_hidden_layers)
            scores = press.score(ctx, queries, keys, values, hidden)
            keep = _token_keep(scores, n_kept)
            keep_t = torch.as_tensor(keep, device=device) if not torch.is_tensor(keep) else keep
            kflat[t_slot_t] = keys[keep_t]
            vflat[t_slot_t] = values[keep_t]
            registry.record("compressed")
        self.compact[req_id] = layout
        self.requests.pop(req_id, None)
        logger.debug("req %s compacted: orig=%d kept=%d k=%d slack=%d",
                     req_id, orig_len, n_kept, layout.k, layout.slack)

    def _progress_summary(self, info: StepInfo) -> None:
        """Periodic INFO summary so a stuck/never-completing prefill phase is
        visible without debug logs."""
        interval = envs.progress_log_interval()
        if interval <= 0 or not info.req_ids or info.step_id % interval != 0:
            return
        prefilling = [r for r in info.req_ids
                      if info.num_computed_before.get(r, 0) < info.num_prompt.get(r, 0)]
        remaining = [info.num_prompt.get(r, 0) - info.num_computed_before.get(r, 0)
                     for r in prefilling]
        logger.info("progress: step=%d reqs=%d prefilling=%d completed_total=%d "
                    "min_remaining=%s max_remaining=%s mid_anchored=%d",
                    info.step_id, len(info.req_ids), len(prefilling), self._completed_total,
                    min(remaining) if remaining else 0,
                    max(remaining) if remaining else 0,
                    len(self.mid_anchors))

    def _heartbeat(self, runner) -> None:
        info = self.step
        if info is None:
            return
        layer_count = 0
        for per_req in self.layouts.values():
            layer_count = max(layer_count, len(per_req))
        core = {
            "press": getattr(self.press, "name", "none") if self.press else "none",
            "ratio": getattr(self.press, "compression_ratio", 0.0) if self.press else 0.0,
            "window": getattr(self.press, "window_size", envs.window_size()) if self.press else envs.window_size(),
            "sink": getattr(self.press, "n_sink", envs.sink_size()) if self.press else envs.sink_size(),
            "mode": self.mode,
            "layers": layer_count,
            "prefilling": self._active_prefills,
            "completed": len(info.completed_prefill),
            "active_compressed": len(self.layouts),
            "mid_anchored": len(self.mid_anchors),
            "attn_state": info.attn_state_name,
        }
        # Always show the key counters (with zeros) so "nothing happened" is
        # visible at a glance instead of an empty stats section.
        stats = registry.stats_snapshot()
        for key in ("compressed", "mid_prefilled", "skipped_short", "skipped_error",
                    "dry_run", "activation", "layout_dropped_recompute"):
            stats.setdefault(key, 0)
        registry.heartbeat(info.step_id, core, stats=stats)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _attn_state_name(state) -> str:
    if state is None:
        return ""
    return getattr(state, "name", state) if not isinstance(state, str) else state


def _as_numpy(t):
    """Device-safe conversion: tensors may live on npu/cuda and .numpy()
    raises there - always .cpu() first (real-machine bug, see RTR)."""
    if torch is not None and torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _qsl_from_meta(meta):
    qsl = getattr(meta, "actual_seq_lengths_q", None)
    if qsl is not None:
        return list(qsl)
    qslc = getattr(meta, "query_start_loc_cpu", None)
    if qslc is not None:
        arr = qslc.tolist()
        return [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
    return None


def _cumsum(xs) -> list[int]:
    out = [0]
    for x in xs:
        out.append(out[-1] + int(x))
    return out


def _layer_ctx(layer_name: str, num_heads: int, kv_heads: int, head_size: int,
               num_hidden_layers: int = 0):
    from kvpress_ascend.presses import LayerCtx

    idx = 0
    try:
        idx = int(layer_name.split(".")[2])
    except Exception:
        pass
    return LayerCtx(
        layer_name=layer_name,
        layer_idx=idx,
        num_hidden_layers=max(1, num_hidden_layers),
        num_heads=num_heads,
        num_kv_heads=max(1, kv_heads),
        head_size=head_size,
    )


def _block_keep(scores, n_blocks: int, bs: int, n_kept_blocks: int):
    """Return (block_scores, sorted kept block indices)."""
    seq = scores.shape[0]
    pad = (-seq) % bs
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad))
    blocks = scores.view(n_blocks, bs).mean(dim=1)
    idx = blocks.topk(n_kept_blocks, dim=-1).indices.sort().values
    return blocks, idx


def _token_keep(scores, n_kept: int):
    return scores.topk(n_kept, dim=-1).indices.sort().values
