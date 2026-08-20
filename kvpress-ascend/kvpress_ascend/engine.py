"""All monkeypatches for kvpress-ascend. Fail-soft everywhere: a broken seam
logs and disables itself; the serving process keeps running uncompressed.

Patch targets (verified against vllm-ascend v0.23.0 source):
  S1  AscendAttentionBackendImpl.forward / AscendC8AttentionBackendImpl.forward
  S2  vllm.model_executor.layers.attention.Attention.forward
  S3  NPUModelRunner._prepare_inputs            (entry: compact row rewrite)
  S4  NPUModelRunner._build_attention_metadata  (post: per-layer view rewrite)
  S5  NPUModelRunner.execute_model              (pre/post: capture + compression)
  S7  MultiGroupBlockTable.compute_slot_mapping (compact mode: positions shift)
"""

from __future__ import annotations

import functools

from kvpress_ascend import envs, registry
from kvpress_ascend.log import logger

ctx = None  # CaptureManager singleton, created by install()


def _defuse_import_cycle() -> bool:
    """Defuse the known vllm_ascend ops import cycle (see skill bug class #9)
    by importing the safe entry first. Failure aborts installation cleanly."""
    try:
        import vllm_ascend  # noqa: F401  (lightweight __init__)
        import vllm_ascend.ops.fused_moe.fused_moe  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("cannot import vllm_ascend (cycle defuse failed): %s", exc)
        return False


def _wrap(name: str, obj, attr: str, fn) -> bool:
    try:
        original = getattr(obj, attr)
        if getattr(original, "_kvpress_ascend_wrapped", False):
            return True  # idempotent
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return fn(original, *args, **kwargs)
        wrapper._kvpress_ascend_wrapped = True  # type: ignore[attr-defined]
        wrapper._kvpress_ascend_orig = original  # type: ignore[attr-defined]
        setattr(obj, attr, wrapper)
        registry.mark_installed(name)
        logger.debug("patched %s.%s", getattr(obj, "__name__", obj), attr)
        return True
    except Exception as exc:  # noqa: BLE001
        registry.mark_installed(name, ok=False)
        logger.error("patch %s failed: %s", name, exc)
        return False


def install() -> bool:
    """Import targets and apply every seam. Returns True if all seams OK."""
    global ctx
    if envs.policy() == "squeeze":
        logger.warning("KVPRESS_ASCEND_POLICY=squeeze -> kvpress-ascend defers to SqueezeAttention-ascend, "
                       "no patches installed")
        return False
    if not _defuse_import_cycle():
        return False

    from kvpress_ascend.capture import CaptureManager

    ctx = CaptureManager()
    ctx.mode = envs.mode()
    if ctx.mode not in ("view", "compact"):
        logger.warning("unknown KVPRESS_ASCEND_MODE=%s, falling back to view", ctx.mode)
        ctx.mode = "view"
    if ctx.mode == "compact" and envs.dry_run():
        logger.warning("compact mode ignores DRY_RUN (physical writes are the point)")
    from kvpress_ascend.presses import build_press

    ctx.press = build_press(envs.press_name(), envs.compression_ratio(),
                            envs.window_size(), envs.sink_size(), envs.kernel_size())
    ctx.capture_w = envs.effective_capture_window()
    logger.info("press=%s ratio=%.3f window=%d sink=%d mode=%s capture_w=%d",
                ctx.press.name, ctx.press.compression_ratio, envs.window_size(),
                envs.sink_size(), ctx.mode, ctx.capture_w)

    results = []

    # ---- S5 / S3 / S4: model runner -------------------------------------
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        def _exec_model(orig, self, *args, **kwargs):
            scheduler_output = args[0] if args else kwargs.get("scheduler_output")
            try:
                ctx.on_step_begin(self, scheduler_output)
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("step-begin hook failed: %s", exc)
            try:
                out = orig(self, *args, **kwargs)
            finally:
                try:
                    ctx.on_step_end(self, scheduler_output)
                except Exception as exc:  # noqa: BLE001
                    registry.record("skipped_error")
                    logger.warning("step-end hook failed: %s", exc)
            return out

        def _prepare_inputs(orig, self, *args, **kwargs):
            try:
                ctx.on_prepare_inputs_entry(self)
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("prepare-inputs hook failed: %s", exc)
            return orig(self, *args, **kwargs)

        def _build_attn_meta(orig, self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            try:
                if not kwargs.get("for_cudagraph_capture", False) and not _is_capturing():
                    attn_metadata, cm = out
                    ctx.on_metadata_built(self, attn_metadata, cm)
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("metadata hook failed: %s", exc)
            return out

        results.append(_wrap("S5_execute_model", NPUModelRunner, "execute_model", _exec_model))
        results.append(_wrap("S3_prepare_inputs", NPUModelRunner, "_prepare_inputs", _prepare_inputs))
        results.append(_wrap("S4_attn_metadata", NPUModelRunner, "_build_attention_metadata", _build_attn_meta))
    except Exception as exc:  # noqa: BLE001
        logger.error("model runner import failed: %s", exc)
        for name in ("S5_execute_model", "S3_prepare_inputs", "S4_attn_metadata"):
            registry.mark_installed(name, ok=False)

    # ---- S1: attention backend forward (query capture) -------------------
    try:
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionBackendImpl,
            AscendC8AttentionBackendImpl,
        )

        def _backend_forward(orig, self, layer, query, key, value, kv_cache,
                             attn_metadata, output=None, *args, **kwargs):
            try:
                if query is not None and not _is_capturing():
                    layer_name = getattr(layer, "layer_name", None) or getattr(self, "_layer_name", None)
                    if layer_name:
                        ctx.on_backend_forward(layer_name, query, attn_metadata, _is_draft())
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("backend capture failed: %s", exc)
            return orig(self, layer, query, key, value, kv_cache, attn_metadata,
                        output, *args, **kwargs)

        results.append(_wrap("S1_backend_forward", AscendAttentionBackendImpl, "forward", _backend_forward))
        results.append(_wrap("S1b_c8_forward", AscendC8AttentionBackendImpl, "forward", _backend_forward))
    except Exception as exc:  # noqa: BLE001
        logger.error("attention backend import failed: %s", exc)
        registry.mark_installed("S1_backend_forward", ok=False)

    # ---- S2: vllm Attention module (hidden capture) ----------------------
    try:
        from vllm.model_executor.layers.attention import Attention

        def _attn_forward(orig, self, layer, hidden_states, *args, **kwargs):
            try:
                if hidden_states is not None and not _is_capturing():
                    layer_name = getattr(layer, "layer_name", None)
                    if layer_name:
                        ctx.on_attn_module(layer_name, hidden_states, _is_draft())
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("attn-module capture failed: %s", exc)
            return orig(self, layer, hidden_states, *args, **kwargs)

        results.append(_wrap("S2_attn_module", Attention, "forward", _attn_forward))
    except Exception as exc:  # noqa: BLE001
        logger.error("Attention import failed: %s", exc)
        registry.mark_installed("S2_attn_module", ok=False)

    # ---- S7: slot mapping positions shift (compact mode only) ------------
    try:
        from vllm_ascend.worker.block_table import MultiGroupBlockTable

        def _compute_slot_mapping(orig, self, num_reqs, query_start_loc, positions, *args, **kwargs):
            try:
                if ctx.mode == "compact":
                    _shift_positions(num_reqs, query_start_loc, positions)
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("slot-mapping shift failed: %s", exc)
            return orig(self, num_reqs, query_start_loc, positions, *args, **kwargs)

        results.append(_wrap("S7_slot_mapping", MultiGroupBlockTable, "compute_slot_mapping", _compute_slot_mapping))
    except Exception as exc:  # noqa: BLE001
        logger.error("block table import failed: %s", exc)
        registry.mark_installed("S7_slot_mapping", ok=False)

    ok = all(results)
    registry.record("activation", 1 if ok else 0)
    return ok


def _shift_positions(num_reqs: int, query_start_loc, positions) -> None:
    """Compact mode: subtract per-request delta from the positions tensor
    (device side, no .item() in the hot path)."""
    import numpy as np
    import torch

    if positions is None or positions.numel() == 0 or ctx is None:
        return
    info = ctx.step
    if info is None or not info.req_ids or not ctx.compact:
        return
    deltas = np.zeros(num_reqs, dtype=np.int64)
    any_delta = False
    for i, req_id in enumerate(info.req_ids[:num_reqs]):
        lay = ctx.compact.get(req_id)
        if lay is not None:
            deltas[i] = lay.delta
            any_delta = True
    if not any_delta:
        return
    delta_t = torch.from_numpy(deltas).to(positions.device)
    qsl = query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]
    req_indices = torch.repeat_interleave(
        torch.arange(num_reqs, device=positions.device), qsl, output_size=positions.numel()
    )
    if req_indices.shape[0] != positions.numel():
        return  # layout mismatch -> leave untouched (fail-soft)
    positions.sub_(delta_t[req_indices])


def _is_draft() -> bool:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        return bool(getattr(_EXTRA_CTX, "is_draft_model", False))
    except Exception:
        return False


def _is_capturing() -> bool:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        return bool(getattr(_EXTRA_CTX, "capturing", False))
    except Exception:
        return False
