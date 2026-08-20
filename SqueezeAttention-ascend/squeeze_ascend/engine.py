"""All monkeypatches for SqueezeAttention-ascend. Fail-soft everywhere.

Patch targets (verified against vllm-ascend v0.23.0 source):
  S1  NPUModelRunner.execute_model             (pre/post hooks)
  S2  decoder layer forward (per-instance)     (layer input capture)
  S3  vllm.model_executor.layers.attention.Attention.forward (attn output capture)
  S4  NPUModelRunner._build_attention_metadata (post: window view rewrite)
"""

from __future__ import annotations

import functools

from squeeze_ascend import envs, registry
from squeeze_ascend.log import logger

ctx = None  # WindowManager singleton
_wrapped_layers: set = set()


def _defuse_import_cycle() -> bool:
    try:
        import vllm_ascend  # noqa: F401
        import vllm_ascend.ops.fused_moe.fused_moe  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("cannot import vllm_ascend (cycle defuse failed): %s", exc)
        return False


def _wrap(name: str, obj, attr: str, fn) -> bool:
    try:
        original = getattr(obj, attr)
        if getattr(original, "_squeeze_ascend_wrapped", False):
            return True
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            return fn(original, *args, **kwargs)
        wrapper._squeeze_ascend_wrapped = True  # type: ignore[attr-defined]
        wrapper._squeeze_ascend_orig = original  # type: ignore[attr-defined]
        setattr(obj, attr, wrapper)
        registry.mark_installed(name)
        return True
    except Exception as exc:  # noqa: BLE001
        registry.mark_installed(name, ok=False)
        logger.error("patch %s failed: %s", name, exc)
        return False


def install() -> bool:
    global ctx
    # When both packages are enabled, kvpress wins by default; the user can
    # force this package with SQUEEZE_ASCEND_POLICY=squeeze.
    try:
        import kvpress_ascend as _kvp
        if _kvp.is_enabled() and envs.policy() != "squeeze":
            logger.warning("kvpress-ascend is enabled too: SqueezeAttention-ascend defers "
                           "(set SQUEEZE_ASCEND_POLICY=squeeze to force)")
            return False
    except Exception:
        pass
    if not _defuse_import_cycle():
        return False

    from squeeze_ascend.capture import WindowManager

    ctx = WindowManager()
    results = []

    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        def _exec_model(orig, self, *args, **kwargs):
            scheduler_output = args[0] if args else kwargs.get("scheduler_output")
            try:
                ensure_layer_wrappers(self)
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

        results.append(_wrap("S1_step_begin", NPUModelRunner, "execute_model", _exec_model))
        results.append(_wrap("S4_metadata", NPUModelRunner, "_build_attention_metadata", _build_attn_meta))
        # S5_cluster / S6_step_end are logical seams inside the S1 hooks: mark
        # them installed together with S1 so the heartbeat never false-FAILs.
        registry.mark_installed("S5_cluster")
        registry.mark_installed("S6_step_end")
    except Exception as exc:  # noqa: BLE001
        logger.error("model runner import failed: %s", exc)
        registry.mark_installed("S1_step_begin", ok=False)
        registry.mark_installed("S4_metadata", ok=False)
        registry.mark_installed("S5_cluster", ok=False)
        registry.mark_installed("S6_step_end", ok=False)

    # S3: Attention module class-level wrapper (attn output capture)
    try:
        from vllm.model_executor.layers.attention import Attention

        def _attn_forward(orig, self, layer, hidden_states, *args, **kwargs):
            out = orig(self, layer, hidden_states, *args, **kwargs)
            try:
                if not _is_capturing():
                    layer_name = getattr(layer, "layer_name", None)
                    if layer_name:
                        ctx.on_attn_output(layer_name, hidden_states, out, _is_draft())
            except Exception as exc:  # noqa: BLE001
                registry.record("skipped_error")
                logger.warning("attn-output capture failed: %s", exc)
            return out

        results.append(_wrap("S3_attn_output", Attention, "forward", _attn_forward))
    except Exception as exc:  # noqa: BLE001
        logger.error("Attention import failed: %s", exc)
        registry.mark_installed("S3_attn_output", ok=False)

    logger.info("SqueezeAttention-ascend: ini_size=%.2f class3_ratio=%.2f start_size=%d",
                envs.ini_size(), envs.class3_ratio(), envs.start_size())
    ok = all(results)
    registry.record("activation", 1 if ok else 0)
    return ok


def ensure_layer_wrappers(runner) -> None:
    """Lazily wrap decoder layer forwards (the model loads after activation).
    Only installed once per layer object."""
    global ctx
    try:
        model = runner.get_model() if hasattr(runner, "get_model") else getattr(runner, "model", None)
        if model is None:
            return
        base = getattr(model, "model", model)
        layers = getattr(base, "layers", None)
        if layers is None:
            return
        for layer in layers:
            layer_id = id(layer)
            if layer_id in _wrapped_layers:
                continue
            self_attn = getattr(layer, "self_attn", None)
            layer_name = getattr(self_attn, "layer_name", None)
            if layer_name is None:
                continue
            original = layer.forward
            if getattr(original, "_squeeze_layer_wrapped", False):
                _wrapped_layers.add(layer_id)
                continue

            @functools.wraps(original)
            def wrapper(self_, *args, **kwargs):
                try:
                    hidden = args[0] if args else kwargs.get("hidden_states")
                    if hidden is not None and not _is_capturing():
                        ctx.on_layer_input(getattr(self_.self_attn, "layer_name", ""), hidden, _is_draft())
                except Exception as exc:  # noqa: BLE001
                    registry.record("skipped_error")
                    logger.warning("layer-input capture failed: %s", exc)
                return original(self_, *args, **kwargs)

            wrapper._squeeze_layer_wrapped = True  # type: ignore[attr-defined]
            layer.forward = wrapper  # type: ignore[method-assign]
            _wrapped_layers.add(layer_id)
            registry.mark_installed("S2_layer_input")
    except Exception as exc:  # noqa: BLE001
        logger.debug("layer wrapper install skipped: %s", exc)


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
