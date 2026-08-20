"""SqueezeAttention-ascend: monkeypatch adapter of SqueezeAttention's
per-layer KV sliding windows onto vllm-ascend v0.23.0.

Activation (one of):
    export squeeze=1
    export squeezeattention=1
    export squeeze_ascend=1
    export exportsqueeze=1

When NOT activated this package imports nothing from torch / vllm /
vllm_ascend (lazy; the .pth import must stay side-effect free).
"""

from __future__ import annotations

import os as _os

_ACTIVATION_NAMES = ("squeeze", "squeezeattention", "squeeze_ascend", "exportsqueeze")


def is_enabled() -> bool:
    for name in _ACTIVATION_NAMES:
        val = _os.environ.get(name)
        if val is not None and str(val).strip().lower() not in ("", "0", "false", "no"):
            return True
    val = _os.environ.get("SQUEEZE_ASCEND_ENABLE")
    if val is not None and str(val).strip().lower() not in ("", "0", "false", "no"):
        return True
    return False


def apply() -> bool:
    if getattr(apply, "_applied", False):
        return True
    apply._applied = True  # noqa: B010

    from squeeze_ascend import log, registry
    from squeeze_ascend.engine import install

    if not is_enabled():
        log.logger.info("SqueezeAttention-ascend not activated (export squeeze=1 to enable) - no patches installed")
        return False

    log.logger.info("SqueezeAttention-ascend activated, installing monkeypatches (vllm-ascend v0.23.0 adapter)")
    try:
        ok = install()
    except Exception as exc:  # noqa: BLE001
        log.logger.error("SqueezeAttention-ascend activation failed: %s", exc)
        registry.record("activation_error", str(exc))
        return False

    from squeeze_ascend.engine import DEFERRED_REASON

    if ok:
        registry.log_activation_summary()
    elif DEFERRED_REASON:
        # Deliberate deferral (e.g. kvpress-ascend is active): NOT an error.
        log.logger.info("deferred: %s (expected, no patches installed)", DEFERRED_REASON)
    else:
        registry.log_activation_summary()
        log.logger.error("SqueezeAttention-ascend installed with FAILED seams - see summary above")
    return ok


if is_enabled():
    apply()

__all__ = ["apply", "is_enabled"]
