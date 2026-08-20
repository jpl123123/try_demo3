"""kvpress-ascend: monkeypatch adapter of kvpress KV-cache compression onto
vllm-ascend v0.23.0.

Activation (one of):
    export kvpress=1
    export kvpress_ascend=1
    export exportkvpress=1

When NOT activated this package imports nothing from torch / vllm / vllm_ascend
(lazy by design; the .pth import must stay side-effect free).

The patches are applied per process (API server, engine-core and worker
subprocesses all import site-packages .pth files at interpreter startup), so
every process that runs the model forward gets the same hooks.
"""

from __future__ import annotations

import os as _os

_ACTIVATION_NAMES = ("kvpress", "kvpress_ascend", "exportkvpress")


def is_enabled() -> bool:
    """Master switch. Any truthy value of one of the activation env vars."""
    for name in _ACTIVATION_NAMES:
        val = _os.environ.get(name)
        if val is not None and str(val).strip().lower() not in ("", "0", "false", "no"):
            return True
    # Explicit per-package switch (works even without the short env names).
    val = _os.environ.get("KVPRESS_ASCEND_ENABLE")
    if val is not None and str(val).strip().lower() not in ("", "0", "false", "no"):
        return True
    return False


def apply() -> bool:
    """Apply all monkeypatches. Returns True if fully applied.

    Lazy: torch/vllm imports happen only here, wrapped fail-soft so a broken
    seam never takes the serving process down.
    """
    if getattr(apply, "_applied", False):
        return True
    apply._applied = True  # noqa: B010 - idempotence marker

    from kvpress_ascend import envs, log, registry
    from kvpress_ascend.engine import install

    if not is_enabled():
        log.logger.info("kvpress-ascend not activated (export kvpress=1 to enable) — no patches installed")
        return False

    log.logger.info("kvpress-ascend activated, installing monkeypatches (vllm-ascend v0.23.0 adapter)")
    try:
        ok = install()
    except Exception as exc:  # noqa: BLE001 - fail-soft at activation
        log.logger.error("kvpress-ascend activation failed: %s", exc)
        registry.record("activation_error", str(exc))
        return False

    if ok:
        registry.log_activation_summary()
    else:
        log.logger.error("kvpress-ascend installed with FAILED seams — see seam summary above")
    return ok


# The .pth file imports this module; we only apply when enabled so that
# disabled processes pay zero import cost (no torch/vllm import at all).
if is_enabled():
    apply()

__all__ = ["apply", "is_enabled"]
