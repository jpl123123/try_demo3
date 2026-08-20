"""Centralized environment variables for SqueezeAttention-ascend."""

from __future__ import annotations

import os
from functools import lru_cache


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() not in ("", "0", "false", "no")


def _float(val: str | None, default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _int(val: str | None, default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def enable() -> bool:
    return _truthy(os.environ.get("SQUEEZE_ASCEND_ENABLE"), False)


def log_level() -> str:
    val = os.environ.get("SQUEEZE_ASCEND_LOG", "info").strip().lower()
    return val if val in ("debug", "info", "warning") else "info"


def step_log() -> bool:
    """SQUEEZE_ASCEND_STEP_LOG: heartbeat per execute_model step (default on)."""
    return _truthy(os.environ.get("SQUEEZE_ASCEND_STEP_LOG"), True)


def ini_size() -> float:
    """SQUEEZE_ASCEND_INI_SIZE: total KV budget as a fraction of the prompt
    length (across all layers). Default 0.3, matches the paper's ini_size."""
    return _float(os.environ.get("SQUEEZE_ASCEND_INI_SIZE"), 0.3)


def class3_ratio() -> float:
    """SQUEEZE_ASCEND_CLASS3_RATIO: per-layer budget (fraction of prompt) for
    class-3 layers (highest input/output cosine similarity). Default 0.1."""
    return _float(os.environ.get("SQUEEZE_ASCEND_CLASS3_RATIO"), 0.1)


def start_size() -> int:
    """SQUEEZE_ASCEND_START_SIZE: sink tokens always kept. Default 4."""
    return _int(os.environ.get("SQUEEZE_ASCEND_START_SIZE"), 4)


def min_prompt_tokens() -> int:
    """SQUEEZE_ASCEND_MIN_PROMPT: shorter prompts are not windowed (default 256)."""
    return _int(os.environ.get("SQUEEZE_ASCEND_MIN_PROMPT"), 256)


def max_tracked_prefills() -> int:
    return _int(os.environ.get("SQUEEZE_ASCEND_MAX_PREFILLS"), 8)


def dry_run() -> bool:
    """SQUEEZE_ASCEND_DRY_RUN=1: cluster but never rewrite views."""
    return _truthy(os.environ.get("SQUEEZE_ASCEND_DRY_RUN"), False)


def policy() -> str:
    """SQUEEZE_ASCEND_POLICY: 'squeeze' forces this package even when kvpress
    is active (default: kvpress wins when both are enabled)."""
    return os.environ.get("SQUEEZE_ASCEND_POLICY", "auto").strip().lower()


def as_dict() -> dict:
    return {
        "enable": enable(),
        "ini_size": ini_size(),
        "class3_ratio": class3_ratio(),
        "start_size": start_size(),
        "min_prompt": min_prompt_tokens(),
        "step_log": step_log(),
        "log_level": log_level(),
        "policy": policy(),
    }
