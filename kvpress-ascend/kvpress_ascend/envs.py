"""Centralized environment variables for kvpress-ascend.

All knobs live here with documentation. Values are read lazily (function per
var) so tests can override os.environ freely.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() not in ("", "0", "false", "no")


def _int(val: str | None, default: int) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float(val: str | None, default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def enable() -> bool:
    """Master switch (per-package). Short names are handled in __init__.is_enabled."""
    return _truthy(os.environ.get("KVPRESS_ASCEND_ENABLE"), False)


def log_level() -> str:
    """KVPRESS_ASCEND_LOG=debug|info|warning (default info)."""
    val = os.environ.get("KVPRESS_ASCEND_LOG", "info").strip().lower()
    return val if val in ("debug", "info", "warning") else "info"


def step_log() -> bool:
    """KVPRESS_ASCEND_STEP_LOG: heartbeat per execute_model step (default on)."""
    return _truthy(os.environ.get("KVPRESS_ASCEND_STEP_LOG"), True)


def press_name() -> str:
    """KVPRESS_ASCEND_PRESS: snapkv|knorm|tova|streamingllm|random|pyramidkv|expected_attention|criticalkv (default snapkv)."""
    return os.environ.get("KVPRESS_ASCEND_PRESS", "snapkv").strip().lower()


def compression_ratio() -> float:
    """KVPRESS_ASCEND_RATIO: fraction of KV pairs removed (default 0.5)."""
    return _float(os.environ.get("KVPRESS_ASCEND_RATIO"), 0.5)


def window_size() -> int:
    """KVPRESS_ASCEND_WINDOW: observation window for SnapKV/TOVA-style presses (default 64)."""
    return _int(os.environ.get("KVPRESS_ASCEND_WINDOW"), 64)


def sink_size() -> int:
    """KVPRESS_ASCEND_SINK: sink tokens kept by StreamingLLM-style presses (default 4)."""
    return _int(os.environ.get("KVPRESS_ASCEND_SINK"), 4)


def kernel_size() -> int:
    """KVPRESS_ASCEND_KERNEL: SnapKV avg_pool kernel (default 5)."""
    return _int(os.environ.get("KVPRESS_ASCEND_KERNEL"), 5)


def mode() -> str:
    """KVPRESS_ASCEND_MODE: view (default) — see PLAN.md §2.2. 'compact' reserved."""
    return os.environ.get("KVPRESS_ASCEND_MODE", "view").strip().lower()


def dry_run() -> bool:
    """KVPRESS_ASCEND_DRY_RUN=1: score but never record views (safety rehearsal)."""
    return _truthy(os.environ.get("KVPRESS_ASCEND_DRY_RUN"), False)


def min_prompt_tokens() -> int:
    """KVPRESS_ASCEND_MIN_PROMPT: prompts shorter than this are never compressed (default 512)."""
    return _int(os.environ.get("KVPRESS_ASCEND_MIN_PROMPT"), 512)


def capture_window() -> int:
    """KVPRESS_ASCEND_CAPTURE_WINDOW: per-request query capture capacity (default max(512, window))."""
    return _int(os.environ.get("KVPRESS_ASCEND_CAPTURE_WINDOW"), 0)  # 0 -> auto


def max_tracked_prefills() -> int:
    """KVPRESS_ASCEND_MAX_PREFILLS: cap on concurrently captured prefill requests (default 8)."""
    return _int(os.environ.get("KVPRESS_ASCEND_MAX_PREFILLS"), 8)


def skip_draft_steps() -> bool:
    """KVPRESS_ASCEND_SKIP_DRAFT_STEPS: skip view rewrite when scheduled_spec_decode_tokens present (default off)."""
    return _truthy(os.environ.get("KVPRESS_ASCEND_SKIP_DRAFT_STEPS"), False)


def expected_attention_future_positions() -> int:
    """KVPRESS_ASCEND_EA_FUTURE: ExpectedAttention n_future_positions (default 512)."""
    return _int(os.environ.get("KVPRESS_ASCEND_EA_FUTURE"), 512)


def policy() -> str:
    """KVPRESS_ASCEND_POLICY: when both packages are active, which one wins.
    'kvpress' | 'squeeze' | 'both' (default both)."""
    return os.environ.get("KVPRESS_ASCEND_POLICY", "both").strip().lower()


@lru_cache(maxsize=1)
def effective_capture_window() -> int:
    cw = capture_window()
    return cw if cw > 0 else max(window_size(), 512)


def as_dict() -> dict:
    """Snapshot of all knobs (for heartbeat / debug logs)."""
    return {
        "enable": enable(),
        "press": press_name(),
        "ratio": compression_ratio(),
        "window": window_size(),
        "sink": sink_size(),
        "kernel": kernel_size(),
        "mode": mode(),
        "dry_run": dry_run(),
        "min_prompt": min_prompt_tokens(),
        "capture_window": effective_capture_window(),
        "step_log": step_log(),
        "log_level": log_level(),
        "policy": policy(),
    }
