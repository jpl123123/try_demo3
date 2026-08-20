"""Seam probes, statistics counters and the per-step heartbeat for
kvpress-ascend.

The heartbeat answers the user's core requirement: *every inference step,
print whether the patches actually entered their core code and with which
core parameters*.
"""

from __future__ import annotations

import threading
from collections import Counter

from kvpress_ascend import envs
from kvpress_ascend.log import logger

# --------------------------------------------------------------------------
# Seam probes
# --------------------------------------------------------------------------

# name -> (description, installed: bool, hit: bool)
SEAMS: dict[str, dict] = {
    "S1_backend_forward": {"desc": "AscendAttentionBackendImpl.forward capture hook", "installed": False, "hit": False},
    "S1b_c8_forward": {"desc": "AscendC8AttentionBackendImpl.forward capture hook", "installed": False, "hit": False},
    "S2_attn_module": {"desc": "vllm Attention.forward capture hook", "installed": False, "hit": False},
    "S3_prepare_inputs": {"desc": "NPUModelRunner._prepare_inputs context hook", "installed": False, "hit": False},
    "S4_attn_metadata": {"desc": "NPUModelRunner._build_attention_metadata view rewrite", "installed": False, "hit": False},
    "S5_execute_model": {"desc": "NPUModelRunner.execute_model pre/post hook", "installed": False, "hit": False},
    "S6_compress_pass": {"desc": "prefill-completion compression pass", "installed": False, "hit": False},
    "S7_slot_mapping": {"desc": "MultiGroupBlockTable.compute_slot_mapping shift (compact)", "installed": False, "hit": False},
}

_lock = threading.Lock()


def mark_installed(name: str, ok: bool = True) -> None:
    with _lock:
        if name in SEAMS:
            SEAMS[name]["installed"] = ok


def mark_hit(name: str) -> None:
    with _lock:
        if name in SEAMS:
            SEAMS[name]["hit"] = True


def seams_summary() -> str:
    with _lock:
        installed = sum(1 for s in SEAMS.values() if s["installed"])
        hit = sum(1 for s in SEAMS.values() if s["hit"])
        failed = [n for n, s in SEAMS.items() if not s["installed"]]
        parts = [f"seams={hit}/{len(SEAMS)}"]
        if failed:
            parts.append("FAIL=" + ",".join(failed))
        return " ".join(parts)


def log_activation_summary() -> None:
    """One-shot summary at activation: which seams are live."""
    with _lock:
        entries = [(name, s["installed"], s["desc"]) for name, s in SEAMS.items()]
    for name, installed, desc in entries:
        status = "OK " if installed else "FAIL"
        logger.info("seam %s [%s] %s", status, name, desc)
    logger.info("activation summary: %s", seams_summary())


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

_stats_lock = threading.Lock()
_stats: Counter = Counter()


def record(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] += n


def stats_snapshot() -> dict:
    with _stats_lock:
        return dict(_stats)


# --------------------------------------------------------------------------
# Heartbeat
# --------------------------------------------------------------------------

_last_step = -1


def heartbeat(step_id: int, core_params: dict, stats: dict | None = None) -> None:
    """One line per inference step (execute_model invocation).

    core_params must contain the press name / ratio / window / sink etc. so a
    missing or FAIL heartbeat is instantly diagnosable.
    """
    global _last_step
    with _lock:
        if step_id == _last_step:
            return  # already emitted for this step
        _last_step = step_id
    if not envs.step_log():
        return
    params = " ".join(f"{k}={v}" for k, v in core_params.items())
    stat = ""
    if stats is not None:
        stat = " " + " ".join(f"{k}={v}" for k, v in stats.items())
    logger.info("step=%s %s %s%s", step_id, seams_summary(), params, stat)
