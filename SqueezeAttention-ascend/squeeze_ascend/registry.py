"""Seam probes, statistics and the per-step heartbeat for SqueezeAttention-ascend."""

from __future__ import annotations

import threading
from collections import Counter

from squeeze_ascend import envs
from squeeze_ascend.log import logger

SEAMS: dict[str, dict] = {
    "S1_step_begin": {"desc": "execute_model pre hook (counters/completion)", "installed": False, "hit": False},
    "S2_layer_input": {"desc": "decoder layer forward wrapper (layer input capture)", "installed": False, "hit": False},
    "S3_attn_output": {"desc": "vllm Attention.forward wrapper (attn output capture)", "installed": False, "hit": False},
    "S4_metadata": {"desc": "_build_attention_metadata window view rewrite", "installed": False, "hit": False},
    "S5_cluster": {"desc": "prefill-completion KMeans cluster pass", "installed": False, "hit": False},
    "S6_step_end": {"desc": "execute_model post hook (heartbeat)", "installed": False, "hit": False},
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
    with _lock:
        for name, s in SEAMS.items():
            status = "OK " if s["installed"] else "FAIL"
            logger.info("seam %s [%s] %s", status, name, s["desc"])
        logger.info("activation summary: %s", seams_summary())


_stats_lock = threading.Lock()
_stats: Counter = Counter()


def record(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] += n


def stats_snapshot() -> dict:
    with _stats_lock:
        return dict(_stats)


_last_step = -1


def heartbeat(step_id: int, core_params: dict, stats: dict | None = None) -> None:
    global _last_step
    with _lock:
        if step_id == _last_step:
            return
        _last_step = step_id
    if not envs.step_log():
        return
    params = " ".join(f"{k}={v}" for k, v in core_params.items())
    stat = ""
    if stats is not None:
        stat = " " + " ".join(f"{k}={v}" for k, v in stats.items())
    logger.info("step=%s %s %s%s", step_id, seams_summary(), params, stat)
