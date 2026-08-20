"""Heartbeat + seam probes: the user-visible switch verification.

Every inference step must print whether the patches entered their core code
(seam probes) and with which core parameters (press/ratio/window/sink).
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvpress_ascend import registry  # noqa: E402
from kvpress_ascend.log import logger  # noqa: E402


class TestSeamRegistry:
    def test_mark_and_summary(self):
        registry.mark_installed("S1_backend_forward", True)
        registry.mark_hit("S1_backend_forward")
        registry.mark_hit("S4_attn_metadata")
        s = registry.seams_summary()
        assert s.startswith("seams=")
        assert "S1_backend_forward" not in s or "FAIL" not in s

    def test_fail_seam_reported(self):
        registry.mark_installed("S1b_c8_forward", False)
        s = registry.seams_summary()
        assert "FAIL=S1b_c8_forward" in s
        registry.mark_installed("S1b_c8_forward", True)


class TestHeartbeat:
    def test_heartbeat_emits_once_per_step(self, caplog):
        registry._last_step = -1  # reset cross-test step guard
        with caplog.at_level(logging.INFO, logger="kvpress-ascend"):
            registry.heartbeat(1, {"press": "snapkv", "ratio": 0.5, "window": 64, "sink": 4},
                               stats={"compressed": 3})
            registry.heartbeat(1, {"press": "snapkv"})  # same step -> suppressed
            registry.heartbeat(2, {"press": "knorm", "ratio": 0.3, "window": 0, "sink": 0})
        lines = [r.message for r in caplog.records]
        step1 = [l for l in lines if l.startswith("step=1")]
        assert len(step1) == 1
        assert "press=snapkv" in step1[0]
        assert "ratio=0.5" in step1[0]
        assert "compressed=3" in step1[0]
        step2 = [l for l in lines if l.startswith("step=2")]
        assert len(step2) == 1
        assert "press=knorm" in step2[0]

    def test_activation_summary_logs(self, caplog):
        with caplog.at_level(logging.INFO, logger="kvpress-ascend"):
            registry.log_activation_summary()
        assert any("seam" in r.message for r in caplog.records)
