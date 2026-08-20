"""Activation policy: deliberate deferral between the two packages must log
INFO (not ERROR "FAILED seams") and set engine.DEFERRED_REASON."""

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SQUEEZE_ROOT = ROOT.parent / "SqueezeAttention-ascend"
for p in (ROOT, SQUEEZE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class TestDeferral:
    def test_kvpress_defers_when_policy_squeeze(self, monkeypatch):
        monkeypatch.setenv("KVPRESS_ASCEND_POLICY", "squeeze")
        import kvpress_ascend.engine as eng
        eng.DEFERRED_REASON = None
        assert eng.install() is False
        assert eng.DEFERRED_REASON
        assert "squeeze" in eng.DEFERRED_REASON

    def test_squeeze_defers_when_kvpress_active(self, monkeypatch):
        monkeypatch.setenv("kvpress", "1")
        import squeeze_ascend.engine as eng
        eng.DEFERRED_REASON = None
        assert eng.install() is False
        assert eng.DEFERRED_REASON
        assert "kvpress" in eng.DEFERRED_REASON

    def test_squeeze_import_time_deferral_no_error(self, monkeypatch, caplog):
        """The real user scenario: both exports set at interpreter startup ->
        squeeze must defer quietly at import time, no ERROR."""
        monkeypatch.setenv("squeeze", "1")
        monkeypatch.setenv("kvpress", "1")
        with caplog.at_level(logging.INFO, logger="squeeze-ascend"):
            import importlib
            import squeeze_ascend
            import squeeze_ascend.engine as eng
            importlib.reload(eng)  # reset module state
            importlib.reload(squeeze_ascend)
        assert getattr(squeeze_ascend.apply, "_applied", False)
        assert eng.DEFERRED_REASON
        messages = [r.message for r in caplog.records]
        assert any("deferred" in m for m in messages), messages
        assert not any("FAILED seams" in m for m in messages), messages

    def test_kvpress_apply_deferral_message(self, monkeypatch, caplog):
        import kvpress_ascend
        import kvpress_ascend.engine as eng
        kvpress_ascend.apply._applied = False  # type: ignore[attr-defined]
        eng.DEFERRED_REASON = None
        monkeypatch.setenv("KVPRESS_ASCEND_ENABLE", "1")
        monkeypatch.setenv("KVPRESS_ASCEND_POLICY", "squeeze")
        with caplog.at_level(logging.INFO, logger="kvpress-ascend"):
            ok = kvpress_ascend.apply()
        assert ok is False
        messages = [r.message for r in caplog.records]
        assert any("deferred:" in m for m in messages), messages
        assert not any("FAILED seams" in m for m in messages), messages
