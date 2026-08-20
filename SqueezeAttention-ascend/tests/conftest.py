"""Shared fixtures: put both package dirs on sys.path (squeeze tests reuse the
kvpress_ascend.simulate fakes, which mirror the vllm-ascend attribute surface)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KVPRESS_ROOT = ROOT.parent / "kvpress-ascend"

for p in (ROOT, KVPRESS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
