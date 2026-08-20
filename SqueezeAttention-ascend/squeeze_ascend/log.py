"""Logging for SqueezeAttention-ascend. Grep with '[squeeze-ascend]'."""

from __future__ import annotations

import logging
import sys

from squeeze_ascend import envs

logger = logging.getLogger("squeeze-ascend")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[squeeze-ascend] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, envs.log_level().upper(), logging.INFO))
    logger.propagate = False


def set_level(level: str) -> None:
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
