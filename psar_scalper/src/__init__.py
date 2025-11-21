"""
Core modules for the PSAR scalper engine.

This subpackage deliberately separates the "runtime" logic (engine, scoring,
data feeds) from user configuration at ``psar_scalper.config`` so tests can
import only the pieces they require.
"""

from .config import (  # noqa: F401
    PairConfig,
    PAIR_CONFIGS,
    ACTIVE_PAIRS,
    TRADE_LOG_PATH,
)

__all__ = [
    "PairConfig",
    "PAIR_CONFIGS",
    "ACTIVE_PAIRS",
    "TRADE_LOG_PATH",
]
