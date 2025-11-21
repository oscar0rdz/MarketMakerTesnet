"""
psar_scalper package
====================

This package contains the PSAR scalper trading engine and utilities.
Modules under ``psar_scalper.src`` host the core logic (data fetching, scoring,
planning, execution) while ``psar_scalper.utils`` contains helper utilities
such as CSV trade logging.

The top-level package re-exports a few commonly used configuration classes to
keep backwards compatibility with earlier imports.
"""

from .config import (  # noqa: F401
    EngineSettings,
    ScoreSettings,
    StrategySettings,
    RegimeSettings,
    RiskSettings,
    ScalpSettings,
    ScannerSettings,
    BotConfig,
)

__all__ = [
    "EngineSettings",
    "ScoreSettings",
    "StrategySettings",
    "RegimeSettings",
    "RiskSettings",
    "ScalpSettings",
    "ScannerSettings",
    "BotConfig",
]
