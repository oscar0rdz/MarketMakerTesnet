from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RegimeConfig:
    name: str
    atr_ref: float
    max_bars: int
    sl_multiplier: float
    tp_multiplier: float


def classify_regime(atr_pct: float, pair_cfg) -> RegimeConfig:
    """
    Clasifica el régimen de volatilidad usando los umbrales definidos por par.
    `atr_pct` viene en fracción (0.01 = 1%), mientras pair_cfg está en %.
    """
    atr_pct_pct = float(atr_pct or 0.0) * 100.0
    if atr_pct_pct < pair_cfg.atr_micro:
        return RegimeConfig(name="MICRO", atr_ref=atr_pct_pct, max_bars=pair_cfg.max_bars_micro, sl_multiplier=0.8, tp_multiplier=1.2)
    if atr_pct_pct < pair_cfg.atr_normal:
        return RegimeConfig(name="NORMAL", atr_ref=atr_pct_pct, max_bars=pair_cfg.max_bars_normal, sl_multiplier=1.0, tp_multiplier=1.0)
    return RegimeConfig(name="EXPLOSIVE", atr_ref=atr_pct_pct, max_bars=pair_cfg.max_bars_explosive, sl_multiplier=1.3, tp_multiplier=0.9)


def compute_tp_sl(atr_pct: float, regime: RegimeConfig, pair_cfg, spread_bps: float) -> Tuple[float, float]:
    """
    Devuelve (tp_pct, sl_pct) expresados en porcentaje (0.5 = 0.5%).
    """
    atr_pct_pct = float(atr_pct or 0.0) * 100.0
    base_sl = max(pair_cfg.min_bars_after_flip * 0.01, 0.25)
    sl_pct = base_sl + atr_pct_pct * 0.15
    sl_pct *= regime.sl_multiplier
    tp_pct = sl_pct * 1.5 * regime.tp_multiplier
    spread_penalty = max(0.0, spread_bps / 50.0)
    tp_pct = max(tp_pct - spread_penalty, sl_pct * 0.8)
    return tp_pct, sl_pct


def compute_adaptive_tp_sl(
    *,
    entry_price: float,
    direction: str,
    atr_pct_5m: float,
    base_sl_frac: float,
    base_tp_rr: float,
    fees_frac: float,
    risk_mult: float,
) -> Tuple[float, float, float]:
    atr_factor = 1.0
    if atr_pct_5m:
        atr_factor = max(0.6, min(1.8, atr_pct_5m / 1.2))
    sl_pct = base_sl_frac * atr_factor / max(0.5, min(2.0, risk_mult))
    tp_pct = sl_pct * base_tp_rr
    tp_pct = max(tp_pct - fees_frac * 2.0, sl_pct * 0.8)

    if direction.upper() == "LONG":
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)

    rrr = tp_pct / max(sl_pct, 1e-9)
    return sl_price, tp_price, rrr


def compute_reentry_distance(atr_pct: float, regime: RegimeConfig) -> float:
    atr_pct_abs = float(atr_pct or 0.0)
    base = 0.001 + atr_pct_abs * 0.5
    if regime.name == "EXPLOSIVE":
        base *= 1.5
    elif regime.name == "MICRO":
        base *= 0.8
    return base


def should_trade_microstructure_soft(atr_pct: float, regime: RegimeConfig) -> bool:
    return regime.name != "EXPLOSIVE" or atr_pct < 0.05


def compute_trailing_sl_pct(atr_pct: float, regime: RegimeConfig) -> float:
    atr_val = float(atr_pct or 0.0)
    if regime.name == "MICRO":
        return max(0.001, atr_val * 0.4)
    if regime.name == "NORMAL":
        return max(0.0015, atr_val * 0.6)
    return max(0.0025, atr_val * 0.8)


__all__ = [
    "RegimeConfig",
    "classify_regime",
    "compute_tp_sl",
    "compute_adaptive_tp_sl",
    "compute_reentry_distance",
    "should_trade_microstructure_soft",
    "compute_trailing_sl_pct",
]
