from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .score_utils import compute_signal_score_new


@dataclass
class SymbolSignals:
    symbol: str
    direction: str
    raw_macro_score: float
    raw_micro_score: float
    adx_30m: Optional[float] = None
    atr_pct_30m: Optional[float] = None
    price_4h: Optional[float] = None
    ema200_4h: Optional[float] = None
    regime: Optional[str] = None
    cluster: Optional[str] = None


@dataclass
class EntryCandidate:
    symbol: str
    direction: str
    score: float
    priority: float
    risk_mult: float
    atr_pct: Optional[float] = None
    atr_pct_5m: Optional[float] = None
    trf_state: Optional[str] = None
    score_macro: float = 0.0
    score_micro: float = 0.0
    rrr: float = 0.0
    from_cooldown: bool = False
    cluster: Optional[str] = None


def _risk_from_priority(priority: float, score_settings) -> float:
    low = score_settings.MIN_RISK_MULT
    high = score_settings.MAX_RISK_MULT
    return max(low, min(high, low + priority * (high - low)))


def build_entry_candidate(signals: SymbolSignals, score_settings) -> EntryCandidate:
    side_sign = 1 if signals.direction.upper() == "LONG" else -1
    mb_dir = 1 if signals.raw_macro_score > 0 else -1 if signals.raw_macro_score < 0 else 0
    trf_dir = 1 if signals.raw_micro_score > 0 else -1 if signals.raw_micro_score < 0 else 0
    atr_pct_percent = signals.atr_pct_30m if signals.atr_pct_30m is not None else 0.0

    score, score_macro, score_micro = compute_signal_score_new(
        side=side_sign,
        raw_ts_score=abs(signals.raw_macro_score) + abs(signals.raw_micro_score),
        regime_30m=signals.regime,
        atr_pct_30m=atr_pct_percent,
        atr_pct_5m=None,
        mb_dir=mb_dir,
        trf_dir=trf_dir,
        trf_state=None,
        slope=None,
        bandwidth=None,
    )

    priority = abs(score)
    risk_mult = _risk_from_priority(priority, score_settings)

    return EntryCandidate(
        symbol=signals.symbol,
        direction=signals.direction,
        score=score,
        priority=priority,
        risk_mult=risk_mult,
        atr_pct=signals.atr_pct_30m,
        score_macro=score_macro,
        score_micro=score_micro,
        rrr=score_settings.BASE_RRR,
        cluster=signals.cluster,
    )


__all__ = ["SymbolSignals", "EntryCandidate", "build_entry_candidate"]
