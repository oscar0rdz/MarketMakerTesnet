from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrendSignal:
    symbol: str
    direction: int
    score: float
    score_macro: float
    score_micro: float
    regime: str
    macro_dir: int
    micro_dir: int
    atr_pct_30m: float
    trf_state_5m: str | None = None
    psar_30_dir: int = 0


def compute_trend_score(macro_dir: int, ema_dir: int, ema_slope_deg: float) -> float:
    slope_factor = max(-1.0, min(1.0, ema_slope_deg / 15.0))
    return 0.6 * macro_dir + 0.4 * ema_dir + slope_factor


def compute_micro_score(micro_dir: int, micro_slope: float) -> float:
    return max(-1.0, min(1.0, micro_dir * abs(micro_slope) * 5.0))


def build_trend_signal(symbol: str, indicator_state: dict) -> TrendSignal:
    macro_dir = int(indicator_state.get("psar_30_dir", 0))
    ema_30_dir = int(indicator_state.get("ema_30_dir", 0))
    ema_30_slope_deg = float(indicator_state.get("ema_30_slope_deg", 0.0) or 0.0)
    macro_score = compute_trend_score(macro_dir, ema_30_dir, ema_30_slope_deg)

    micro_dir = 0
    ema_fast = indicator_state.get("ema_1_fast")
    ema_slow = indicator_state.get("ema_1_slow")
    if ema_fast is not None and ema_slow is not None:
        if ema_fast > ema_slow:
            micro_dir = 1
        elif ema_fast < ema_slow:
            micro_dir = -1
    micro_slope = float(indicator_state.get("ema_1_fast_slope", 0.0) or 0.0)
    micro_score = compute_micro_score(micro_dir, micro_slope)

    score = macro_score + micro_score
    direction = 1 if score >= 0 else -1

    regime_label = "TREND" if abs(macro_score) > 0.5 else "RANGE"

    return TrendSignal(
        symbol=symbol,
        direction=direction,
        score=round(score, 4),
        score_macro=round(macro_score, 4),
        score_micro=round(micro_score, 4),
        regime=regime_label,
        macro_dir=macro_dir,
        micro_dir=micro_dir,
        atr_pct_30m=float(indicator_state.get("atr_pct_30m", 0.0) or 0.0) * 100.0,
        trf_state_5m=indicator_state.get("trf_state_5m"),
        psar_30_dir=macro_dir,
    )


__all__ = ["TrendSignal", "build_trend_signal"]
