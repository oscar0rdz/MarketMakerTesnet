from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import pandas as pd


@dataclass
class MarketBiasSnapshot:
    bias_dir: int
    bias_strength: float
    osc_value: float


@dataclass
class TwinRangeFilterSnapshot:
    dir: int
    color: str
    slope: float
    band_width_pct: float


def compute_market_bias(
    *,
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 60,
    smooth: int = 10,
    osc_period: int = 7,
) -> MarketBiasSnapshot:
    df = pd.DataFrame(
        {
            "open": pd.Series(opens, dtype=float),
            "high": pd.Series(highs, dtype=float),
            "low": pd.Series(lows, dtype=float),
            "close": pd.Series(closes, dtype=float),
        }
    ).dropna()
    if df.empty:
        return MarketBiasSnapshot(0, 0.0, 0.0)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    ha_open.iloc[0] = df["open"].iloc[0]
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_mid = (ha_open + ha_close) / 2.0
    smooth_series = ha_mid.ewm(span=max(2, smooth), adjust=False).mean()
    if len(smooth_series) < max(period, 2):
        slope = smooth_series.iloc[-1] - smooth_series.iloc[0]
    else:
        slope = smooth_series.iloc[-1] - smooth_series.iloc[-period]
    bias_dir = 1 if slope > 0 else -1 if slope < 0 else 0
    strength = min(1.0, abs(slope) / max(smooth_series.abs().mean(), 1e-9))
    osc = smooth_series.diff().ewm(span=max(2, osc_period), adjust=False).mean()
    osc_value = float(osc.iloc[-1]) if len(osc) else 0.0
    return MarketBiasSnapshot(bias_dir=bias_dir, bias_strength=strength, osc_value=osc_value)


def compute_twin_range_filter(
    *,
    closes: Sequence[float],
    fast_period: int = 27,
    fast_range: float = 1.6,
    slow_period: int = 55,
    slow_range: float = 2.0,
) -> TwinRangeFilterSnapshot:
    series = pd.Series(closes, dtype=float).dropna()
    if series.empty:
        return TwinRangeFilterSnapshot(0, "neutral", 0.0, 0.0)
    fast = series.ewm(span=max(2, fast_period), adjust=False).mean()
    slow = series.ewm(span=max(2, slow_period), adjust=False).mean()
    last_fast = float(fast.iloc[-1])
    last_slow = float(slow.iloc[-1])
    dir_flag = 1 if last_fast > last_slow else -1 if last_fast < last_slow else 0
    color = "bull" if dir_flag > 0 else "bear" if dir_flag < 0 else "neutral"
    slope = float(slow.diff().iloc[-1]) if len(slow) > 1 else 0.0
    atr = series.diff().abs().ewm(span=max(2, int(slow_range * 2)), adjust=False).mean()
    atr_val = float(atr.iloc[-1]) if len(atr) else 0.0
    band_width_pct = (atr_val * fast_range) / max(abs(last_slow), 1e-9)
    return TwinRangeFilterSnapshot(
        dir=dir_flag,
        color=color,
        slope=slope,
        band_width_pct=band_width_pct,
    )


__all__ = [
    "MarketBiasSnapshot",
    "TwinRangeFilterSnapshot",
    "compute_market_bias",
    "compute_twin_range_filter",
]
