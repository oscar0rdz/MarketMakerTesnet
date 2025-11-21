from __future__ import annotations

from typing import Optional
import pandas as pd

from .config import RegimeSettings
from .indicators import atr
from .state import RegimeState


def compute_regime_state(df_30m: Optional[pd.DataFrame], settings: RegimeSettings) -> RegimeState:
    if df_30m is None or df_30m.empty or len(df_30m) < max(10, settings.tr_slow_len):
        return RegimeState()

    work = df_30m[["open", "high", "low", "close"]].astype(float).reset_index(drop=True)
    close = work["close"]
    fast = close.ewm(span=max(2, settings.ha_bias_len // 2), adjust=False).mean()
    slow = close.ewm(span=max(3, settings.ha_bias_len), adjust=False).mean()

    bias = 0
    if fast.iloc[-1] > slow.iloc[-1]:
        bias = 1
    elif fast.iloc[-1] < slow.iloc[-1]:
        bias = -1

    atr_series = atr(work, settings.tr_atr_len)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    twin_mid = float(slow.iloc[-1])
    twin_upper = twin_mid + atr_val * settings.tr_band_mult
    twin_lower = twin_mid - atr_val * settings.tr_band_mult
    twin_slope = float(slow.diff().iloc[-1]) if len(slow) > 1 else 0.0

    last_close = float(close.iloc[-1])
    if last_close > twin_upper:
        trend = "up"
    elif last_close < twin_lower:
        trend = "down"
    else:
        trend = "range"

    in_band = twin_lower <= last_close <= twin_upper

    return RegimeState(
        trend=trend,
        bias=bias,
        twin_upper=twin_upper,
        twin_lower=twin_lower,
        twin_mid=twin_mid,
        twin_slope=twin_slope,
        in_band=in_band,
    )


__all__ = ["compute_regime_state"]
