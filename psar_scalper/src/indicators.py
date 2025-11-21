from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average with sane defaults. Returns an empty series if the
    input is empty or the period is invalid.
    """
    if series is None or len(series) == 0 or period <= 0:
        return pd.Series(dtype=float)
    return series.astype(float).ewm(span=period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder smoothing.
    """
    if df is None or df.empty or period <= 0:
        return pd.Series(dtype=float)
    tr = _true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_ema(df: pd.DataFrame, period: int, ema_period: int) -> pd.Series:
    base = atr(df, period)
    if base.empty:
        return base
    return base.ewm(span=max(1, ema_period), adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or df.empty or period <= 0:
        return pd.Series(dtype=float)
    return rsi(df["close"], period)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if series is None or len(series) == 0 or period <= 0:
        return pd.Series(dtype=float)
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(0.0)


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    if series is None or len(series) == 0 or period <= 1:
        return pd.DataFrame({"bb_mid": [], "bb_upper": [], "bb_lower": []})
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or df.empty or period <= 0:
        return pd.Series(dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = _true_range(df)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def compute_psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Simplified PSAR implementation that follows the classic Wilder definition.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    step = max(0.001, float(step))
    max_step = max(step, float(max_step))
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    close = df["close"].astype(float).reset_index(drop=True)

    length = len(df)
    psar = close.copy()
    bull = True
    af = step
    ep = high.iloc[0]
    psar.iloc[0] = low.iloc[0]

    for i in range(1, length):
        prev_psar = psar.iloc[i - 1]
        if bull:
            psar_i = prev_psar + af * (ep - prev_psar)
            psar_i = min(psar_i, low.iloc[i - 1], low.iloc[i - 2] if i >= 2 else low.iloc[i - 1])
        else:
            psar_i = prev_psar + af * (ep - prev_psar)
            psar_i = max(psar_i, high.iloc[i - 1], high.iloc[i - 2] if i >= 2 else high.iloc[i - 1])

        reverse = False
        if bull:
            if low.iloc[i] < psar_i:
                bull = False
                psar_i = ep
                ep = low.iloc[i]
                af = step
                reverse = True
        else:
            if high.iloc[i] > psar_i:
                bull = True
                psar_i = ep
                ep = high.iloc[i]
                af = step
                reverse = True

        if not reverse:
            if bull and high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(max_step, af + step)
            elif not bull and low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(max_step, af + step)

        psar.iloc[i] = psar_i

    psar.index = df.index
    return psar
