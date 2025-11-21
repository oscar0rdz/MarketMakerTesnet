from __future__ import annotations

from typing import Optional
import os

from .trend import build_trend_signal, TrendSignal

# Rango de volatilidad considerado sano
ATR_HARD_MIN = float(os.getenv("SCORING_ATR_HARD_MIN", "0.4"))
ATR_SWEET_MIN = float(os.getenv("SCORING_ATR_SWEET_MIN", "0.7"))
ATR_SWEET_MAX = float(os.getenv("SCORING_ATR_SWEET_MAX", "1.8"))
ATR_HARD_MAX = float(os.getenv("SCORING_ATR_HARD_MAX", "2.5"))

# Pesos por componente
MACRO_WEIGHT = float(os.getenv("SCORING_MACRO_WEIGHT", "0.3"))
MICRO_WEIGHT = float(os.getenv("SCORING_MICRO_WEIGHT", "0.7"))

# Magnitud final permitida
SIGNAL_BASE_MAG = float(os.getenv("SIGNAL_BASE_MAG", "0.32"))
SIGNAL_MAX_MAG = float(os.getenv("SIGNAL_MAX_MAG", "0.45"))

# Penalizaciones por posición reciente
SHORT_PRIORITY_DISCOUNT = float(os.getenv("SHORT_PRIORITY_DISCOUNT", "0.92"))
MACRO_CONTRA_FACTOR = float(os.getenv("MACRO_CONTRA_FACTOR", "0.35"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_atr_health(
    atr_pct_30m: float,
    *,
    hard_min: float = ATR_HARD_MIN,
    sweet_min: float = ATR_SWEET_MIN,
    sweet_max: float = ATR_SWEET_MAX,
    hard_max: float = ATR_HARD_MAX,
) -> float:
    """
    Devuelve un factor [0, 1] que representa qué tan saludable es la volatilidad.
    Valores extremadamente bajos o altos regresan 0, mientras que el "sweet spot"
    devuelve 1.0. Entre los rangos se hace una transición lineal.
    """
    atr = max(0.0, float(atr_pct_30m))
    if atr <= hard_min or atr >= hard_max:
        return 0.0
    if sweet_min <= atr <= sweet_max:
        return 1.0
    if atr < sweet_min:
        return (atr - hard_min) / max(sweet_min - hard_min, 1e-9)
    return (hard_max - atr) / max(hard_max - sweet_max, 1e-9)


def compute_macro_alignment(side: int, regime: Optional[str], contra_penalty: float = MACRO_CONTRA_FACTOR) -> float:
    """
    Evalúa si la operación propuesta (side) está alineada con el régimen de 30m.
    side: +1 para LONG, -1 para SHORT
    regime: "up" | "down" | "flat" | None
    """
    dir_sign = 1 if side >= 0 else -1
    regime = (regime or "").lower()
    if regime == "up":
        return 1.0 if dir_sign > 0 else -abs(contra_penalty)
    if regime == "down":
        return 1.0 if dir_sign < 0 else -abs(contra_penalty)
    if regime == "flat":
        return 0.0
    return 0.0


def _sign(value: Optional[float]) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compute_micro_alignment(
    side: int,
    mb_dir: Optional[int],
    trf_dir: Optional[int],
    trf_state: Optional[str],
    slope: Optional[float],
    bandwidth: Optional[float],
) -> float:
    """
    Componente micro del score combinando los indicadores de corto plazo.
    Retorna un valor en [-1, 1].
    """
    side_sign = 1 if side >= 0 else -1
    align = 0.0

    # --- Market Bias ---
    mb_weight = 0.2
    mb_sign = _sign(mb_dir)
    if mb_sign:
        align += mb_weight * (1.0 if mb_sign == side_sign else -1.0)

    # --- Twin Range Filter ---
    trf_weight = 0.8
    trf_sign = _sign(trf_dir)
    if trf_sign:
        base = 1.0 if trf_sign == side_sign else -1.0
        state = (trf_state or "").lower()
        if state in {"bull", "bear"}:
            # Estado fuerte incrementa ligeramente el peso si coincide
            if (state == "bull" and side_sign > 0) or (state == "bear" and side_sign < 0):
                base *= 1.05
            else:
                base *= 0.85
        align += trf_weight * base

    # --- Slope de EMA 1m ---
    slope_weight = 0.2
    if slope is not None:
        slope_norm = _clamp(slope / 0.02, -1.0, 1.0)
        align += slope_weight * slope_norm

    # --- Bandwidth (Bollinger) ---
    if bandwidth is not None:
        if bandwidth < 0.8:
            align -= 0.3
        elif bandwidth > 1.2:
            align += 0.1

    return _clamp(align, -1.0, 1.0)


def _apply_trend_signal_mod(score: float, raw_ts_score: float) -> float:
    if raw_ts_score < 0.10:
        return score * 0.60
    if raw_ts_score > 0.45:
        boost = min(0.20, (raw_ts_score - 0.45))
        return score * (1.0 + boost)
    return score


def _apply_atr5_adjust(score: float, atr_pct_5m: Optional[float]) -> float:
    if atr_pct_5m is None:
        return score
    atr5 = max(0.0, float(atr_pct_5m))
    if atr5 < 0.35:
        return score * 0.85
    if atr5 > 3.0:
        return score * 0.90
    return score


def compute_signal_score_new(
    *,
    side: int,
    raw_ts_score: float,
    regime_30m: Optional[str],
    atr_pct_30m: float,
    atr_pct_5m: Optional[float],
    mb_dir: Optional[int],
    trf_dir: Optional[int],
    trf_state: Optional[str],
    slope: Optional[float],
    bandwidth: Optional[float],
) -> tuple[float, float, float]:
    """
    Calcula la puntuación final del candidato de entrada.
    Regresa (score_final, score_macro, score_micro) donde el primero ya está
    escalado al rango [-SIGNAL_MAX_MAG, SIGNAL_MAX_MAG].
    """
    atr_health = compute_atr_health(atr_pct_30m)
    macro_align = compute_macro_alignment(side, regime_30m, MACRO_CONTRA_FACTOR)
    micro_align = compute_micro_alignment(side, mb_dir, trf_dir, trf_state, slope, bandwidth)

    score_macro = MACRO_WEIGHT * macro_align * atr_health
    score_micro = MICRO_WEIGHT * micro_align

    base_score = score_macro + score_micro
    base_score = _apply_trend_signal_mod(base_score, raw_ts_score)
    base_score = _apply_atr5_adjust(base_score, atr_pct_5m)

    # Clamp a [-1, 1] antes de aplicar magnitud final
    base_score = _clamp(base_score, -1.0, 1.0)

    # Escala a magnitud final
    mag = abs(base_score)
    # Empieza desde SIGNAL_BASE_MAG para señales débiles
    scaled = SIGNAL_BASE_MAG + (SIGNAL_MAX_MAG - SIGNAL_BASE_MAG) * mag

    score_final = scaled if side >= 0 else -scaled
    return score_final, score_macro, score_micro


def compute_signal_score(
    *,
    direction: str,
    regime: Optional[str],
    atr_pct: float,
    mb_dir: Optional[int] = None,
    trf_dir: Optional[int] = None,
    bandwidth: Optional[float] = None,
    slope: Optional[float] = None,
    raw_ts_score: float = 0.0,
    atr_pct_5m: Optional[float] = None,
) -> float:
    side_sign = 1 if direction.lower() == "long" else -1
    score, _, _ = compute_signal_score_new(
        side=side_sign,
        raw_ts_score=abs(raw_ts_score),
        regime_30m=regime,
        atr_pct_30m=float(atr_pct or 0.0) * 100.0,
        atr_pct_5m=atr_pct_5m,
        mb_dir=mb_dir,
        trf_dir=trf_dir,
        trf_state=None,
        slope=slope,
        bandwidth=bandwidth,
    )
    return score


def compute_symbol_trend_signal(symbol: str, indicator_state: dict) -> TrendSignal:
    """
    Wrapper used by the engine to build a TrendSignal out of the indicator snapshot.
    """
    return build_trend_signal(symbol, indicator_state)
