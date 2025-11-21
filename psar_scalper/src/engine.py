# -*- coding: utf-8 -*-
"""
Motor principal de la estrategia de trading.

Este módulo contiene la lógica central para el procesamiento de datos de mercado,
la toma de decisiones de trading (entradas, salidas, gestión de posiciones),
y la ejecución de órdenes para cada símbolo.

Funciones clave:
- run_symbol_step: Orquesta el proceso para un símbolo en cada iteración.
- plan_entries_with_min_open_symbols: Decide qué nuevas posiciones abrir.
- evaluate_entry_candidate: Evalúa el potencial de una señal de entrada.
- open_entry_from_candidate: Ejecuta la apertura de una posición.
- should_exit_position: Lógica para decidir el cierre de una posición.
"""
# src/engine.py
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import os
import time
import pandas as pd
from typing import Tuple, Dict, List
from types import SimpleNamespace

# --- Config de tamaño de posición (scalping) ---

# Notional fijo por operación (USDT). Por defecto 30 USDT de tamaño por trade.
TRADE_NOTIONAL_USDT = float(os.getenv("TRADE_NOTIONAL_USDT", "30.0"))

# Piso y techo de notional para evitar que las entradas queden por debajo del mínimo deseado.
MIN_TRADE_NOTIONAL_USDT = float(os.getenv("MIN_TRADE_NOTIONAL_USDT", "30.0"))
MAX_NOTIONAL_USDT = float(os.getenv("MAX_NOTIONAL_USDT", "30.0"))


def compute_order_notional(symbol: str,
                           entry_price: float,
                           risk_mult: float) -> float:
    """
    Devuelve un notional fijo por operación:
    - Usa TRADE_NOTIONAL_USDT para cada entrada (por defecto 10 USDT).
    - Respeta el piso MIN_TRADE_NOTIONAL_USDT y el límite MAX_NOTIONAL_USDT.
    - El multiplicador de riesgo se conserva sólo para compatibilidad/logs.
    """

    _ = risk_mult  # mantenemos la firma; no afecta el tamaño fijo
    notional = max(TRADE_NOTIONAL_USDT, MIN_TRADE_NOTIONAL_USDT)
    if MAX_NOTIONAL_USDT > 0:
        notional = min(notional, MAX_NOTIONAL_USDT)
    
    _logger.debug(
        "[%s] compute_order_notional: TRADE_NOTIONAL_USDT=%.2f, MIN=%.2f, MAX=%.2f => notional=%.2f",
        symbol, TRADE_NOTIONAL_USDT, MIN_TRADE_NOTIONAL_USDT, MAX_NOTIONAL_USDT, notional
    )

    return notional


from .config import (
    PAIR_CONFIGS,
    PairConfig,
    RSI_PERIOD,
    MAX_OPEN_PER_CLUSTER,
)
from .mb_trf_indicators import compute_market_bias, compute_twin_range_filter
from .engine_helpers import (
    SymbolState,
    PositionState,
    build_position_state_from_fill,
    open_core_position,
    maybe_scale_in,
    TradeLogEntry,
    append_trade_log,
    update_loss_streak_and_daily_pnl,
    ExitReason,
    close_position_safely,
    get_cluster_for_symbol,
    update_position_telemetry_from_market,
    should_trend_exit_slowing,
    ensure_stop_loss_order,
)
from .risk_utils import (
    classify_regime,
    compute_tp_sl,
    compute_adaptive_tp_sl,
    compute_reentry_distance,
    should_trade_microstructure_soft,
    compute_trailing_sl_pct,
)
from .regime import compute_regime_state
from .indicators import adx, compute_psar, ema, atr, calculate_rsi
from .score_utils import compute_symbol_trend_signal
from .trend import TrendSignal
from .scoring import EntryCandidate, SymbolSignals, build_entry_candidate

_logger = logging.getLogger(__name__)

# =========================
# CONFIG ENTRADAS / R:R
# =========================
# SL base muy apretado para scalping (0.6% aprox antes de ATR scaling)
BASE_SL_FRAC = float(os.getenv("BASE_SL_FRAC", "0.006"))
# R:R por defecto (se ajusta según ATR internamente)
BASE_TP_RR = float(os.getenv("BASE_TP_RR", "1.4"))
# Fees aproximados en Binance futures (0.08% notional round trip)
FEE_FRAC = float(os.getenv("FEE_FRAC", "0.0008"))

MIN_SCORE_FOR_ENTRY = float(os.getenv("MIN_SCORE_FOR_ENTRY", "0.5"))
COOLDOWN_AFTER_CLOSE_SEC = float(os.getenv("COOLDOWN_AFTER_CLOSE_SEC", "60"))
MIN_SCORE_FOR_FLIP = int(os.getenv("MIN_SCORE_FOR_FLIP", "4"))

# --- Knobs para la lógica de salida trend_exit_slowing ---
MIN_TREND_SLOWING_BARS = int(os.getenv("MIN_TREND_SLOWING_BARS", "4"))
# Mínimo movimiento en contra (en %) para permitir trend_exit_slowing
TREND_SLOWING_MIN_ADVERSE_PCT = float(
    os.getenv("TREND_SLOWING_MIN_ADVERSE_PCT", "0.0015")
)  # 0.15 %

MAX_BARS_IN_TRADE_BY_VOL = {
    "MICRO": 12,
    "NORMAL": 28,
    "EXPLOSIVE": 8,
}
SCORE_FLIP_STRONG = float(os.getenv("SCORE_FLIP_STRONG", "0.35"))   # flip fuerte contra
SCORE_EXIT_WEAK = float(os.getenv("SCORE_EXIT_WEAK", "0.08"))       # tendencia muy débil

# Umbrales para el score macro basado en Market Bias + Twin Range Filter
MB_TREND_WEIGHT = float(os.getenv("MB_TREND_WEIGHT", "0.5"))     # peso del Market Bias
TRF_TREND_WEIGHT = float(os.getenv("TRF_TREND_WEIGHT", "0.3"))   # peso del Twin Range Filter
ATR_TREND_WEIGHT = float(os.getenv("ATR_TREND_WEIGHT", "0.2"))   # peso del ATR%

ATR_REF_TREND = float(os.getenv("ATR_REF_TREND", "2.5"))         # ATR% ~2.5 en alts suele ser tendencia sana
ATR_MIN_CHOP = float(os.getenv("ATR_MIN_CHOP", "0.4"))           # por debajo de esto es calma extrema

MAX_POS_PER_SYMBOL = int(os.getenv("MAX_POS_PER_SYMBOL", "2"))
HIGH_SCORE_THRESHOLD = float(os.getenv("HIGH_SCORE_THRESHOLD", "0.4"))
SHORT_PRIORITY_DISCOUNT = float(os.getenv("SHORT_PRIORITY_DISCOUNT", "0.92"))
MACRO_CONTRA_FACTOR = float(os.getenv("MACRO_CONTRA_FACTOR", "0.35"))
SIGNAL_BASE_MAG = float(os.getenv("SIGNAL_BASE_MAG", "0.32"))
SIGNAL_MAX_MAG = float(os.getenv("SIGNAL_MAX_MAG", "0.45"))
ATR_IDEAL_MIN_PCT = float(os.getenv("ATR_IDEAL_MIN_PCT", "0.8"))
ATR_IDEAL_MAX_PCT = float(os.getenv("ATR_IDEAL_MAX_PCT", "2.5"))
ATR_TOO_LOW_PCT = float(os.getenv("ATR_TOO_LOW_PCT", "0.6"))
ATR_TOO_HIGH_PCT = float(os.getenv("ATR_TOO_HIGH_PCT", "3.8"))
MICRO_FULL_ALIGN_BONUS = float(os.getenv("MICRO_FULL_ALIGN_BONUS", "1.2"))
MICRO_PARTIAL_ALIGN = float(os.getenv("MICRO_PARTIAL_ALIGN", "1.0"))
MICRO_CONTRA_PENALTY = float(os.getenv("MICRO_CONTRA_PENALTY", "0.6"))
CHOP_BW_HIGH = float(os.getenv("CHOP_BW_HIGH", "1.3"))
CHOP_BW_LOW = float(os.getenv("CHOP_BW_LOW", "0.8"))

# Guarda el timestamp (en segundos) del último SL por símbolo
last_sl_ts = {}
LOSS_COOLDOWN_SEC = int(os.getenv("LOSS_COOLDOWN_SEC", "60"))  # 1 minuto por defecto

def can_open_new_trade(state: SymbolState, current_bar_index: int) -> bool:
    return current_bar_index >= state.pause_until_bar


def _compute_dynamic_threshold(
    max_abs_score: float, open_count: int, score_threshold: float
) -> float:
    """
    Calcula un threshold dinámico que se ajusta al número de posiciones abiertas.
    Con pocas posiciones, es más exigente para asegurar que solo entren las mejores señales.
    A medida que se llenan los slots, se relaja para no quedarse fuera del mercado.
    """
    if max_abs_score <= 0:
        return 0.0

    if open_count <= 1:
        fill_factor = 0.9
    elif open_count == 2:
        fill_factor = 0.75
    elif open_count == 3:
        fill_factor = 0.65
    else:
        fill_factor = 0.55  # Con 4+ posiciones, somos más selectivos

    dynamic_threshold = max_abs_score * fill_factor
    return min(score_threshold, dynamic_threshold)


def _count_open_by_symbol(open_positions: list) -> Dict[str, int]:
    """Cuenta las posiciones abiertas por cada símbolo."""
    counts = {}
    for pos in open_positions:
        counts[pos.symbol] = counts.get(pos.symbol, 0) + 1
    return counts


def maybe_log_entry_veto(logger, symbol: str, reason: str, **kwargs) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    if extra:
        logger.info("[%s] ENTRY vetoed: %s (%s)", symbol, reason, extra)
    else:
        logger.info("[%s] ENTRY vetoed: %s", symbol, reason)


@dataclass
class PlannerState:
    empty_iters_below_target: int = 0
    last_exit_bar: Dict[str, int] = field(default_factory=dict)


PLANNER_STATE = PlannerState()


def always_in_market_select(
    candidates: List[EntryCandidate],
    open_positions: Dict[str, str],
    engine_settings,
    planner_state: PlannerState,
    symbol_bar_indices: Dict[str, int],
    score_settings,
    logger,
) -> List[EntryCandidate]:
    """
    Selecciona los mejores candidatos para entrar al mercado bajo la estrategia "Always In Market".
    
    Esta estrategia intenta mantener un número mínimo de posiciones abiertas.
    1. Filtra candidatos con score alto (modo estricto).
    2. Si no hay candidatos fuertes y ha pasado un tiempo, relaja los criterios (modo soft-fill).
    3. Respeta límites de posiciones por símbolo y por clúster de correlación.
    """

    open_count = len(open_positions)
    max_open = engine_settings.always_in_market_max_open
    target_min = min(engine_settings.always_in_market_target_min, max_open)

    slots_to_fill = max(0, min(max_open - open_count, max(0, target_min - open_count)))

    logger.info(
        "ALWAYS_IN_MARKET planner: open_count=%d, target_min=%d, effective_max=%d",
        open_count,
        target_min,
        max_open,
    )

    if slots_to_fill <= 0:
        planner_state.empty_iters_below_target = 0
        return []

    filtered = [c for c in candidates if c.symbol not in open_positions]
    filtered = [c for c in filtered if c.rrr >= engine_settings.min_rrr]

    if not filtered:
        planner_state.empty_iters_below_target += 1
        logger.info(
            "ALWAYS_IN_MARKET: sin candidatos tras filtro RRR. empty_iters_below_target=%d",
            planner_state.empty_iters_below_target,
        )
        return []

    max_abs_score = max(abs(c.score) for c in filtered)
    dyn_th = max(
        engine_settings.score_threshold_soft,
        min(engine_settings.score_threshold, max_abs_score * 0.7),
    )

    logger.info(
        "ALWAYS_IN_MARKET: max_abs_score=%.4f → dynamic_threshold=%.4f (SCORE_THRESHOLD=%.4f, SOFT=%.4f)",
        max_abs_score,
        dyn_th,
        engine_settings.score_threshold,
        engine_settings.score_threshold_soft,
    )

    strong = [c for c in filtered if abs(c.score) >= dyn_th]
    strong.sort(key=lambda c: c.priority, reverse=True)

    cluster_counts: Dict[str, int] = {}
    symbol_counts: Dict[str, int] = {}
    for sym in open_positions:
        cluster = get_cluster_for_symbol(sym)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

    def _select(pool: List[EntryCandidate]) -> List[EntryCandidate]:
        picks: List[EntryCandidate] = []
        for cand in pool:
            if len(picks) >= slots_to_fill:
                break
            if symbol_counts.get(cand.symbol, 0) >= MAX_POS_PER_SYMBOL:
                continue
            cluster = cand.cluster or get_cluster_for_symbol(cand.symbol)
            if cluster_counts.get(cluster, 0) >= MAX_OPEN_PER_CLUSTER:
                continue
            picks.append(cand)
            symbol_counts[cand.symbol] = symbol_counts.get(cand.symbol, 0) + 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        return picks

    selected = _select(strong)

    if selected:
        planner_state.empty_iters_below_target = 0
        for c in selected:
            logger.info(
                "ALWAYS_IN_MARKET SELECTED [%s] dir=%s score=%.4f notional_base=%.2f risk_mult=%.2f RRR=%.2f cluster=%s (modo estricto)",
                c.symbol,
                c.direction,
                c.score,
                score_settings.BASE_NOTIONAL,
                c.risk_mult,
                c.rrr,
                c.cluster,
            )
        return selected

    planner_state.empty_iters_below_target += 1
    logger.info(
        "ALWAYS_IN_MARKET: no se seleccionó ningún candidato fuerte. empty_iters_below_target=%d",
        planner_state.empty_iters_below_target,
    )

    if planner_state.empty_iters_below_target < engine_settings.soft_fill_max_empty_iters:
        logger.info(
            "ALWAYS_IN_MARKET: todavía no activamos soft-fill (SOFT_FILL_MAX_EMPTY_ITERS=%d).",
            engine_settings.soft_fill_max_empty_iters,
        )
        return []

    if max_abs_score < engine_settings.score_threshold_soft:
        logger.info(
            "ALWAYS_IN_MARKET: max_abs_score %.4f < SCORE_THRESHOLD_SOFT=%.4f → ni siquiera para soft-fill.",
            max_abs_score,
            engine_settings.score_threshold_soft,
        )
        return []

    soft: List[EntryCandidate] = []
    for c in filtered:
        last_exit = planner_state.last_exit_bar.get(c.symbol, -10**9)
        current_bar = symbol_bar_indices.get(c.symbol, 0)
        bars_since_exit = current_bar - last_exit
        if (
            bars_since_exit < engine_settings.symbol_cooldown_bars
            and abs(c.score) < engine_settings.cooldown_force_score
        ):
            logger.info(
                "ALWAYS_IN_MARKET: %s en cooldown (%d barras desde cierre) con score=%.4f < force_score=%.4f → skip.",
                c.symbol,
                bars_since_exit,
                c.score,
                engine_settings.cooldown_force_score,
            )
            continue
        soft.append(c)

    soft.sort(key=lambda c: c.priority, reverse=True)
    selected = _select(soft)

    for c in selected:
        logger.info(
            "ALWAYS_IN_MARKET SELECTED [%s] dir=%s score=%.4f notional_base=%.2f risk_mult=%.2f RRR=%.2f cluster=%s (modo soft-fill)",
            c.symbol,
            c.direction,
            c.score,
            score_settings.BASE_NOTIONAL,
            c.risk_mult,
            c.rrr,
            c.cluster,
        )

    return selected


def on_position_closed(symbol: str, current_bar_index: int, planner_state: PlannerState, logger) -> None:
    planner_state.last_exit_bar[symbol] = current_bar_index
    logger.info(
        "ALWAYS_IN_MARKET: posición cerrada en %s → cooldown activado (last_exit_bar=%d).",
        symbol,
        current_bar_index,
    )


def _psar_direction(trend: str) -> int:
    normalized = (trend or "").lower()
    if normalized == "up":
        return 1
    if normalized == "down":
        return -1
    return 0


def _compute_short_term_trend(
    df_1m: pd.DataFrame, fast: int, slow: int, confirm_bars: int
) -> Tuple[int, float, float]:
    """
    Director de tendencia de corto plazo (1m) basado en cruce EMA y pendiente.
    Devuelve (dirección, pendiente media, fuerza normalizada 0–2.5).
    """
    if df_1m.empty or len(df_1m) < max(slow, fast) + confirm_bars:
        return 0, 0.0, 0.0

    close = df_1m["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    spread = ema_fast - ema_slow

    recent = spread.iloc[-confirm_bars:]
    avg_spread = recent.mean()
    slope = recent.diff().mean(skipna=True) or 0.0
    price_ref = float(close.iloc[-1])

    if abs(avg_spread) < 1e-8:
        return 0, slope, 0.0

    direction = 1 if avg_spread > 0 else -1
    # Si la pendiente contradice el spread, considerar señal débil
    if slope * direction < 0:
        direction = direction if abs(avg_spread) > abs(slope) else 0

    # Convertir pendiente y spread a bps para cuantificar fuerza
    momentum_bps = abs(avg_spread) / max(price_ref, 1e-9) * 10000.0
    slope_bps = abs(slope) / max(price_ref, 1e-9) * 10000.0

    # Fuerza pondera momentum (spread) y pendiente reciente; topada en 2.5
    strength = 0.25 + min(1.6, momentum_bps / 4.0) + min(0.9, slope_bps / 3.0)
    strength = min(2.5, strength)
    if direction == 0:
        strength = 0.0
    return direction, slope, strength


def _ema_slope(series: pd.Series, lookback: int) -> float:
    """Slope normalizada por lookback; tolerante a histórico corto."""
    if series is None or series.empty or lookback <= 0:
        return 0.0
    if len(series) <= lookback:
        return 0.0
    recent = float(series.iloc[-1])
    prev = float(series.iloc[-(lookback + 1)])
    if prev == 0:
        return 0.0
    return (recent - prev) / (abs(prev) * lookback)


def _ema_slope_degrees(series: pd.Series, lookback: int) -> float:
    if series is None or series.empty or lookback <= 0 or len(series) <= lookback:
        return 0.0
    recent = float(series.iloc[-1])
    prev = float(series.iloc[-(lookback + 1)])
    if prev == 0:
        return 0.0
    normalized = (recent - prev) / abs(prev)
    return math.degrees(math.atan(normalized))


def _build_indicator_state(symbol: str, bar: pd.Series, df_1m: pd.DataFrame, df_30m: pd.DataFrame, strategy_settings, trf_state_5m: str | None = None) -> Dict[str, float] | None:
    """
    Extrae los indicadores necesarios para construir TrendSignal.
    Devuelve None si faltan datos esenciales.
    """
    if df_1m is None or df_30m is None or df_1m.empty or df_30m.empty:
        return None

    try:
        df_1m_local = df_1m.copy()
        df_30m_local = df_30m.copy()

        df_1m_local["rsi"] = calculate_rsi(df_1m_local, period=RSI_PERIOD)
        df_30m_local["rsi"] = calculate_rsi(df_30m_local, period=RSI_PERIOD)

        close_1m = float(bar.close)

        psar_1_series = compute_psar(df_1m_local, strategy_settings.micro_psar_step, strategy_settings.micro_psar_max_step)
        psar_1 = float(psar_1_series.iloc[-1]) if not psar_1_series.empty else close_1m

        psar_30_series = compute_psar(df_30m_local, strategy_settings.sar_step, strategy_settings.sar_max_step)
        psar_30 = float(psar_30_series.iloc[-1]) if not psar_30_series.empty else close_1m

        ema_1_fast_series = ema(df_1m_local["close"], strategy_settings.micro_monitor_ema_fast)
        ema_1_slow_series = ema(df_1m_local["close"], strategy_settings.micro_monitor_ema_slow)
        ema_1_fast = float(ema_1_fast_series.iloc[-1])
        ema_1_slow = float(ema_1_slow_series.iloc[-1])

        ema_30_fast_series = ema(df_30m_local["close"], strategy_settings.ema_fast_period)
        ema_30_slow_series = ema(df_30m_local["close"], strategy_settings.ema_period)
        ema_30_fast = float(ema_30_fast_series.iloc[-1])
        ema_30_slow = float(ema_30_slow_series.iloc[-1])

        atr_30_series = atr(df_30m_local, period=strategy_settings.atr_period)
        atr_30 = float(atr_30_series.iloc[-1]) if not atr_30_series.empty else 0.0

        ema_1_fast_slope = _ema_slope(ema_1_fast_series, strategy_settings.ema_slope_lookback)
        ema_1_slow_slope = _ema_slope(ema_1_slow_series, strategy_settings.ema_slope_lookback)
        ema_30_slope_deg = _ema_slope_degrees(ema_30_fast_series, strategy_settings.ema_slope_lookback)

        rsi_1m = float(df_1m_local["rsi"].iloc[-1])
        rsi_30m = float(df_30m_local["rsi"].iloc[-1])

        psar_30_dir = 1 if close_1m > psar_30 else -1
        if len(psar_30_series) > 1:
            prev_price = float(df_30m_local["close"].iloc[-2])
            prev_psar = float(psar_30_series.iloc[-2])
            prev_dir = 1 if prev_price > prev_psar else -1
        else:
            prev_dir = psar_30_dir
        psar_flipped_recent = psar_30_dir != prev_dir

        ema_30_dir = 0
        if ema_30_fast > ema_30_slow:
            ema_30_dir = 1
        elif ema_30_fast < ema_30_slow:
            ema_30_dir = -1

        atr_pct_30m = (atr_30 / close_1m) if close_1m > 0 else 0.0
    except Exception as exc:
        _logger.debug("[%s] indicator_state build failed: %s", symbol, exc)
        return None

    return {
        "close_1m": close_1m,
        "psar_1": psar_1,
        "psar_30": psar_30,
        "ema_1_fast": ema_1_fast,
        "ema_1_slow": ema_1_slow,
        "ema_30_fast": ema_30_fast,
        "ema_30_slow": ema_30_slow,
        "ema_1_fast_slope": ema_1_fast_slope,
        "ema_1_slow_slope": ema_1_slow_slope,
        "atr_30": atr_30,
        "ema_30_slope_deg": ema_30_slope_deg,
        "psar_30_dir": psar_30_dir,
        "ema_30_dir": ema_30_dir,
        "psar_flipped_recent": psar_flipped_recent,
        "atr_pct_30m": atr_pct_30m,
        "rsi_1m": rsi_1m,
        "rsi_30m": rsi_30m,
        "trf_state_5m": trf_state_5m,
    }


def _determine_entry_direction(
    regime_state,
    psar_dir: int,
    short_dir: int,
    short_strength: float,
    volatility_regime,
    spread_bps: float,
    last_direction: int,
) -> Tuple[int, str, float]:
    """
    Score direccional multidimensional:
      - sesgo y tendencia de 30m (Market Bias + Twin Range)
      - psar 30m
      - momentum 1m (fuerza normalizada)
      - penalización por rango estrecho/spread alto
    """
    entry_mode = "trend" if regime_state.trend in ("up", "down") else "range"
    score = 0.0

    # 1) Sesgo estructural (HA bias) y tendencia Twin Range
    bias_weight = 2.5
    trend_weight = 1.6
    if regime_state.bias != 0:
        score += bias_weight * regime_state.bias
    twin_mid = getattr(regime_state, "twin_mid", None) or 0.0
    twin_slope = getattr(regime_state, "twin_slope", None) or 0.0
    slope_boost = 0.0
    if abs(twin_mid) > 1e-8:
        slope_pct = abs(twin_slope) / abs(twin_mid)
        slope_boost = min(0.8, slope_pct * 900.0)  # refuerza tendencias limpias

    if regime_state.trend == "up":
        score += trend_weight + slope_boost
    elif regime_state.trend == "down":
        score -= trend_weight + slope_boost

    # 2) PSAR 30m (conservador en rango)
    psar_weight = 1.4
    if regime_state.trend == "range":
        psar_weight *= 0.7
    score += psar_weight * psar_dir

    # 3) Tendencia de 1m con fuerza normalizada
    if short_dir != 0 and short_strength > 0:
        score += short_dir * short_strength

    # 4) Ajustes por régimen de volatilidad y microestructura
    if volatility_regime.name == "MICRO":
        score *= 0.85
    elif volatility_regime.name == "EXPLOSIVE":
        score *= 1.05

    if regime_state.trend == "range" and getattr(regime_state, "in_band", False):
        score *= 0.6

    # Spread alto resta convicción
    if spread_bps > 0:
        score -= min(1.2, spread_bps / 8.0)

    # Evitar score plano: empujar al último sentido conocido
    if abs(score) < 1e-6:
        score = float(last_direction or 1) * 0.1

    direction = 1 if score > 0 else -1
    return direction, entry_mode, round(score, 2)


def _risk_mult_from_score(score: float, volatility_regime) -> float:
    """
    Ajusta el tamaño base según la fuerza/confianza del score y el régimen de volatilidad.
    Nunca devuelve menos de 0.35 ni más de 1.5 para evitar tamaños extremos.
    """
    if score >= 4:
        base = 1.25
    elif score >= 2:
        base = 1.0
    elif score >= 0:
        base = 0.75
    else:
        base = 0.50

    if getattr(volatility_regime, "name", "") == "MICRO":
        base *= 0.7  # defensivo
    elif getattr(volatility_regime, "name", "") == "EXPLOSIVE":
        base *= 1.1  # un poco más agresivo para capturar momentum

    return max(0.35, min(base, 1.5))


def _should_exit_position(
    symbol: str,
    position,
    regime_state,
    psar_dir: int,
    short_dir: int,
    current_price: float,
) -> str | None:
    """
    Evalúa desalineación de indicadores para cerrar/flip rápido.
    Requiere al menos 2 señales fuertes en contra.
    """
    misaligned = []
    if psar_dir and psar_dir != position.direction:
        misaligned.append("psar30")
    if regime_state.bias and regime_state.bias != position.direction:
        misaligned.append("market_bias")

    if regime_state.trend == "up" and position.direction == -1:
        misaligned.append("twin_range")
    elif regime_state.trend == "down" and position.direction == 1:
        misaligned.append("twin_range")

    if short_dir and short_dir != position.direction:
        misaligned.append("short_trend")

    # Ruptura de banda invalida inmediatamente la posición
    if regime_state.twin_upper and regime_state.twin_lower:
        if position.direction == 1 and current_price < regime_state.twin_lower:
            _logger.info("[%s] Band break: price %.6f < lower %.6f", symbol, current_price, regime_state.twin_lower)
            return "trend_exit"
        if position.direction == -1 and current_price > regime_state.twin_upper:
            _logger.info("[%s] Band break: price %.6f > upper %.6f", symbol, current_price, regime_state.twin_upper)
            return "trend_exit"

    if len(misaligned) >= 2:
        _logger.info("[%s] Exit due to misalignment: %s", symbol, ",".join(misaligned))
        return "trend_exit"
    return None


def _update_trailing_stop(symbol: str, position, current_price: float, atr_pct: float, volatility_regime) -> bool:
    """
    Ajusta trailing stop dinámico y devuelve True si debe cerrarse por protección.
    """
    trail_pct = compute_trailing_sl_pct(atr_pct, volatility_regime)
    if trail_pct <= 0:
        return False

    triggered = False
    if position.direction == 1:
        candidate = current_price * (1 - trail_pct)
        if current_price > position.entry_price and candidate > position.trailing_sl_price:
            position.trailing_sl_price = candidate
            _logger.info("[%s] Trailing SL raised to %.6f (%.2f%% behind)", symbol, candidate, trail_pct * 100)
        if current_price <= position.trailing_sl_price:
            triggered = True
    else:
        candidate = current_price * (1 + trail_pct)
        if current_price < position.entry_price and candidate < position.trailing_sl_price:
            position.trailing_sl_price = candidate
            _logger.info("[%s] Trailing SL lowered to %.6f (%.2f%% ahead)", symbol, candidate, trail_pct * 100)
        if current_price >= position.trailing_sl_price:
            triggered = True

    return triggered


def compute_signal_score(
    direction: str,
    regime: str | None,
    atr_pct: float | None,
    mb_dir: int | None = None,
    trf_dir: int | None = None,
    bandwidth: float | None = None,
    slope: float | None = None,
    raw_ts_score: float = 0.0,
    atr_pct_5m: float | None = None,
    trf_state: str | None = None,
) -> float:
    """
    Wrapper que usa el nuevo sistema de scoring de score_utils.py.
    Mantiene compatibilidad con llamadas existentes pero usa el nuevo algoritmo.
    Devuelve un valor en [-SIGNAL_MAX_MAG, SIGNAL_MAX_MAG]; el signo define la dirección.
    """
    from .score_utils import compute_signal_score_new
    
    dir_sign = 1 if (direction or "").upper() == "LONG" else -1
    regime_str = (regime or "").upper()
    
    # Normalizar ATR a porcentaje si viene como fracción
    atr_30m = float(atr_pct or 0.0)
    if atr_30m < 10.0:  # Probablemente es fracción, convertir a %
        atr_30m = atr_30m * 100.0
    
    atr_5m = float(atr_pct_5m or atr_30m or 0.0)
    if atr_5m < 10.0:
        atr_5m = atr_5m * 100.0
    
    # Usar el nuevo sistema de scoring
    score, score_macro, score_micro = compute_signal_score_new(
        side=dir_sign,
        raw_ts_score=raw_ts_score,
        regime_30m=regime_str,
        atr_pct_30m=atr_30m,
        atr_pct_5m=atr_5m,
        mb_dir=mb_dir,
        trf_dir=trf_dir,
        trf_state=trf_state,
        slope=slope,
        bandwidth=bandwidth,
        signal_max_mag=SIGNAL_MAX_MAG,
        macro_contra_factor=MACRO_CONTRA_FACTOR,
    )
    
    return score


def compute_mb_trf_score(symbol: str, ohlc_5m, atr_pct_5m: float) -> Tuple[float, dict]:
    """
    Convierte Market Bias + Twin Range Filter + ATR% en un score en [-1, 1].
    >0 → sesgo LONG; <0 → sesgo SHORT.
    Devuelve también un dict con info para logging.
    """

    if not ohlc_5m:
        return 0.0, {
            "mb_dir": 0,
            "mb_strength": 0.0,
            "mb_osc": 0.0,
            "trf_dir": 0,
            "trf_color": "neutral",
            "trf_slope": 0.0,
            "trf_band_width_pct": 0.0,
            "atr_pct": round(atr_pct_5m, 4),
            "atr_factor": 0.0,
            "raw_score": 0.0,
            "score": 0.0,
        }

    opens = [float(c["open"]) for c in ohlc_5m if c.get("open") is not None]
    highs = [float(c["high"]) for c in ohlc_5m if c.get("high") is not None]
    lows = [float(c["low"]) for c in ohlc_5m if c.get("low") is not None]
    closes = [float(c["close"]) for c in ohlc_5m if c.get("close") is not None]

    if not closes or not opens or not highs or not lows:
        return 0.0, {
            "mb_dir": 0,
            "mb_strength": 0.0,
            "mb_osc": 0.0,
            "trf_dir": 0,
            "trf_color": "neutral",
            "trf_slope": 0.0,
            "trf_band_width_pct": 0.0,
            "atr_pct": round(atr_pct_5m, 4),
            "atr_factor": 0.0,
            "raw_score": 0.0,
            "score": 0.0,
        }

    mb = compute_market_bias(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        period=60,
        smooth=10,
        osc_period=7,
    )

    trf = compute_twin_range_filter(
        closes=closes,
        fast_period=27,
        fast_range=1.6,
        slow_period=55,
        slow_range=2.0,
    )

    # Convertimos ATR% en factor 0..1 (atr_pct_5m viene en porcentaje, no en fracción)
    atr_pct_val = float(atr_pct_5m or 0.0)
    if atr_pct_val <= ATR_MIN_CHOP:
        atr_factor = 0.2
    else:
        atr_factor = min(1.0, atr_pct_val / ATR_REF_TREND)

    # Direcciones elementales
    mb_dir = mb.bias_dir  # +1 / -1 / 0
    trf_dir = trf.dir  # +1 / -1 / 0

    # Score base: combinación lineal
    raw_score = (
        MB_TREND_WEIGHT * mb_dir * mb.bias_strength
        + TRF_TREND_WEIGHT * trf_dir
        + ATR_TREND_WEIGHT * (1 if atr_pct_val >= ATR_MIN_CHOP else 0)
    )

    # Aplicamos factor de volatilidad
    score = raw_score * atr_factor

    # Clamp a [-1, 1]
    score = max(-1.0, min(1.0, score))

    info = {
        "mb_dir": mb_dir,
        "mb_strength": round(mb.bias_strength, 4),
        "mb_osc": round(mb.osc_value, 5),
        "trf_dir": trf_dir,
        "trf_color": trf.color,
        "trf_slope": round(trf.slope, 4),
        "trf_band_width_pct": round(trf.band_width_pct, 4),
        "atr_pct": round(atr_pct_val, 4),
        "atr_factor": round(atr_factor, 4),
        "raw_score": round(raw_score, 4),
        "score": round(score, 4),
    }

    return score, info


def evaluate_entry_candidate(
    symbol: str,
    snap,
    state,
    now_ts: float,
    logger,
    filling_deficit: bool,
    score_settings,
) -> EntryCandidate | None:
    """
    Evalúa si un símbolo es un buen candidato para una nueva entrada.

    - Construye una señal de trading (`TrendSignal`) a partir de los indicadores.
    - Calcula un score basado en el régimen de mercado, ATR, y microestructura.
    - La prioridad se basa en el valor absoluto del score.
    - Aplica una penalización si el símbolo está en un período de "cooldown" tras una pérdida.
    - Devuelve un objeto `EntryCandidate` o `None` si no es un buen candidato.
    """

    trend_sig: TrendSignal | None = getattr(snap, "trend_signal", None)

    raw_macro = float(getattr(trend_sig, "score_macro", 0.0) or 0.0) if trend_sig else 0.0
    raw_micro = float(getattr(trend_sig, "score_micro", 0.0) or 0.0) if trend_sig else 0.0
    regime_label = getattr(trend_sig, "regime", getattr(snap, "vol_regime", "N/A"))
    atr_pct_macro_pct = None
    if trend_sig and getattr(trend_sig, "atr_pct_30m", None) is not None:
        atr_pct_macro_pct = float(trend_sig.atr_pct_30m)
    else:
        atr_base = float(getattr(snap, "atr_pct", 0.0) or 0.0)
        if atr_base > 0:
            atr_pct_macro_pct = atr_base * 100.0

    direction_flag = 1
    if trend_sig and trend_sig.direction != 0:
        direction_flag = 1 if trend_sig.direction > 0 else -1
    elif raw_macro or raw_micro:
        direction_flag = 1 if (raw_macro + raw_micro) >= 0 else -1

    direction = "LONG" if direction_flag >= 0 else "SHORT"
    trf_state = getattr(trend_sig, "trf_state_5m", None) or getattr(snap, "trf_state_5m", None)

    if not trend_sig:
        ohlc_5m = getattr(snap, "ohlc_5m", None) or []
        atr_pct_5m = float(getattr(snap, "atr_pct_5m", 0.0) or 0.0)
        raw_score, mb_trf_info = compute_mb_trf_score(symbol, ohlc_5m, atr_pct_5m)
        raw_micro = raw_score
        raw_macro = float(mb_trf_info.get("mb_dir") or 0.0)
        if raw_score == 0:
            direction = "LONG"
        elif raw_score < 0:
            direction = "SHORT"
        if not regime_label or regime_label in ("N/A", "UNKNOWN"):
            regime_label = getattr(getattr(snap, "volatility_regime", None), "name", "MICRO")
        trf_state = mb_trf_info.get("trf_color")

    adx_val = getattr(snap, "adx_30m", None)
    atr_pct_for_candidate = atr_pct_macro_pct if atr_pct_macro_pct is not None else 0.0
    price_4h = getattr(snap, "price_4h", None)
    ema200_4h = getattr(snap, "ema200_4h", None)
    cluster = get_cluster_for_symbol(symbol)

    sig = SymbolSignals(
        symbol=symbol,
        direction=direction,
        raw_macro_score=raw_macro,
        raw_micro_score=raw_micro,
        adx_30m=adx_val,
        atr_pct_30m=atr_pct_macro_pct,
        price_4h=price_4h,
        ema200_4h=ema200_4h,
        regime=regime_label,
        cluster=cluster,
    )

    candidate = build_entry_candidate(sig, score_settings)
    candidate.atr_pct_5m = float(getattr(snap, "atr_pct_5m", 0.0) or 0.0)
    candidate.trf_state = trf_state

    cooldown_applied = False
    if symbol in last_sl_ts and now_ts - last_sl_ts[symbol] < LOSS_COOLDOWN_SEC:
        candidate.priority *= 0.4
        cooldown_applied = True
    candidate.from_cooldown = cooldown_applied

    logger.info(
        "[%s] ENTRY candidate: dir=%s | score=%.4f | macro=%.3f | micro=%.3f | priority=%.4f | regime=%s | atr_pct=%.3f | filling_deficit=%s%s",
        symbol,
        direction,
        candidate.score,
        raw_macro,
        raw_micro,
        candidate.priority,
        regime_label,
        atr_pct_for_candidate,
        filling_deficit,
        " | cooldown" if cooldown_applied else "",
    )

    return candidate


def should_exit_with_trend_signal(symbol: str, position, trend_sig: TrendSignal | None, logger) -> tuple[bool, str]:
    """
    Evalúa salidas usando TrendSignal (bias flip y debilidad de score).
    """
    if trend_sig is None:
        return False, ""

    score = trend_sig.score
    dir_sig = trend_sig.direction

    # Flip fuerte en contra
    if position.direction == 1 and dir_sig < 0 and abs(score) >= SCORE_FLIP_STRONG:
        logger.info(
            "[%s] EXIT decision: bias_flip_strong contra LONG | score=%.4f | macro_dir=%d | micro_dir=%d | regime=%s",
            symbol,
            score,
            trend_sig.macro_dir,
            trend_sig.micro_dir,
            trend_sig.regime,
        )
        return True, "bias_flip_strong"

    if position.direction == -1 and dir_sig > 0 and abs(score) >= SCORE_FLIP_STRONG:
        logger.info(
            "[%s] EXIT decision: bias_flip_strong contra SHORT | score=%.4f | macro_dir=%d | micro_dir=%d | regime=%s",
            symbol,
            score,
            trend_sig.macro_dir,
            trend_sig.micro_dir,
            trend_sig.regime,
        )
        return True, "bias_flip_strong"

    return False, ""


def should_exit_position(
    symbol: str,
    position,
    score: float,
    vol_regime: str,
    bars_in_trade: int,
    logger,
) -> tuple[bool, str]:
    """
    Decide si una posición abierta debe ser cerrada.

    Razones para cerrar:
    - `bias_flip`: El score de la señal cambia fuertemente en contra de la posición.
    - `time_stop`: La posición ha estado abierta por más tiempo del permitido para su régimen de volatilidad.
    
    Devuelve una tupla (bool, str) indicando si se debe cerrar y la razón.
    """
    # bias flip solo con score fuerte contrario
    if position.direction == 1 and score <= -MIN_SCORE_FOR_FLIP:
        logger.info(
            f"[{symbol}] EXIT decision bias_flip | side=LONG -> score={score:.2f} <= -{MIN_SCORE_FOR_FLIP}"
        )
        return True, "bias_flip"
    if position.direction == -1 and score >= MIN_SCORE_FOR_FLIP:
        logger.info(
            f"[{symbol}] EXIT decision bias_flip | side=SHORT -> score={score:.2f} >= +{MIN_SCORE_FOR_FLIP}"
        )
        return True, "bias_flip"

    max_bars = MAX_BARS_IN_TRADE_BY_VOL.get(vol_regime)
    if max_bars is not None and bars_in_trade >= max_bars:
        logger.info(
            f"[{symbol}] EXIT decision time_stop | vol_regime={vol_regime} bars_in_trade={bars_in_trade} >= {max_bars}"
        )
        return True, "time_stop"

    # misalign / trailing se evalúan después
    return False, ""


def _fetch_exchange_position(api, symbol: str):
    """
    Obtiene la posición actual del exchange para un símbolo.
    Devuelve un dict con al menos positionAmt y entryPrice, o None si no hay.
    """
    try:
        if hasattr(api, "exchange") and hasattr(api.exchange, "fetch_positions"):
            raw = api.exchange.fetch_positions([symbol])
            for pos in raw:
                if pos.get("symbol") == symbol:
                    return pos
    except Exception:
        pass
    # Fallback: intentar vía balance.info.positions (binance)
    try:
        if hasattr(api, "exchange"):
            bal = api.exchange.fetch_balance()
            info = bal.get("info") or {}
            for pos in info.get("positions", []):
                if pos.get("symbol") == symbol:
                    return pos
    except Exception:
        pass
    return None


def sync_symbol_state_from_exchange(symbol: str, state: SymbolState, exchange_pos, current_bar_index: int, logger) -> None:
    """
    Sincroniza el estado interno con la posición del exchange.
    Nunca cierra posiciones; sólo adopta o marca plano.
    """
    pos_amt = 0.0
    try:
        pos_amt = float(exchange_pos.get("positionAmt", 0)) if exchange_pos else 0.0
    except Exception:
        pos_amt = 0.0

    if exchange_pos is None or abs(pos_amt) < 1e-8:
        if state.position is not None:
            logger.info(f"[{symbol}] Sync: no exchange position, marking internal flat")
        state.position = None
        state.current_direction = 0
        return

    # Adoptar si no hay posición interna
    if state.position is None:
        side_dir = 1 if pos_amt > 0 else -1
        entry_price = float(exchange_pos.get("entryPrice", 0) or 0.0)
        qty = abs(pos_amt)
        logger.info(
            f"[{symbol}] Sync: adopting existing exchange position | side={'LONG' if side_dir==1 else 'SHORT'} qty={qty} entry={entry_price}"
        )
        mark_price = float(exchange_pos.get("markPrice") or entry_price or 0.0)
        fill_reference = entry_price if entry_price > 0 else mark_price
        state.position = build_position_state_from_fill(
            symbol=symbol,
            direction=side_dir,
            fill_price=fill_reference,
            filled_qty=qty,
            tp_price=None,
            sl_price=entry_price if entry_price > 0 else None,
            now_ts=time.time(),
            open_bar_index=current_bar_index,
            atr_pct_entry=0.0,
            regime_entry="sync",
        )
        state.current_direction = side_dir
        return

    # Si ya hay posición interna, puedes validar consistencia pero no cerrar aquí.
    return


def plan_entries_with_min_open_symbols(
    symbols,
    snapshots: dict,
    states: dict,
    engine_settings,
    score_settings,
    logger,
) -> list[EntryCandidate]:
    """
    Planifica nuevas entradas para mantener un número mínimo de posiciones abiertas.

    Esta es la función principal de planificación para la estrategia "Always In Market".
    1. Recopila el estado actual de todas las posiciones.
    2. Para cada símbolo sin posición, evalúa si es un buen candidato para una entrada.
    3. Pasa la lista de candidatos a `always_in_market_select` para la selección final.
    4. Devuelve la lista de candidatos seleccionados que se deben abrir.
    """
    now_ts = time.time()
    open_positions: Dict[str, str] = {}
    for sym, st in states.items():
        if st.position is not None:
            direction = "LONG" if st.position.direction == 1 else "SHORT"
            open_positions[sym] = direction

    candidates: list[EntryCandidate] = []
    symbol_bar_indices: Dict[str, int] = {}
    for symbol in symbols:
        snap = snapshots[symbol]
        state = states[symbol]
        symbol_bar_indices[symbol] = getattr(snap, "bar_index", 0)
        if getattr(snap, "position", None) is not None:
            continue
        candidate = evaluate_entry_candidate(
            symbol=symbol,
            snap=snap,
            state=state,
            now_ts=now_ts,
            logger=logger,
            filling_deficit=True,
            score_settings=score_settings,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return []

    return always_in_market_select(
        candidates=candidates,
        open_positions=open_positions,
        engine_settings=engine_settings,
        planner_state=PLANNER_STATE,
        symbol_bar_indices=symbol_bar_indices,
        score_settings=score_settings,
        logger=logger,
    )


def build_sl_tp(side: str, price: float, atr_pct_5m: float, atr_pct_30m: float | None, score_settings) -> Tuple[float, float, float, float, float]:
    """Construye SL/TP desacoplados del score usando ATR y RRR base."""

    atr_5m = float(atr_pct_5m or 0.0)
    atr_30m = float(atr_pct_30m or 0.0)
    atr_mix = 0.6 * atr_30m + 0.4 * atr_5m
    base_sl_pct = max(0.35, min(1.20, atr_mix))

    sl_pct = base_sl_pct
    tp_pct = sl_pct * score_settings.BASE_RRR

    side_up = (side or "LONG").upper() == "LONG"
    if side_up:
        sl_price = price * (1.0 - sl_pct / 100.0)
        tp_price = price * (1.0 + tp_pct / 100.0)
    else:
        sl_price = price * (1.0 + sl_pct / 100.0)
        tp_price = price * (1.0 - tp_pct / 100.0)

    rrr = tp_pct / sl_pct if sl_pct > 0 else score_settings.BASE_RRR
    return sl_price, tp_price, sl_pct, tp_pct, rrr


def open_entry_from_candidate(api, symbol: str, candidate: EntryCandidate, snapshot, state: SymbolState, score_settings, logger) -> None:
    """
    Ejecuta la orden de apertura para un `EntryCandidate` seleccionado.

    - Extrae los datos necesarios del candidato y del snapshot del mercado.
    - Calcula los precios de Take Profit y Stop Loss.
    - Determina el tamaño de la posición (notional).
    - Llama a `open_core_position` para enviar la orden al exchange.
    - Actualiza el estado del símbolo con la nueva posición.
    """
    trend_sig: TrendSignal | None = getattr(snapshot, "trend_signal", None)
    current_price = getattr(snapshot, "close", None) or getattr(snapshot, "price", None)
    if current_price is None:
        logger.info("[%s] ENTRY skipped: no price in snapshot", symbol)
        return

    pair_cfg = PAIR_CONFIGS[symbol]
    atr_pct = float(getattr(snapshot, "atr_pct", 0.0) or 0.0)
    risk_mult = float(getattr(snapshot, "risk_mult", 1.0) or 1.0)
    bar_index = getattr(snapshot, "bar_index", 0)
    volatility_regime = getattr(snapshot, "volatility_regime", None)
    regime_state = getattr(snapshot, "regime_state", None)
    psar_dir = getattr(snapshot, "psar_dir", 0)
    short_dir = getattr(snapshot, "short_dir", 0)
    effective_rr = float(getattr(snapshot, "effective_rr", 0.0) or 0.0)
    atr_pct_5m = float(getattr(snapshot, "atr_pct_5m", 0.0) or 0.0)

    entry_direction = 1 if candidate.direction == "LONG" else -1
    raw_regime_label = getattr(trend_sig, "regime", None) or getattr(regime_state, "trend", None)
    regime_label = raw_regime_label.upper() if isinstance(raw_regime_label, str) else "N/A"
    macro_regime_value = raw_regime_label.lower() if isinstance(raw_regime_label, str) else None
    psar_log_dir = getattr(trend_sig, "psar_30_dir", psar_dir)
    score_log = candidate.score
    if trend_sig and getattr(trend_sig, "atr_pct_30m", None):
        atr_pct = trend_sig.atr_pct_30m / 100.0

    atr_pct_30m = candidate.atr_pct if candidate.atr_pct is not None else (
        trend_sig.atr_pct_30m if trend_sig else 0.0
    )
    sl_price, tp_price, sl_pct_pct, tp_pct_pct, rrr = build_sl_tp(
        side=candidate.direction,
        price=float(current_price),
        atr_pct_5m=float(getattr(snapshot, "atr_pct_5m", 0.0) or 0.0),
        atr_pct_30m=atr_pct_30m,
        score_settings=score_settings,
    )

    logger.info(
        "APERTURA ALWAYS_IN_MARKET %s | dir=%s | score=%.4f | price=%.6f | SL=%.6f (%.2f%%) | TP=%.6f (%.2f%%) | "
        "ATR=%.4f%% | ATR5m=%.3f%% | regime=%s | psar30=%d short_dir=%+d | risk_mult=%.2f | RRR=%.2f | notional=%.2f",
        symbol,
        "LONG" if entry_direction == 1 else "SHORT",
        score_log,
        current_price,
        sl_price,
        sl_pct_pct,
        tp_price,
        tp_pct_pct,
        (atr_pct or 0.0) * 100.0,
        atr_pct_5m,
        regime_label,
        psar_log_dir,
        short_dir,
        risk_mult,
        rrr,
        score_settings.BASE_NOTIONAL * candidate.risk_mult,
    )

    notional = score_settings.BASE_NOTIONAL * candidate.risk_mult

    position = open_core_position(
        api,
        symbol,
        pair_cfg,
        entry_direction,
        tp_pct_pct / 100.0,
        sl_pct_pct / 100.0,
        current_price,
        bar_index,
        atr_pct,
        volatility_regime,
        risk_mult,
        override_notional=notional,
        entry_score=candidate.score,
        score_macro=candidate.score_macro,
        score_micro=candidate.score_micro,
        trf_state=candidate.trf_state,
        atr_pct_5m_entry=candidate.atr_pct_5m,
        macro_regime_entry=macro_regime_value,
    )
    state.position = position
    state.last_bias_flip_bar = bar_index
    state.current_direction = entry_direction


def run_symbol_step(api, symbol: str, bar: pd.Series, df_1m: pd.DataFrame, df_30m: pd.DataFrame, df_4h: pd.DataFrame, state: SymbolState) -> SimpleNamespace | None:
    """
    Ejecuta una iteración completa de la lógica de trading para un único símbolo.

    Esta es la función central del motor para cada par.
    1. Sincroniza el estado con el exchange para detectar posiciones externas.
    2. Calcula todos los indicadores y regímenes de mercado necesarios.
    3. Si hay una posición abierta, evalúa las condiciones de salida (ej. stop loss, cambio de tendencia).
    4. Si no hay posición, se prepara para la planificación de entradas.
    5. Devuelve un "snapshot" con el estado actual y los datos calculados para el planificador.
    """
    current_price = bar.close
    current_bar_index = bar.name  # Asumimos que el índice es el número de la vela

    # --- 0. Sync con exchange: adoptar posiciones, no cerrarlas ---
    exchange_pos = _fetch_exchange_position(api, symbol)
    sync_symbol_state_from_exchange(symbol, state, exchange_pos, current_bar_index, _logger)

    pair_cfg = PAIR_CONFIGS[symbol]

    # --- 1. Recopilar datos y calcular regímenes ---
    spread_bps = api.compute_spread_bps(symbol)
    last_volume = bar.volume
    atr_pct = api.compute_atr_pct(symbol)
    volatility_regime = classify_regime(atr_pct, pair_cfg)
    base_tp_pct, base_sl_pct = compute_tp_sl(atr_pct, volatility_regime, pair_cfg, spread_bps)

    # Calcular PSAR 30m
    psar_state = api.compute_psar_state(symbol)
    psar_dir = _psar_direction(getattr(psar_state, "trend", ""))
    
    strategy_settings = api.get_strategy_settings(symbol)
    short_dir, short_slope, short_strength = _compute_short_term_trend(
        df_1m,
        fast=strategy_settings.micro_monitor_ema_fast,
        slow=strategy_settings.micro_monitor_ema_slow,
        confirm_bars=strategy_settings.micro_monitor_confirm_bars,
    )

    # OHLC 5m para score MB/TRF y ATR% 5m en porcentaje
    ohlc_5m_records: list[dict] = []
    atr_pct_5m = 0.0
    trf_state_5m = None
    trf_dir_5m = None
    slope_5m = None
    try:
        df_exec_sorted = df_1m.sort_values("timestamp")
        df_5m = (
            df_exec_sorted.set_index("timestamp")
            .resample("5min", label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        if not df_5m.empty:
            atr_series_5m = atr(df_5m, period=strategy_settings.atr_period)
            last_close_5m = float(df_5m["close"].iloc[-1])
            last_atr_5m = float(atr_series_5m.iloc[-1]) if not atr_series_5m.empty else 0.0
            if last_close_5m > 0:
                atr_pct_5m = (last_atr_5m / last_close_5m) * 100.0
            ohlc_5m_records = (
                df_5m.tail(120)[["open", "high", "low", "close"]]
                .reset_index(drop=True)
                .to_dict("records")
            )
            closes_5m = [float(c["close"]) for c in ohlc_5m_records if c.get("close") is not None]
            if closes_5m:
                trf_snapshot = compute_twin_range_filter(
                    closes=closes_5m,
                    fast_period=27,
                    fast_range=1.6,
                    slow_period=55,
                    slow_range=2.0,
                )
                trf_state_5m = trf_snapshot.color
                trf_dir_5m = trf_snapshot.dir
                slope_5m = trf_snapshot.slope
    except Exception as exc:
        _logger.debug("[%s] failed to build 5m OHLC for MB/TRF: %s", symbol, exc)
    
    # Calcular regime con sistema 30m
    regime_state = compute_regime_state(df_30m, strategy_settings.regime)

    adx_val = None
    try:
        adx_series = adx(df_30m, period=strategy_settings.atr_period)
        if not adx_series.empty:
            adx_val = float(adx_series.iloc[-1])
    except Exception as exc:
        _logger.debug("[%s] ADX computation failed: %s", symbol, exc)

    price_4h = None
    ema200_4h = None
    try:
        if df_4h is not None and not df_4h.empty:
            price_4h = float(df_4h["close"].iloc[-1])
            if len(df_4h) >= 200:
                ema_series_4h = ema(df_4h["close"], 200)
                ema200_4h = float(ema_series_4h.iloc[-1])
    except Exception as exc:
        _logger.debug("[%s] 4H data processing failed: %s", symbol, exc)

    indicator_state = _build_indicator_state(symbol, bar, df_1m, df_30m, strategy_settings, trf_state_5m=trf_state_5m)
    trend_sig: TrendSignal | None = None
    if indicator_state:
        try:
            trend_sig = compute_symbol_trend_signal(symbol, indicator_state)
        except Exception as exc:
            _logger.debug("[%s] TrendSignal build failed: %s", symbol, exc)
    
    _logger.debug(
        "[%s] Regime: trend=%s, bias=%+d, PSAR30m=%s (%+d), short_dir=%+d slope=%.6f strength=%.2f",
        symbol, regime_state.trend, regime_state.bias, psar_state.trend, psar_dir, short_dir, short_slope, short_strength
    )

    # Determinar dirección deseada
    last_direction = state.position.direction if state.position else 0
    desired_dir, entry_mode, direction_score = _determine_entry_direction(
        regime_state, psar_dir, short_dir, short_strength, volatility_regime, spread_bps, last_direction
    )
    risk_mult = _risk_mult_from_score(direction_score, volatility_regime)

    direction_for_targets = desired_dir if desired_dir != 0 else 1
    direction_label = "LONG" if direction_for_targets >= 0 else "SHORT"
    
    sl_price, tp_price, effective_rr = compute_adaptive_tp_sl(
        entry_price=float(current_price),
        direction=direction_label,
        atr_pct_5m=atr_pct_5m,
        base_sl_frac=BASE_SL_FRAC,
        base_tp_rr=BASE_TP_RR,
        fees_frac=FEE_FRAC,
        risk_mult=risk_mult,          # <<< NUEVO
    )
    
    _logger.info(
        "SL/TP adaptados %s | SL=%.6f (%.2f%%) | TP=%.6f (%.2f%%) | ATR%%=%.2f | R:R=%.2f",
        symbol,
        sl_price, (abs(sl_price - current_price) / current_price) * 100.0,
        tp_price, (abs(tp_price - current_price) / current_price) * 100.0,
        atr_pct_5m,
        effective_rr,
    )

    tp_pct = abs(tp_price - current_price) / current_price
    sl_pct = abs(sl_price - current_price) / current_price
    rrr = (tp_pct / sl_pct) if sl_pct > 0 else 0.0

    # Log de cálculo completo en tiempo real
    twin_mid = getattr(regime_state, "twin_mid", None)
    twin_up = getattr(regime_state, "twin_upper", None)
    twin_low = getattr(regime_state, "twin_lower", None)
    twin_slope = getattr(regime_state, "twin_slope", None)
    in_band = getattr(regime_state, "in_band", None)
    diag_tp = bar.close * (1 + tp_pct) if desired_dir == 1 else bar.close * (1 - tp_pct)
    diag_sl = bar.close * (1 - sl_pct) if desired_dir == 1 else bar.close * (1 + sl_pct)
    pos_label = "FLAT"
    pos_qty = 0.0
    pos_entry = 0.0
    if state.position:
        pos_label = "LONG" if state.position.direction == 1 else "SHORT"
        pos_qty = state.position.qty
        pos_entry = state.position.entry_price
    _logger.info(
        "[%s] Calc price=%.6f | bias=%+d trend=%s psar=%s short=%+d(%.2f) score=%+.2f dir_now=%s dir_next=%s | ATR=%.4f%% spread=%.2fbps | twin[mid=%s up=%s low=%s slope=%s in_band=%s] | TP@%.6f (%.2f%%) SL@%.6f (%.2f%%) | risk_mult=%.2f | pos=%s qty=%.4f entry=%.6f",
        symbol,
        bar.close,
        regime_state.bias,
        regime_state.trend,
        psar_state.trend,
        short_dir,
        short_strength,
        direction_score,
        pos_label,
        "LONG" if desired_dir == 1 else "SHORT",
        atr_pct * 100.0,
        spread_bps,
        f"{twin_mid:.6f}" if twin_mid is not None else "n/a",
        f"{twin_up:.6f}" if twin_up is not None else "n/a",
        f"{twin_low:.6f}" if twin_low is not None else "n/a",
        f"{twin_slope:.6f}" if twin_slope is not None else "n/a",
        in_band if in_band is not None else "n/a",
        diag_tp,
        tp_pct * 100.0,
        diag_sl,
        sl_pct * 100.0,
        risk_mult,
        pos_label,
        pos_qty,
        pos_entry,
    )

    if direction_score == 0:
        fallback_label = "LONG" if desired_dir == 1 else "SHORT"
        _logger.debug(
            "[%s] direction score=0, falling back to last direction: %s",
            symbol,
            fallback_label,
        )

    # --- 2. Salidas ---
    exit_reason = None
    if state.position:
        pos = state.position
        bars_in_trade = current_bar_index - pos.open_bar_index
        exit_flag = False
        exit_reason = None

        ensure_stop_loss_order(api, symbol, pos, strategy_settings)

        if trend_sig and getattr(trend_sig, "score_micro", None) is not None:
            micro_score_for_exit = float(trend_sig.score_micro)
        else:
            normalized_strength = min(1.0, abs(short_strength) / 2.5) if short_strength is not None else 0.0
            micro_score_for_exit = (short_dir or 0) * normalized_strength
        ema_slope_deg_1m = math.degrees(math.atan(short_slope)) if short_slope is not None else 0.0
        update_position_telemetry_from_market(
            pos=pos,
            last_price=current_price,
            micro_score=micro_score_for_exit,
            ema_slope_deg_1m=ema_slope_deg_1m,
        )

        if trend_sig:
            exit_flag, exit_reason = should_exit_with_trend_signal(symbol, pos, trend_sig, _logger)

        if not exit_flag and should_trend_exit_slowing(
            side=pos.direction,
            bars_in_trade=bars_in_trade,
            entry_price=pos.entry_price,
            last_price=current_price,
            trf_state_5m=trf_state_5m,
            trf_dir_5m=trf_dir_5m,
            slope_5m=slope_5m,
        ):
            exit_flag = True
            exit_reason = "trend_exit_slowing"

        if exit_flag and exit_reason:
            _logger.info(
                "[%s] EXIT decision: reason=%s | dir=%s | score=%+.2f | regime=%s | bars_in_trade=%d",
                symbol,
                exit_reason,
                "LONG" if pos.direction == 1 else "SHORT",
                direction_score,
                volatility_regime.name,
                bars_in_trade,
            )
            close_position_safely(api, symbol, pos, reason=exit_reason)
            _log_and_reset(state, symbol, current_price, current_bar_index, exit_reason, decision_score=direction_score, regime_label=volatility_regime.name)
            state.pause_until_bar = max(state.pause_until_bar, current_bar_index + pair_cfg.min_bars_after_flip)
            state.current_direction = 0

    # --- 4. Escalado ---
    if state.position is not None and volatility_regime.name == "NORMAL":
        reentry_dist = compute_reentry_distance(atr_pct, volatility_regime)
        old_qty = state.position.qty
        maybe_scale_in(api, symbol, pair_cfg, state.position, volatility_regime, current_price, reentry_dist)
        if state.position.qty > old_qty:
            _logger.info(f"SCALE-IN {symbol} | Qty: {old_qty:.4f} -> {state.position.qty:.4f} | Price: {current_price:.6f}")

    # Estructura de snapshot para planificación (datos tolerantes a ausencia)
    try:
        ema_fast_series = df_1m["close"].ewm(span=strategy_settings.micro_monitor_ema_fast, adjust=False).mean()
        ema_slow_series = df_1m["close"].ewm(span=strategy_settings.micro_monitor_ema_slow, adjust=False).mean()
        ema_fast_val = float(ema_fast_series.iloc[-1]) if not ema_fast_series.empty else None
        ema_slow_val = float(ema_slow_series.iloc[-1]) if not ema_slow_series.empty else None
    except Exception:
        ema_fast_val = None
        ema_slow_val = None

    atr_pct_30m = None
    if trend_sig and getattr(trend_sig, "atr_pct_30m", None) is not None:
        atr_pct_30m = float(trend_sig.atr_pct_30m)
    elif atr_pct:
        atr_pct_30m = atr_pct * 100.0

    snapshot = SimpleNamespace(
        position=state.position,
        close=current_price,
        price=current_price,
        bias=getattr(regime_state, "bias", None),
        ema_fast=ema_fast_val,
        ema_slow=ema_slow_val,
        atr_pct=atr_pct,
        atr_pct_5m=atr_pct_5m,
        vol_regime=volatility_regime.name,
        volatility_regime=volatility_regime,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        effective_rr=effective_rr,
        risk_mult=risk_mult,
        bar_index=current_bar_index,
        psar_dir=psar_dir,
        short_dir=short_dir,
        short_strength=short_strength,
        regime_state=regime_state,
        trend_signal=trend_sig,
        ohlc_5m=ohlc_5m_records,
        trf_state_5m=trf_state_5m,
        adx_30m=adx_val,
        price_4h=price_4h,
        ema200_4h=ema200_4h,
        atr_pct_30m=atr_pct_30m,
    )

    return snapshot


def _log_and_reset(
    state: SymbolState,
    symbol: str,
    current_price: float,
    current_bar_index: int,
    reason: ExitReason,
    decision_score: float | None = None,
    regime_label: str | None = None,
) -> None:
    position = state.position
    if position is None:
        return

    pair_cfg = PAIR_CONFIGS[symbol]
    gross_pnl = (current_price - position.entry_price) * position.qty * position.direction
    fee_rate = pair_cfg.fee_bps / 10000.0
    fees = (position.entry_price * position.qty + current_price * position.qty) * fee_rate
    net_pnl = gross_pnl - fees
    bars_in_trade = position.bars_held if position.bars_held > 0 else max(0, current_bar_index - position.open_bar_index)
    
    side_str = "LONG" if position.direction == 1 else "SHORT"
    pnl_pct = (net_pnl / (position.entry_price * position.qty)) * 100 if position.qty > 0 else 0
    pnl_str = "PROFIT" if net_pnl > 0 else "LOSS" if net_pnl < 0 else "BREAK-EVEN"
    _logger.info(
        f"CIERRE {symbol} [{pnl_str}] | {side_str} | Entry: {position.entry_price:.6f} -> Exit: {current_price:.6f} | "
        f"PnL: ${net_pnl:.2f} ({pnl_pct:+.2f}%) | Razón: {reason} | Barras: {bars_in_trade} | "
        f"Régimen: {position.regime_entry} | score_exit={decision_score if decision_score is not None else 'n/a'} | "
        f"regime_now={regime_label or 'n/a'}"
    )

    if reason in ("sl", "trailing_sl"):
        last_sl_ts[symbol] = time.time()

    trade = TradeLogEntry(
        timestamp_open=position.open_timestamp,
        timestamp_close=datetime.utcnow().isoformat(),
        pair=symbol,
        direction=position.direction,
        entry_price=position.entry_price,
        exit_price=current_price,
        gross_pnl=gross_pnl,
        fees=fees,
        net_pnl=net_pnl,
        atr_pct_entry=position.atr_pct_entry,
        regime_entry=position.regime_entry,
        exit_reason=reason,
        spread_bps=0, # spread_bps no está disponible aquí, necesita pasarse
        bars_in_trade=bars_in_trade,
        entry_score=position.entry_score,
        score_macro=position.score_macro,
        score_micro=position.score_micro,
        regime_30m=position.macro_regime_entry or position.regime_entry,
        trf_state_5m=getattr(position, "trf_state_entry", None),
        atr_pct_30m=position.atr_pct_entry,
        atr_pct_5m=getattr(position, "atr_pct_5m_entry", None),
    )
    append_trade_log(trade)
    update_loss_streak_and_daily_pnl(state, net_pnl, pair_cfg, current_bar_index, current_price)
    on_position_closed(symbol, current_bar_index, PLANNER_STATE, _logger)
    state.position = None
    state.last_close_ts = time.time()
    state.last_exit_ts = state.last_close_ts
