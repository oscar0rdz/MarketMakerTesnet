from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _get_pct(key: str, default: float) -> float:
    val = _get_float(key, default)
    return val / 100.0 if val > 1 else val


def _split_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")

@dataclass
class PairConfig:
    symbol: str
    fee_bps: float           # comisión total ida+vuelta en bps, ej. 2.0 = 0.02%
    atr_off: float           # por debajo de esto (en %) no operas (rango MUY estrecho)
    atr_micro: float         # fin de régimen MICRO → pasa a NORMAL
    atr_normal: float        # fin de régimen NORMAL → EXPLOSIVE
    atr_micro_max: float
    atr_normal_max: float
    atr_explosive_min: float
    max_bars_micro: int      # stop por tiempo en régimen MICRO
    max_bars_normal: int     # stop por tiempo en NORMAL
    max_bars_explosive: int  # stop por tiempo en EXPLOSIVE
    base_size_usdt: float    # tamaño base de la posición (notional en USDT)
    scale_size_usdt: float   # tamaño de cada scale-in (notional en USDT)
    max_spread_bps: float
    min_1m_volume: float
    max_wick_ratio: float
    min_bars_after_flip: int
    max_loss_streak: int
    max_daily_loss_pct: float
    slippage_cost_multiplier: float


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default

DEFAULT_BASE_SIZE_USDT = _env_float("PAIR_BASE_SIZE_USDT", float(os.getenv("TRADE_NOTIONAL_USDT", "30.0")))
DEFAULT_SCALE_SIZE_USDT = _env_float("PAIR_SCALE_SIZE_USDT", float(os.getenv("TRADE_NOTIONAL_USDT", "30.0")))

# === SCORING / SNIPER LOGIC ===

# Periodos estándar
ATR_PERIOD = 14
RSI_PERIOD = 14

# ---------------------------
# SNIPER SCORE / TREND EXIT
# ---------------------------
# Pesos Macro/Micro
MACRO_WEIGHT = 0.60
MICRO_WEIGHT = 0.40

# Umbrales y buckets de score
MIN_SLOPE_DEG = 12.0
SCORE_THRESHOLD = 0.25
SCORE_STRONG = 0.75
SCORE_GOOD = 0.50
SCORE_MEDIUM = 0.30

# --- SNIPER / TREND EXIT KNOBS ---
# Barras mínimas antes de permitir trend_exit_slowing
TREND_EXIT_MIN_BARS = 3
# No cortar si la operación todavía está mejor que este R/R
TREND_EXIT_LOCK_MIN_R = -0.35
# Mientras el micro score esté por encima de este umbral (en valor absoluto) se considera fuerte
TREND_EXIT_MIN_MICRO_ABS = 0.18
# Momentum colapsado si cae por debajo de este valor
TREND_EXIT_COLLAPSE_MICRO_ABS = 0.08
# Pendiente mínima (en grados) para considerar plana la EMA 1m
TREND_EXIT_FLAT_SLOPE_DEG = 3.0

# Parámetros de R:R dinámicos
RRR_BASE = 1.30
RRR_STRONG_MAX = 1.70
RRR_WEAK_MIN = 1.15

# Penalización TRF
TRF_RANGE_PENALTY = 0.50

# Límite de correlación
MAX_OPEN_PER_CLUSTER = 2
CLUSTER_MAP = {
    "BNBUSDT": "LARGE_CAP",
    "LTCUSDT": "LARGE_CAP",
    "LINKUSDT": "LARGE_CAP",
    "XRPUSDT": "ALT_CLUSTER",
    "XLMUSDT": "ALT_CLUSTER",
    "ADAUSDT": "ALT_CLUSTER",
    "DOGEUSDT": "ALT_CLUSTER",
    "TRXUSDT": "ALT_CLUSTER",
}


# Parámetros por par (AJÚSTALOS tras ver backtests)
PAIR_CONFIGS = {
    "XRPUSDT": PairConfig(
        symbol="XRPUSDT",
        fee_bps=_env_float("XRPUSDT_FEE_BPS", 3.5),
        atr_off=_env_float("XRPUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("XRPUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("XRPUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("XRPUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("XRPUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("XRPUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("XRPUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("XRPUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=4.0,
        min_1m_volume=50_000.0,
        max_wick_ratio=3.5,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "BNBUSDT": PairConfig(
        symbol="BNBUSDT",
        fee_bps=_env_float("BNBUSDT_FEE_BPS", 3.0),
        atr_off=_env_float("BNBUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("BNBUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("BNBUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("BNBUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("BNBUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("BNBUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("BNBUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("BNBUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=3.0,
        min_1m_volume=30_000.0,
        max_wick_ratio=3.0,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "LTCUSDT": PairConfig(
        symbol="LTCUSDT",
        fee_bps=_env_float("LTCUSDT_FEE_BPS", 3.5),
        atr_off=_env_float("LTCUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("LTCUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("LTCUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("LTCUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("LTCUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("LTCUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("LTCUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("LTCUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=4.0,
        min_1m_volume=50_000.0,
        max_wick_ratio=3.5,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "LINKUSDT": PairConfig(
        symbol="LINKUSDT",
        fee_bps=_env_float("LINKUSDT_FEE_BPS", 3.0),
        atr_off=_env_float("LINKUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("LINKUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("LINKUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("LINKUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("LINKUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("LINKUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("LINKUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("LINKUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=3.0,
        min_1m_volume=30_000.0,
        max_wick_ratio=3.0,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "ADAUSDT": PairConfig(
        symbol="ADAUSDT",
        fee_bps=_env_float("ADAUSDT_FEE_BPS", 3.5),
        atr_off=_env_float("ADAUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("ADAUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("ADAUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("ADAUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("ADAUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("ADAUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("ADAUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("ADAUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=4.0,
        min_1m_volume=30_000.0,
        max_wick_ratio=3.5,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "DOGEUSDT": PairConfig(
        symbol="DOGEUSDT",
        fee_bps=_env_float("DOGEUSDT_FEE_BPS", 3.5),
        atr_off=_env_float("DOGEUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("DOGEUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("DOGEUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("DOGEUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("DOGEUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("DOGEUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("DOGEUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("DOGEUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=5.0,
        min_1m_volume=80_000.0,
        max_wick_ratio=3.5,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "TRXUSDT": PairConfig(
        symbol="TRXUSDT",
        fee_bps=_env_float("TRXUSDT_FEE_BPS", 3.0),
        atr_off=_env_float("TRXUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("TRXUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("TRXUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("TRXUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("TRXUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("TRXUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("TRXUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("TRXUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=4.0,
        min_1m_volume=50_000.0,
        max_wick_ratio=3.5,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
    "XLMUSDT": PairConfig(
        symbol="XLMUSDT",
        fee_bps=_env_float("XLMUSDT_FEE_BPS", 3.5),
        atr_off=_env_float("XLMUSDT_ATR_OFF", 0.10),
        atr_micro=_env_float("XLMUSDT_ATR_MICRO", 0.25),
        atr_normal=_env_float("XLMUSDT_ATR_NORMAL", 0.80),
        atr_micro_max=0.10,
        atr_normal_max=0.25,
        atr_explosive_min=0.80,
        max_bars_micro=_env_int("XLMUSDT_MAX_BARS_MICRO", 12),
        max_bars_normal=_env_int("XLMUSDT_MAX_BARS_NORMAL", 28),
        max_bars_explosive=_env_int("XLMUSDT_MAX_BARS_EXPLOSIVE", 8),
        base_size_usdt=_env_float("XLMUSDT_BASE_SIZE_USDT", DEFAULT_BASE_SIZE_USDT),
        scale_size_usdt=_env_float("XLMUSDT_SCALE_SIZE_USDT", DEFAULT_SCALE_SIZE_USDT),
        max_spread_bps=5.0,
        min_1m_volume=40_000.0,
        max_wick_ratio=3.5,
        min_bars_after_flip=2,
        max_loss_streak=4,
        max_daily_loss_pct=4.0,
        slippage_cost_multiplier=4.0,
    ),
}

ACTIVE_PAIRS = [
    p.strip()
    for p in os.getenv(
        "SCALPER_ACTIVE_PAIRS",
        "XRPUSDT,BNBUSDT,LTCUSDT,LINKUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,XLMUSDT"
    ).split(",")
    if p.strip()
]

TRADE_LOG_PATH = "logs/trades.csv"

@dataclass
class RegimeSettings:
    enabled: bool = True          # activar/desactivar filtro de régimen
    ha_bias_len: int = 60         # periodo base Market Bias (velas 30m)
    ha_osc_len: int = 7           # oscilador HA
    ha_smooth: int = 10           # suavizado extra

    tr_fast_len: int = 27         # Twin Range: EMA rápida
    tr_slow_len: int = 55         # Twin Range: EMA lenta
    tr_atr_len: int = 14          # Twin Range: ATR bandas
    tr_band_mult: float = 1.5     # Twin Range: multiplicador ATR


@dataclass
class StrategySettings:
    sar_step: float = 0.02
    sar_max_step: float = 0.2
    sar_fast_step: float = 0.03
    sar_fast_max_step: float = 0.25
    sar_flip_confirmation: int = 2
    ema_period: int = 200
    ema_fast_period: int = 50
    ema_slope_lookback: int = 3
    ema_slope_min: float = 0.00015
    atr_period: int = 14
    atr_ema_period: int = 100
    atr_floor_pct: float = 0.0008
    min_history: int = 220
    bollinger_period: int = 20
    bollinger_stddev: float = 2.0
    rsi_period: int = 14
    rsi_overbought: float = 65.0
    rsi_oversold: float = 35.0
    range_band_tolerance: float = 0.0015
    micro_momentum_window: int = 5
    micro_momentum_min_pct: float = 0.0005
    aggressive_psar_entries: bool = True
    micro_psar_step: float = 0.02
    micro_psar_max_step: float = 0.2
    micro_monitor_confirm_bars: int = 3
    micro_monitor_exit_pct: float = 0.0015
    micro_monitor_ema_fast: int = 9
    micro_monitor_ema_slow: int = 21
    sl_max_retries: int = 5
    sl_retry_tick_step: int = 1
    sl_retry_max_ticks: int = 10
    sl_require_hard: bool = True
    reentry_min_move_bps: float = 5.0    # mínimo 5 bps (~0.05%) para re-entrar

    micro_stop_pct: float = 0.0035       # stop micro de emergencia (~0.35%)
    hard_stop_pct: float = 0.0080        # hard stop global del monitor (~0.8%)
    micro_stop_min_adverse_pct: float = 0.005  # mínimo 0.5% de pérdida para activar micro-stop
    swing_move_atr_multiplier: float = 0.5     # multiplica ATR% para calcular mínimo dinámico de reentrada
    trend_exit_min_bars: int = 3
    trend_exit_lock_min_r: float = -0.35
    trend_exit_min_micro_abs: float = 0.18
    trend_exit_collapse_micro_abs: float = 0.08
    trend_exit_flat_slope_deg: float = 3.0
    trend_exit_min_favour_bps: float = 3.0
    regime: RegimeSettings = field(default_factory=RegimeSettings)


@dataclass
class RiskSettings:
    risk_per_trade: float = 0.75     # % de equity base por trade (0.75%)
    min_risk_pct: float = 0.25       # riesgo mínimo efectivo (0.25%)
    max_risk_pct: float = 1.50       # riesgo máximo efectivo (1.50%)
    micro_stop_pct: float = 0.35     # stop micro duro (~0.35% adverso)
    max_notional: Optional[float] = 50.0
    min_notional_buffer: float = 1.05
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    min_risk_mult: float = 0.6       # límite inferior del multiplicador
    max_risk_mult: float = 1.5       # límite superior del multiplicador


@dataclass
class ScoreSettings:
    """Ajustes de scoring y tamaño de posición."""

    ADX_TREND_LEVEL: float = 30.0
    ADX_RANGE_LEVEL: float = 20.0

    MACRO_WEIGHT_TREND: float = 0.60
    MICRO_WEIGHT_TREND: float = 0.40

    MACRO_WEIGHT_RANGE: float = 0.10
    MICRO_WEIGHT_RANGE: float = 0.90

    BULL_LONG_BIAS: float = 1.00
    BULL_SHORT_BIAS: float = 0.92
    BEAR_LONG_BIAS: float = 0.90
    BEAR_SHORT_BIAS: float = 1.05

    ATR_LOW_PCT: float = 0.40
    ATR_HIGH_PCT: float = 2.50

    ATR_LOW_SIZE_MULT: float = 0.50
    ATR_MID_SIZE_MULT: float = 1.00
    ATR_HIGH_SIZE_MULT: float = 0.30

    ATR_LOW_SCORE_MULT: float = 0.95
    ATR_MID_SCORE_MULT: float = 1.00
    ATR_HIGH_SCORE_MULT: float = 0.90

    BASE_RRR: float = 1.45
    MIN_RRR: float = 1.40

    MIN_SCORE_FOR_SIZE: float = 0.10
    MAX_SCORE_FOR_SIZE: float = 0.60

    MIN_RISK_MULT: float = 0.25
    MAX_RISK_MULT: float = 0.85

    BASE_NOTIONAL: float = 70.0


@dataclass
class EngineSettings:
    signal_timeframe: str = "30m"
    execution_timeframe: str = "1m"
    macro_timeframe: str = "4h"
    history_limit: int = 400
    monitor_history_limit: int = 400
    poll_seconds: float = 20.0
    equity_refresh_sec: float = 45.0
    position_sync_sec: float = 90.0
    idle_log_interval_sec: float = 60.0
    log_idle_snapshots: bool = True
    micro_reentry_cooldown_sec: float = 45.0
    always_in_market_target_min: int = 4
    always_in_market_max_open: int = 5
    score_threshold: float = 0.25
    score_threshold_soft: float = 0.18
    soft_fill_max_empty_iters: int = 5
    symbol_cooldown_bars: int = 3
    cooldown_force_score: float = 0.32
    min_bars_before_micro_exit: int = 3
    min_micro_profit_bps: int = 8
    min_rrr: float = ScoreSettings().MIN_RRR


@dataclass
class ScalpSettings:
    enabled: bool = True
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 1.8
    max_concurrent: int = 2
    max_hold_minutes: float = 8.0
    risk_pct: float = 0.003


@dataclass
class ScannerSettings:
    enabled: bool = False
    refresh_interval_sec: int = 1800  # 30 minutes
    top_n: int = 10
    min_volume: float = 10_000_000.0  # 10M USDT
    denylist: Optional[List[str]] = None
    candidates: Optional[List[str]] = None
    collateral_asset: str = "USDT"


@dataclass
class BotConfig:
    broker: str
    api_mode: str
    api_key: Optional[str]
    api_secret: Optional[str]
    symbols: List[str]
    engine: EngineSettings
    strategy: StrategySettings
    risk: RiskSettings
    scalp: ScalpSettings
    scanner: ScannerSettings
    collateral_asset: str
    score: ScoreSettings
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "BotConfig":
        broker = _get_env("BROKER", "binanceusdm")
        api_mode = _get_env("API_MODE", "testnet").lower()
        api_key = os.getenv("API_KEY") or os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_APIKEY")
        api_secret = os.getenv("API_SECRET") or os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_APISECRET")
        symbols = _split_symbols(_get_env("SYMBOLS", "SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,NEOUSDT,ADAUSDT"))

        strategy = StrategySettings(
            sar_step=_get_float("SAR_STEP", 0.02),
            sar_max_step=_get_float("SAR_MAX_STEP", 0.2),
            sar_fast_step=_get_float("SAR_FAST_STEP", 0.03),
            sar_fast_max_step=_get_float("SAR_FAST_MAX_STEP", 0.25),
            sar_flip_confirmation=_get_int("SAR_FLIP_CONFIRMATION", 2),
            ema_period=_get_int("EMA_PERIOD", 200),
            ema_fast_period=_get_int("EMA_FAST_PERIOD", 50),
            ema_slope_lookback=_get_int("EMA_SLOPE_LOOKBACK", 3),
            ema_slope_min=_get_pct("EMA_SLOPE_MIN", 0.00005),
            atr_period=_get_int("ATR_PERIOD", 14),
            atr_ema_period=_get_int("ATR_EMA_PERIOD", 100),
            atr_floor_pct=_get_float("ATR_FLOOR_PCT", 0.0008),
            min_history=_get_int("MIN_HISTORY", 220),
            bollinger_period=_get_int("BOLLINGER_PERIOD", 20),
            bollinger_stddev=_get_float("BOLLINGER_STDDEV", 2.0),
            rsi_period=_get_int("RSI_PERIOD", 14),
            rsi_overbought=_get_float("RSI_OVERBOUGHT", 65.0),
            rsi_oversold=_get_float("RSI_OVERSOLD", 35.0),
            range_band_tolerance=_get_pct("RANGE_BAND_TOLERANCE", 0.0015),
            micro_momentum_window=_get_int("MICRO_MOMENTUM_WINDOW", 5),
            micro_momentum_min_pct=_get_pct("MICRO_MOMENTUM_MIN_PCT", 0.0002),
            aggressive_psar_entries=_get_bool("AGGRESSIVE_PSAR_ENTRIES", True),
            micro_psar_step=_get_float("MICRO_PSAR_STEP", 0.02),
            micro_psar_max_step=_get_float("MICRO_PSAR_MAX_STEP", 0.2),
            micro_monitor_confirm_bars=_get_int("MICRO_MONITOR_CONFIRM_BARS", 3),
            micro_monitor_exit_pct=_get_pct("MICRO_MONITOR_EXIT_PCT", 0.0015),
            micro_monitor_ema_fast=_get_int("MICRO_MONITOR_EMA_FAST", 9),
            micro_monitor_ema_slow=_get_int("MICRO_MONITOR_EMA_SLOW", 21),
            sl_max_retries=_get_int("SL_MAX_RETRIES", 5),
            sl_retry_tick_step=_get_int("SL_RETRY_TICK_STEP", 1),
            sl_retry_max_ticks=_get_int("SL_RETRY_MAX_TICKS", 10),
            sl_require_hard=_get_bool("SL_REQUIRE_HARD", True),
            trend_exit_min_bars=_get_int("TREND_EXIT_MIN_BARS", 3),
            trend_exit_lock_min_r=_get_float("TREND_EXIT_LOCK_MIN_R", -0.35),
            trend_exit_min_micro_abs=_get_float("TREND_EXIT_MIN_MICRO_ABS", 0.18),
            trend_exit_collapse_micro_abs=_get_float("TREND_EXIT_COLLAPSE_MICRO_ABS", 0.08),
            trend_exit_flat_slope_deg=_get_float("TREND_EXIT_FLAT_SLOPE_DEG", 3.0),
            trend_exit_min_favour_bps=_get_float("TREND_EXIT_MIN_FAVOUR_BPS", 3.0),
        )

        risk = RiskSettings(
            risk_per_trade=_get_float("RISK_PER_TRADE", 0.75),
            max_notional=_get_float("MAX_NOTIONAL", 50.0) or None,
            min_notional_buffer=_get_float("MIN_NOTIONAL_BUFFER", 1.05),
            sl_atr_mult=_get_float("SL_ATR_MULT", 1.5),
            tp_atr_mult=_get_float("TP_ATR_MULT", 3.0),
        )

        min_history_needed = max(strategy.min_history, strategy.ema_period + 5, strategy.atr_period + 5)
        engine = EngineSettings(
            signal_timeframe=_get_env("SIGNAL_TIMEFRAME", "30m"),
            execution_timeframe=_get_env("EXECUTION_TIMEFRAME", "1m"),
            macro_timeframe=_get_env("MACRO_TIMEFRAME", "4h"),
            history_limit=_get_int("HISTORY_LIMIT", max(400, min_history_needed + 50)),
            monitor_history_limit=_get_int("MONITOR_HISTORY_LIMIT", 200),
            poll_seconds=_get_float("POLL_SECONDS", 20.0),
            equity_refresh_sec=_get_float("EQUITY_REFRESH_SEC", 45.0),
            position_sync_sec=_get_float("POSITION_SYNC_SEC", 90.0),
            idle_log_interval_sec=_get_float("IDLE_LOG_INTERVAL_SEC", 60.0),
            log_idle_snapshots=_get_env("LOG_IDLE_SNAPSHOTS", "1").lower() not in ("0", "false", "no", "off"),
            micro_reentry_cooldown_sec=_get_float("MICRO_REENTRY_COOLDOWN_SEC", 45.0),
        )

        scalp = ScalpSettings(
            enabled=_get_bool("SCALP_ENABLED", True),
            ema_fast=_get_int("SCALP_EMA_FAST", 9),
            ema_slow=_get_int("SCALP_EMA_SLOW", 21),
            atr_period=_get_int("SCALP_ATR_PERIOD", 14),
            sl_atr_mult=_get_float("SCALP_SL_ATR_MULT", 1.0),
            tp_atr_mult=_get_float("SCALP_TP_ATR_MULT", 1.8),
            max_concurrent=_get_int("SCALP_MAX_CONCURRENT", 2),
            max_hold_minutes=_get_float("SCALP_MAX_HOLD_MINUTES", 8.0),
            risk_pct=_get_pct("SCALP_RISK_PCT", 0.003),
        )

        scanner = ScannerSettings(
            enabled=_get_bool("SCANNER_ENABLED", False),
            refresh_interval_sec=_get_int("SCANNER_REFRESH_INTERVAL_SEC", 1800),
            top_n=_get_int("SCANNER_TOP_N", 10),
            min_volume=_get_float("SCANNER_MIN_VOLUME", 10_000_000.0),
            denylist=_split_symbols(_get_env("SCANNER_DENYLIST", "")),
            candidates=_split_symbols(_get_env("SCANNER_CANDIDATES", "")),
            collateral_asset=_get_env("COLLATERAL_ASSET", "USDT").upper(),
        )

        collateral_asset = _get_env("COLLATERAL_ASSET", "USDT").upper()
        log_level = _get_env("LOG_LEVEL", "INFO").upper()

        score = ScoreSettings()

        return cls(
            broker=broker,
            api_mode=api_mode,
            api_key=api_key,
            api_secret=api_secret,
            symbols=symbols,
            engine=engine,
            strategy=strategy,
            risk=risk,
            scalp=scalp,
            scanner=scanner,
            collateral_asset=collateral_asset,
            log_level=log_level,
            score=score,
        )