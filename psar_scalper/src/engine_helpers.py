from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from .config import CLUSTER_MAP, PAIR_CONFIGS, TRADE_LOG_PATH

_logger = logging.getLogger(__name__)


ExitReason = str


@dataclass
class PositionState:
    symbol: str
    direction: int
    qty: float
    entry_price: float
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    trailing_sl_price: float = 0.0
    open_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    open_bar_index: int = 0
    bars_held: int = 0
    atr_pct_entry: float = 0.0
    regime_entry: str = "unknown"
    entry_score: Optional[float] = None
    score_macro: Optional[float] = None
    score_micro: Optional[float] = None
    macro_regime_entry: Optional[str] = None
    trf_state_entry: Optional[str] = None
    atr_pct_5m_entry: Optional[float] = None
    risk_mult: float = 1.0
    last_price: float = 0.0
    last_micro_score: float = 0.0
    last_ema_slope_deg: float = 0.0
    last_sl_refresh_ts: float = 0.0


@dataclass
class SymbolState:
    position: Optional[PositionState] = None
    pause_until_bar: int = 0
    last_bias_flip_bar: int = -10**9
    current_direction: int = 0
    loss_streak: int = 0
    daily_pnl: float = 0.0
    last_trade_day: Optional[date] = None
    last_close_ts: float = 0.0
    last_exit_ts: float = 0.0


@dataclass
class TradeLogEntry:
    timestamp_open: str
    timestamp_close: str
    pair: str
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    fees: float
    net_pnl: float
    atr_pct_entry: float
    regime_entry: str
    exit_reason: str
    spread_bps: float
    bars_in_trade: int
    entry_score: Optional[float]
    score_macro: Optional[float]
    score_micro: Optional[float]
    regime_30m: Optional[str]
    trf_state_5m: Optional[str]
    atr_pct_30m: Optional[float]
    atr_pct_5m: Optional[float]


TRADE_LOG_FIELDS = [
    "timestamp_open",
    "timestamp_close",
    "pair",
    "direction",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "fees",
    "net_pnl",
    "atr_pct_entry",
    "regime_entry",
    "exit_reason",
    "spread_bps",
    "bars_in_trade",
    "entry_score",
    "score_macro",
    "score_micro",
    "regime_30m",
    "trf_state_5m",
    "atr_pct_30m",
    "atr_pct_5m",
]


def get_cluster_for_symbol(symbol: str) -> str:
    return CLUSTER_MAP.get(symbol, "GENERIC")


def build_position_state_from_fill(
    *,
    symbol: str,
    direction: int,
    fill_price: float,
    filled_qty: float,
    tp_price: Optional[float],
    sl_price: Optional[float],
    now_ts: float,
    open_bar_index: int,
    atr_pct_entry: float,
    regime_entry: str,
    entry_score: Optional[float] = None,
    score_macro: Optional[float] = None,
    score_micro: Optional[float] = None,
    macro_regime_entry: Optional[str] = None,
) -> PositionState:
    pos = PositionState(
        symbol=symbol,
        direction=direction,
        qty=abs(filled_qty),
        entry_price=float(fill_price),
        tp_price=tp_price,
        sl_price=sl_price,
        trailing_sl_price=sl_price or fill_price,
        open_timestamp=datetime.utcfromtimestamp(now_ts).isoformat(),
        open_bar_index=open_bar_index,
        atr_pct_entry=float(atr_pct_entry or 0.0),
        regime_entry=regime_entry,
        entry_score=entry_score,
        score_macro=score_macro,
        score_micro=score_micro,
        macro_regime_entry=macro_regime_entry,
    )
    return pos


def open_core_position(
    api,
    symbol: str,
    pair_cfg,
    direction: int,
    tp_pct: float,
    sl_pct: float,
    current_price: float,
    bar_index: int,
    atr_pct: float,
    volatility_regime,
    risk_mult: float,
    override_notional: Optional[float] = None,
    entry_score: Optional[float] = None,
    score_macro: Optional[float] = None,
    score_micro: Optional[float] = None,
    trf_state: Optional[str] = None,
    atr_pct_5m_entry: Optional[float] = None,
    macro_regime_entry: Optional[str] = None,
) -> PositionState:
    notional = override_notional or pair_cfg.base_size_usdt
    notional *= max(0.1, risk_mult)
    price_ref = max(current_price, 1e-6)
    qty = max(notional / price_ref, 0.0)
    order = api.open_position(symbol, direction, qty)
    entry_price = price_ref
    filled_qty = qty
    if order:
        entry_price = float(order.get("average") or order.get("price") or price_ref)
        filled_qty = abs(float(order.get("filled") or order.get("amount") or qty))

    tp_price = entry_price * (1 + tp_pct * direction)
    sl_price = entry_price * (1 - sl_pct * direction)
    api.place_tp_sl(symbol, direction, entry_price, tp_pct, sl_pct)

    position = build_position_state_from_fill(
        symbol=symbol,
        direction=direction,
        fill_price=entry_price,
        filled_qty=filled_qty,
        tp_price=tp_price,
        sl_price=sl_price,
        now_ts=time.time(),
        open_bar_index=bar_index,
        atr_pct_entry=atr_pct,
        regime_entry=getattr(volatility_regime, "name", "UNKNOWN"),
        entry_score=entry_score,
        score_macro=score_macro,
        score_micro=score_micro,
        macro_regime_entry=macro_regime_entry,
    )
    position.trf_state_entry = trf_state
    position.atr_pct_5m_entry = atr_pct_5m_entry
    position.risk_mult = risk_mult
    position.trailing_sl_price = sl_price or entry_price
    return position


def maybe_scale_in(api, symbol: str, pair_cfg, position: PositionState, volatility_regime, current_price: float, reentry_dist: float) -> None:
    if reentry_dist <= 0 or pair_cfg.scale_size_usdt <= 0:
        return
    move = abs(current_price - position.entry_price) / max(position.entry_price, 1e-9)
    if move < reentry_dist:
        return

    notional = pair_cfg.scale_size_usdt
    qty = notional / max(current_price, 1e-9)
    if qty <= 0:
        return

    order = api.open_position(symbol, position.direction, qty)
    filled_qty = qty
    avg_price = current_price
    if order:
        filled_qty = abs(float(order.get("filled") or order.get("amount") or qty))
        avg_price = float(order.get("average") or order.get("price") or current_price)

    total_qty = position.qty + filled_qty
    if total_qty <= 0:
        return
    position.entry_price = (position.entry_price * position.qty + avg_price * filled_qty) / total_qty
    position.qty = total_qty
    position.trailing_sl_price = min(position.trailing_sl_price, position.entry_price) if position.direction < 0 else max(position.trailing_sl_price, position.entry_price)
    _logger.info("[%s] Scale-in executed qty=%.4f total=%.4f avg_entry=%.6f", symbol, filled_qty, total_qty, position.entry_price)


def append_trade_log(trade: TradeLogEntry) -> None:
    path = Path(TRADE_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADE_LOG_FIELDS)
        writer.writerow(asdict(trade))


def update_loss_streak_and_daily_pnl(state: SymbolState, net_pnl: float, pair_cfg, current_bar_index: int, current_price: float) -> None:
    today = datetime.utcnow().date()
    if state.last_trade_day != today:
        state.last_trade_day = today
        state.daily_pnl = 0.0
        state.loss_streak = 0

    state.daily_pnl += net_pnl
    if net_pnl < 0:
        state.loss_streak += 1
    else:
        state.loss_streak = 0

    if state.loss_streak >= pair_cfg.max_loss_streak:
        _logger.warning("[%s] Max loss streak reached (%d). Cooling down entries.", pair_cfg.symbol, state.loss_streak)
        state.pause_until_bar = max(state.pause_until_bar, current_bar_index + pair_cfg.min_bars_after_flip)


def close_position_safely(api, symbol: str, position: PositionState, reason: str = "manual_exit") -> None:
    try:
        api.close_position(symbol, position.direction, position.qty, reason=reason)
        if hasattr(api, "cancel_all_orders"):
            api.cancel_all_orders(symbol)
    except Exception as exc:
        _logger.error("[%s] Failed to close position safely: %s", symbol, exc)


def update_position_telemetry_from_market(*, pos: PositionState, last_price: float, micro_score: float, ema_slope_deg_1m: float) -> None:
    pos.last_price = float(last_price)
    pos.last_micro_score = float(micro_score or 0.0)
    pos.last_ema_slope_deg = float(ema_slope_deg_1m or 0.0)


def should_trend_exit_slowing(
    *,
    side: int,
    bars_in_trade: int,
    entry_price: float,
    last_price: float,
    trf_state_5m: Optional[str],
    trf_dir_5m: Optional[int],
    slope_5m: Optional[float],
) -> bool:
    if bars_in_trade < 3:
        return False
    progress = (last_price - entry_price) / max(entry_price, 1e-9) * (1 if side > 0 else -1)
    if progress <= -0.002:
        return True
    if trf_state_5m:
        opposite = trf_state_5m.lower().startswith("bear") and side > 0 or trf_state_5m.lower().startswith("bull") and side < 0
        if opposite and abs(progress) < 0.003:
            return True
    if trf_dir_5m and trf_dir_5m != side and (abs(slope_5m or 0.0) > 0):
        return True
    return False


def ensure_stop_loss_order(api, symbol: str, pos: PositionState, strategy_settings) -> None:
    if not pos.sl_price or pos.qty <= 0:
        return
    refresh_interval = max(30, getattr(strategy_settings, "sl_refresh_seconds", 60))
    now = time.time()
    if now - pos.last_sl_refresh_ts < refresh_interval:
        return
    direction = "LONG" if pos.direction > 0 else "SHORT"
    close_side = "sell" if pos.direction > 0 else "buy"
    try:
        api.place_stop_loss(
            symbol=symbol,
            direction=direction,
            entry_price=pos.entry_price,
            sl_price=pos.sl_price,
            sl_side=close_side,
            qty=pos.qty,
            max_attempts=getattr(strategy_settings, "sl_max_retries", 3),
        )
        pos.last_sl_refresh_ts = now
    except Exception as exc:
        _logger.debug("[%s] ensure_stop_loss_order failed: %s", symbol, exc)


__all__ = [
    "SymbolState",
    "PositionState",
    "TradeLogEntry",
    "TRADE_LOG_FIELDS",
    "build_position_state_from_fill",
    "open_core_position",
    "maybe_scale_in",
    "append_trade_log",
    "update_loss_streak_and_daily_pnl",
    "ExitReason",
    "close_position_safely",
    "get_cluster_for_symbol",
    "update_position_telemetry_from_market",
    "should_trend_exit_slowing",
    "ensure_stop_loss_order",
]
