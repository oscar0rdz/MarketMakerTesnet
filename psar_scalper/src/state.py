from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime

SignalSide = Literal["long", "short", "flat"]


@dataclass
class Signal:
    """
    Generic signal container used by the strategy and the engine. All fields are
    optional so simple health-check signals can still be created.
    """

    side: Optional[SignalSide] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    score: Optional[float] = None
    priority: Optional[float] = None
    confidence: Optional[float] = None
    reason: str = ""
    metadata: Optional[dict] = field(default_factory=dict)
    direction: Optional[int] = None
    timestamp: Optional[datetime] = None
    atr: Optional[float] = None
    exit: bool = False


@dataclass
class Position:
    """
    Lightweight representation of a strategy position. The execution engine
    uses a richer PositionState, but the public API exposes this simplified
    view to keep backwards compatibility with older notebooks/tests.
    """

    side: SignalSide
    entry_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    size: float = 0.0
    opened_at: Optional[datetime] = None


@dataclass
class TrendState:
    trend: str = "range"
    slope: float = 0.0
    strength: float = 0.0


@dataclass
class RegimeState:
    trend: str = "range"
    bias: int = 0
    twin_upper: Optional[float] = None
    twin_lower: Optional[float] = None
    twin_mid: Optional[float] = None
    twin_slope: Optional[float] = None
    in_band: bool = False


@dataclass
class SymbolMeta:
    symbol: str
    leverage: Optional[int] = None
    cluster: Optional[str] = None


@dataclass
class ScalpLayer:
    symbol: str
    state: str
    last_update: float


def to_order_side(side: SignalSide | int | str) -> str:
    """
    Normalise the textual representation of a side (long/short/flat).
    """
    if isinstance(side, str):
        side_l = side.lower()
        if side_l in {"long", "buy"}:
            return "LONG"
        if side_l in {"short", "sell"}:
            return "SHORT"
        return "FLAT"
    if isinstance(side, int):
        if side > 0:
            return "LONG"
        if side < 0:
            return "SHORT"
    return "FLAT"


__all__ = [
    "Signal",
    "SignalSide",
    "Position",
    "TrendState",
    "RegimeState",
    "SymbolMeta",
    "ScalpLayer",
    "to_order_side",
]
