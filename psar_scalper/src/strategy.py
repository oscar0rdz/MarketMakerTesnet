from __future__ import annotations

from typing import Dict, Optional
 
import numpy as np
import pandas as pd
import logging

from .config import PairConfig, RiskSettings, StrategySettings
from .indicators import atr_ema, bollinger_bands, compute_psar, ema, rsi
from .state import Position, Signal, SignalSide
from .regime import RegimeState
from .risk_utils import RegimeConfig, compute_tp_sl
from .score_utils import compute_signal_score


class SARScalperStrategy:
    def __init__(self, settings: StrategySettings, risk: RiskSettings):
        self.settings = settings
        self.risk = risk
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        df: pd.DataFrame,
        regime_state: RegimeState,
        volatility_regime: RegimeConfig,
        pair_cfg: PairConfig,
        spread_bps: float,
        position: Optional[Position] = None,
        micro_df: Optional[pd.DataFrame] = None,
    ) -> Signal:
        min_history = max(self.settings.min_history, self.settings.ema_period + 5, self.settings.atr_period + 5)
        if len(df) < min_history:
            return Signal(reason="insufficient_history")

        work = self._build_signal_frame(df)
        if len(work) < 5:
            return Signal(reason="insufficient_history")

        last = work.iloc[-1]
        timestamp = work.index[-1]

        close = float(last["close"])
        ema_now = float(last["ema"])
        atr_now = float(last["atr"])
        psar_now = float(last["psar"])
        atr_pct_val = float(last["atr_pct"])
        long_flip = bool(last["long_flip"])
        short_flip = bool(last["short_flip"])
        direction = int(last["direction"])
        ema_fast = float(last.get("ema_fast", ema_now))
        ema_slope = float(last.get("ema_slope", 0.0))
        rsi_now = float(last.get("rsi", 50.0))
        bb_upper = float(last.get("bb_upper", close))
        bb_lower = float(last.get("bb_lower", close))
        bb_mid = float(last.get("bb_mid", close))
        micro_momentum = self._micro_momentum(micro_df)

        metadata = self._build_metadata(
            psar_now,
            ema_now,
            ema_fast,
            ema_slope,
            atr_pct_val,
            long_flip,
            short_flip,
            rsi_now,
            bb_upper,
            bb_lower,
            bb_mid,
            micro_momentum,
            regime_state,
            volatility_regime,
        )

        in_trend = regime_state.trend in ("up", "down")
        in_range = regime_state.trend == "range"

        risk_context = {
            "volatility_regime": volatility_regime,
            "pair_cfg": pair_cfg,
            "spread_bps": spread_bps,
        }

        if position is None:
            if in_trend:
                if regime_state.trend == "up" and direction > 0 and close > ema_fast:
                    tag = "sar_long_flip" if long_flip else "regime_trend_follow_long"
                    signal = self._build_signal("long", close, atr_now, timestamp, direction, metadata, risk_context, tag=tag, is_trend=True)
                    self.logger.debug(f"📶 Señal LONG (TENDENCIA) generada: {tag}, precio={close:.6f}, ATR%={atr_pct_val:.4f}")
                    return signal

                if regime_state.trend == "down" and direction < 0 and close < ema_fast:
                    tag = "sar_short_flip" if short_flip else "regime_trend_follow_short"
                    signal = self._build_signal("short", close, atr_now, timestamp, direction, metadata, risk_context, tag=tag, is_trend=True)
                    self.logger.debug(f"📶 Señal SHORT (TENDENCIA) generada: {tag}, precio={close:.6f}, ATR%={atr_pct_val:.4f}")
                    return signal

            if in_range:
                range_signal = self._range_entry(
                    close,
                    atr_now,
                    timestamp,
                    direction,
                    metadata,
                    risk_context,
                    rsi_now,
                    bb_upper,
                    bb_lower,
                    micro_momentum,
                )
                if range_signal:
                    self.logger.debug(f"📶 Señal {range_signal.side.upper()} (RANGO) generada: {range_signal.reason}, precio={close:.6f}")
                    return range_signal

        if position and ((position.side == "long" and short_flip) or (position.side == "short" and long_flip)):
            return Signal(
                side="flat",
                exit=True,
                reason="sar_flip_exit",
                entry=close,
                metadata=metadata,
                direction=direction,
                timestamp=timestamp,
            )

        if position and self._micro_exit_required(position.side, micro_momentum):
            return Signal(
                side="flat",
                exit=True,
                reason="micro_momentum_exit",
                entry=close,
                metadata=metadata,
                direction=direction,
                timestamp=timestamp,
            )

        return Signal(
            reason="hold",
            metadata=metadata,
            direction=direction,
            timestamp=timestamp,
            atr=atr_now,
        )

    def _build_signal(
        self,
        side: SignalSide,
        price: float,
        atr_now: float,
        timestamp,
        direction: int,
        metadata: Dict[str, object],
        risk_context: Dict[str, object],
        tag: str,
        is_trend: bool = False,
    ) -> Signal:
        volatility_regime: RegimeConfig = risk_context["volatility_regime"]
        pair_cfg: PairConfig = risk_context["pair_cfg"]
        spread_bps: float = risk_context["spread_bps"]
        atr_pct_val = float(metadata.get("atr_pct", 0.0) or 0.0)

        # New score calculation
        score = compute_signal_score(
            direction=side.upper(),
            regime=str(metadata.get("macro_trend", "micro")).upper(),
            atr_pct=atr_pct_val,
            mb_dir=np.sign(metadata.get("micro_momentum", 0.0)) if np.isfinite(metadata.get("micro_momentum", 0.0)) else 0,
            trf_dir=direction,
            bw=metadata.get("bb_bw"),
            slope=metadata.get("ema_slope"),
        )
        priority = abs(score)

        tp_pct, sl_pct = compute_tp_sl(
            atr_pct=atr_pct_val,
            regimen=volatility_regime,
            pair_cfg=pair_cfg,
            spread_bps=spread_bps,
        )

        if side == "long":
            sl = price * (1 - sl_pct)
            tp = None if is_trend else price * (1 + tp_pct)
        else:  # short
            sl = price * (1 + sl_pct)
            tp = None if is_trend else price * (1 - tp_pct)

        confidence = priority 
        meta = dict(metadata)
        meta["setup"] = tag
        meta["tp_pct"] = tp_pct
        meta["sl_pct"] = sl_pct

        return Signal(
            side=side,
            entry=price,
            sl=sl,
            tp=tp,
            score=score,
            priority=priority,
            confidence=confidence,
            reason=tag,
            metadata=meta,
            direction=direction,
            timestamp=timestamp,
            atr=atr_now,
        )

    def _atr_floor_threshold(self) -> float:
        scale = self.settings.atr_floor_pct_scale
        if scale <= 0:
            scale = 1.0
        return self.settings.atr_floor_pct * scale

    def _build_signal_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "timestamp" in work.columns:
            work = work.set_index("timestamp")
        work["psar"] = compute_psar(work, self.settings.sar_step, self.settings.sar_max_step)
        work["ema"] = ema(work["close"], self.settings.ema_period)
        work["ema_fast"] = ema(work["close"], self.settings.ema_fast_period)
        work["atr"] = atr_ema(work, self.settings.atr_period, self.settings.atr_ema_period)
        work = work.dropna(subset=["psar", "ema", "atr"])
        if work.empty:
            return work
        work["direction"] = np.where(work["psar"] < work["close"], 1, -1)
        work["prev_direction"] = work["direction"].shift(1)
        work["signal_delta"] = work["direction"] - work["prev_direction"]
        work["long_flip"] = work["signal_delta"] == 2
        work["short_flip"] = work["signal_delta"] == -2
        atr_pct = work["atr"] / work["close"].replace(0, np.nan)
        work["atr_pct"] = atr_pct.fillna(0.0)
        lookback = max(1, self.settings.ema_slope_lookback)
        ema_prev = work["ema"].shift(lookback)
        work["ema_slope"] = ((work["ema"] - ema_prev) / ema_prev.replace(0.0, np.nan)).fillna(0.0)
        work["rsi"] = rsi(work["close"], self.settings.rsi_period)
        bands = bollinger_bands(work["close"], self.settings.bollinger_period, self.settings.bollinger_stddev)
        for col in ("bb_mid", "bb_upper", "bb_lower"):
            work[col] = bands[col]
        return work

    @staticmethod
    def _build_metadata(
        psar_now: float,
        ema_now: float,
        ema_fast: float,
        ema_slope: float,
        atr_pct: float,
        long_flip: bool,
        short_flip: bool,
        rsi_now: float,
        bb_upper: float,
        bb_lower: float,
        bb_mid: float,
        micro_momentum: float,
        regime_state: RegimeState,
        volatility_regime: RegimeConfig,
    ) -> Dict[str, object]:
        bb_bw = ((bb_upper - bb_lower) / bb_mid) if bb_mid > 0 else 0.0
        return {
            "psar": psar_now,
            "ema": ema_now,
            "ema_fast": ema_fast,
            "ema_slope": ema_slope,
            "atr_pct": atr_pct,
            "long_flip": long_flip,
            "short_flip": short_flip,
            "rsi": rsi_now,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_mid": bb_mid,
            "bb_bw": bb_bw,
            "micro_momentum": micro_momentum,
            "macro_trend": regime_state.trend,
            "macro_bias": regime_state.bias,
            "volatility_regime": volatility_regime.name,
        }

    def _trend_ok(self, side: SignalSide, ema_slope: float, ema_fast: float, ema_slow: float) -> bool:
        slope_min = self.settings.ema_slope_min
        if side == "long":
            return ema_slope >= slope_min and ema_fast >= ema_slow
        return ema_slope <= -slope_min and ema_fast <= ema_slow

    def _range_entry(
        self,
        close: float,
        atr_now: float,
        timestamp,
        direction: int,
        metadata: Dict[str, object],
        risk_context: Dict[str, object],
        rsi_now: float,
        bb_upper: float,
        bb_lower: float,
        micro_momentum: float,
    ) -> Optional[Signal]:
        tolerance = self.settings.range_band_tolerance
        valid_lower = np.isfinite(bb_lower)
        valid_upper = np.isfinite(bb_upper)
        lower_trigger = valid_lower and close <= bb_lower * (1 + tolerance)
        upper_trigger = valid_upper and close >= bb_upper * (1 - tolerance)
        oversold = rsi_now <= self.settings.rsi_oversold
        overbought = rsi_now >= self.settings.rsi_overbought

        if lower_trigger and oversold and self._micro_confirms("long", micro_momentum):
            meta = dict(metadata)
            meta["range_trigger"] = "lower_band"
            return self._build_signal(
                "long", close, atr_now, timestamp, direction, meta, risk_context, tag="regime_range_long", is_trend=False
            )
        if upper_trigger and overbought and self._micro_confirms("short", micro_momentum):
            meta = dict(metadata)
            meta["range_trigger"] = "upper_band"
            return self._build_signal(
                "short", close, atr_now, timestamp, direction, meta, risk_context, tag="regime_range_short", is_trend=False
            )
        return None

    def _micro_momentum(self, micro_df: Optional[pd.DataFrame]) -> float:
        if micro_df is None or micro_df.empty:
            return float("nan")
        closes = micro_df["close"].astype(float)
        window = min(len(closes) - 1, self.settings.micro_momentum_window)
        if window < 1:
            return float("nan")
        recent = closes.iloc[-1]
        past = closes.iloc[-(window + 1)]
        if past == 0.0:
            return float("nan")
        return (recent - past) / past

    def _micro_confirms(self, side: SignalSide, micro_momentum: float) -> bool:
        if not np.isfinite(micro_momentum):
            return True
        threshold = self.settings.micro_momentum_min_pct
        if threshold <= 0:
            return True
        if side == "long":
            return micro_momentum >= threshold
        if side == "short":
            return micro_momentum <= -threshold
        return True

    def _micro_exit_required(self, side: SignalSide, micro_momentum: float) -> bool:
        if not np.isfinite(micro_momentum):
            return False
        threshold = self.settings.micro_momentum_min_pct
        if threshold <= 0:
            return False
        if side == "long" and micro_momentum <= -threshold:
            return True
        if side == "short" and micro_momentum >= threshold:
            return True
        return False
