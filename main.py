# -*- coding: utf-8 -*-
"""
Punto de entrada principal para el bot de trading PSAR Scalper.

Este script inicializa y ejecuta el bucle de trading principal.
Responsabilidades:
- Cargar la configuración desde variables de entorno.
- Establecer la conexión con el exchange (Binance Futures).
- Inicializar el proveedor de datos (DataFeed) y el estado del bot.
- Ejecutar el bucle infinito que procesa los datos del mercado,
  evalúa la estrategia y gestiona las posiciones.
- Manejar el cierre ordenado de posiciones al detener el bot.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict
from types import SimpleNamespace

import ccxt
import pandas as pd
from dotenv import load_dotenv
from ccxt.base.errors import ExchangeError, NetworkError

# Cargar variables de entorno PRIMERO
load_dotenv()

from logger_setup import get_logger
from psar_scalper.src.config import (
    ACTIVE_PAIRS,
    TRADE_LOG_PATH,
    EngineSettings,
    ScoreSettings,
    StrategySettings,
)
from psar_scalper.src.data import DataFeed
from psar_scalper.src.engine import (
    open_entry_from_candidate,
    plan_entries_with_min_open_symbols,
    run_symbol_step,
)
from psar_scalper.src.engine_helpers import SymbolState, TRADE_LOG_FIELDS
from psar_scalper.src.indicators import atr, compute_psar
from pathlib import Path

_logger = get_logger("main", level=os.getenv("LOG_LEVEL", "INFO"))

MIN_STOP_DIST_FRAC = float(os.getenv("MIN_STOP_DIST_FRAC", "0.0015"))  # 0.15%


@dataclass
class PSARState:
    trend: str


def adjust_stop_price(api: "BinanceFuturesAPI",
                      symbol: str,
                      direction: str,
                      entry_price: float,
                      stop_price: float) -> float:
    """
    Ajusta el precio del SL para que no quede pegado al precio de entrada
    y evitar el error -2021 'Order would immediately trigger'.

    direction: 'LONG' o 'SHORT'
    """

    direction = direction.upper()

    # Para un SHORT el SL es una orden BUY por encima del entry
    if direction == "SHORT":
        min_sl = entry_price * (1.0 + MIN_STOP_DIST_FRAC)
        if stop_price <= min_sl:
            stop_price = min_sl
    else:
        # Para un LONG el SL es una orden SELL por debajo del entry
        max_sl = entry_price * (1.0 - MIN_STOP_DIST_FRAC)
        if stop_price >= max_sl:
            stop_price = max_sl

    # Usa tu función actual de redondeo/tick size
    return api._round_price(symbol, stop_price)


class BinanceFuturesAPI:
    """
    Clase wrapper para interactuar con la API de Binance Futures.
    Abstrae las llamadas a ccxt para la apertura, cierre y gestión de órdenes,
    así como el cálculo de indicadores básicos.
    """
    def __init__(self, exchange: ccxt.BaseExchange, symbols: tuple[str, ...], strategy_settings: StrategySettings):
        self.exchange = exchange
        self.symbols = symbols
        self._strategy_settings = strategy_settings
        self._market_data: Dict[str, Dict[str, pd.DataFrame]] = {symbol: {} for symbol in symbols}
        self._last_qty: Dict[str, float] = {}

    def _round_price(self, symbol: str, price: float) -> float:
        """Rounds price to the market's tick size."""
        if not price or price <= 0:
            return 0.0
        try:
            return float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            return price

    def _get_tick_size(self, symbol: str) -> float | None:
        market = self.exchange.markets.get(symbol, {})
        info = market.get("info", {}) or {}
        filters = info.get("filters") or []
        for f in filters:
            if f.get("filterType") == "PRICE_FILTER":
                tick = f.get("tickSize")
                if tick:
                    try:
                        return float(tick)
                    except (TypeError, ValueError):
                        continue
        precision = (market.get("precision") or {}).get("price")
        if precision is not None:
            try:
                return pow(10, -int(precision))
            except (TypeError, ValueError):
                return None
        return None

    def get_tick_size(self, symbol: str) -> float | None:
        """
        Expose tick size so other components (watchdogs) can reuse the exchange metadata.
        """
        return self._get_tick_size(symbol)

    def update_market_data(self, symbol: str, df_1m: pd.DataFrame, df_30m: pd.DataFrame) -> None:
        self._market_data.setdefault(symbol, {})
        self._market_data[symbol]["df_1m"] = df_1m
        self._market_data[symbol]["df_30m"] = df_30m

    def compute_spread_bps(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            if bid <= 0 or ask <= 0:
                return 0.0
            mid = (bid + ask) / 2.0
            spread = max(0.0, ask - bid)
            return (spread / max(mid, 1e-9)) * 10000.0
        except Exception as exc:
            _logger.debug("[%s] spread fetch failed: %s", symbol, exc)
            return 0.0

    def compute_atr_pct(self, symbol: str) -> float:
        df = self._market_data.get(symbol, {}).get("df_30m")
        if df is None or df.empty:
            return 0.0
        atr_series = atr(df)
        if atr_series.empty:
            return 0.0
        last_atr = float(atr_series.iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_close <= 0:
            return 0.0
        return last_atr / last_close

    def compute_psar_state(self, symbol: str) -> PSARState:
        df = self._market_data.get(symbol, {}).get("df_30m")
        if df is None or len(df) < 5:
            return PSARState(trend="range")
        psar_series = compute_psar(df)
        if psar_series.empty:
            return PSARState(trend="range")
        last_val = float(psar_series.iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_close > last_val:
            trend = "up"
        elif last_close < last_val:
            trend = "down"
        else:
            trend = "range"
        return PSARState(trend=trend)

    def get_strategy_settings(self, symbol: str):
        return self._strategy_settings

    def normalize_qty(self, symbol: str, qty: float) -> float:
        return self._round_qty(symbol, qty)

    def open_position(self, symbol: str, direction: int, size: float):
        if size <= 0:
            _logger.warning("[%s] open_position received size <= 0: %f", symbol, size)
            return None

        normalized_size = self.normalize_qty(symbol, size)
        
        # Validar contra mínimos del mercado
        market = self.exchange.markets.get(symbol, {})
        min_amount = market.get("limits", {}).get("amount", {}).get("min")
        min_notional = market.get("limits", {}).get("cost", {}).get("min", 5.0)  # Default 5 USDT
        
        _logger.debug(
            "[%s] Order prep: direction=%d, size=%.6f, normalized=%.6f, min_amount=%s, min_notional=%.2f",
            symbol, direction, size, normalized_size, min_amount, min_notional
        )
        
        if normalized_size <= 0:
            _logger.warning(
                "[%s] Normalized order size is %f (too small). Original size was %f. Market min: %s",
                symbol,
                normalized_size,
                size,
                min_amount,
            )
            return None
        
        if min_amount and normalized_size < min_amount:
            _logger.warning(
                "[%s] Normalized size %f is below market minimum %f. Cannot place order.",
                symbol,
                normalized_size,
                min_amount,
            )
            return None

        side = "buy" if direction == 1 else "sell"
        
        # Calculate and validate notional value
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker.get('last', 0)
            notional = normalized_size * current_price
            
            if notional < min_notional:
                _logger.warning(
                    "[%s] Order notional %.2f USDT is below minimum %.2f USDT. Cannot place order.",
                    symbol, notional, min_notional
                )
                return None
            
            _logger.info("[%s] Placing %s order: size=%f (normalized from %f), price=%.2f, notional=%.2f USDT", 
                        symbol, side.upper(), normalized_size, size, current_price, notional)
        except Exception as e:
            _logger.warning("[%s] Could not validate notional: %s", symbol, e)
            _logger.info("[%s] Placing %s order: size=%f (normalized from %f)", symbol, side.upper(), normalized_size, size)
        
        order = self.exchange.create_order(symbol, "market", side, normalized_size)
        if not order:
            _logger.error("[%s] Order creation failed - exchange returned None", symbol)
            return None
        
        filled = order.get("filled") or order.get("amount") or normalized_size
        try:
            qty = abs(float(filled))
        except (TypeError, ValueError):
            qty = abs(normalized_size)
        self._last_qty[symbol] = qty
        
        order_id = order.get("id", "N/A")
        _logger.info(
            "[%s] Order FILLED: id=%s, side=%s, qty=%.6f, price=%.2f",
            symbol, order_id, side.upper(), qty, current_price
        )
        return order

    def place_tp_sl(self, symbol: str, direction: int, entry_price: float, tp_pct: float, sl_pct: float):
        qty = self._last_qty.get(symbol)
        if not qty or entry_price <= 0:
            _logger.debug("[%s] Cannot place TP/SL: qty=%s, entry_price=%.6f", symbol, qty, entry_price)
            return {"sl_order": None, "tp_order": None}
        
        close_side = "sell" if direction == 1 else "buy"
        direction_str = "LONG" if direction == 1 else "SHORT"
        sl_order = None
        tp_order = None

        if sl_pct > 0:
            sl_price = entry_price * (1 - sl_pct) if direction == 1 else entry_price * (1 + sl_pct)
            try:
                sl_order = self.place_stop_loss(symbol, direction_str, entry_price, sl_price, close_side, qty)
            except Exception as exc:
                # The error is already logged in place_stop_loss, but we catch here to avoid crashing the loop
                _logger.warning("[%s] place_stop_loss failed from caller: %s", symbol, exc)

        if tp_pct > 0:
            tp_price = entry_price * (1 + tp_pct) if direction == 1 else entry_price * (1 - tp_pct)
            params = {"reduceOnly": True, "workingType": "MARK_PRICE"}
            try:
                tp_order = self.exchange.create_order(
                    symbol,
                    "TAKE_PROFIT_MARKET",
                    close_side,
                    qty,
                    None,
                    {**params, "stopPrice": self._round_price(symbol, tp_price)},
                )
                _logger.info("[%s] TP placed: price=%.6f (%.2f%%), qty=%.6f", 
                           symbol, tp_price, tp_pct * 100, qty)
            except Exception as exc:
                _logger.warning("[%s] failed to place TP: %s", symbol, exc)
        return {"sl_order": sl_order, "tp_order": tp_order}

    def is_order_open(self, symbol: str, order_id: str | int | None) -> bool:
        """
        Verifica si una orden sigue viva consultando fetch_order y, como fallback,
        la lista de órdenes abiertas. Si no se puede determinar, asume que NO está abierta.
        """
        if order_id is None:
            return False

        order_id_str = str(order_id)
        try:
            order = self.exchange.fetch_order(order_id_str, symbol)
            status = (order.get("status") or "").lower()
            if status in ("open", "pending", "new"):
                return True
            if status:
                return False
        except Exception as exc:
            _logger.debug("[%s] fetch_order failed for id=%s: %s", symbol, order_id_str, exc)

        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                if str(order.get("id")) == order_id_str:
                    return True
        except Exception as exc:
            _logger.debug("[%s] fetch_open_orders failed while checking id=%s: %s", symbol, order_id_str, exc)

        return False

    def place_stop_loss(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        sl_side: str,
        qty: float,
        max_attempts: int = 3,
    ):
        """
        Envía el SL con buffers progresivos. Si falla, devuelve None para que el watchdog
        pueda volver a intentarlo sin dejar la posición desprotegida.
        """
        params = {"reduceOnly": True, "workingType": "CONTRACT_PRICE"}
        adj_sl_price = adjust_stop_price(self, symbol, direction, entry_price, sl_price)
        raw_price = adj_sl_price

        settings = self.get_strategy_settings(symbol)
        max_attempts = max(
            1,
            max_attempts or getattr(settings, "sl_max_retries", 3),
        )
        tick_step = max(1, getattr(settings, "sl_retry_tick_step", 1))
        max_ticks = max(tick_step, getattr(settings, "sl_retry_max_ticks", 10))
        tick_size = self._get_tick_size(symbol) or abs(adj_sl_price * MIN_STOP_DIST_FRAC)
        if not tick_size or tick_size <= 0:
            tick_size = max(abs(adj_sl_price) * 1e-6, 1e-6)

        pos_direction = 1 if direction.upper() == "LONG" else -1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            rounded_price = self._round_price(symbol, adj_sl_price)
            try:
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type="STOP_MARKET",
                    side=sl_side,
                    amount=qty,
                    price=None,
                    params={**params, "stopPrice": f"{rounded_price:.6f}"},
                )
                _logger.info(
                    "[%s] SL placed: price=%.6f, qty=%.6f (attempt %d/%d)",
                    symbol,
                    rounded_price,
                    qty,
                    attempt,
                    max_attempts,
                )
                return sl_order

            except ExchangeError as exc:
                last_error = exc
                if "-2021" not in str(exc):
                    _logger.error("[%s] SL placement failed with non-retriable error: %s", symbol, exc)
                    break

                buffer_ticks = min(attempt * tick_step, max_ticks)
                adj = buffer_ticks * tick_size
                if pos_direction > 0:
                    new_price = raw_price - adj
                else:
                    new_price = raw_price + adj

                _logger.warning(
                    "[%s] SL price %.6f would immediately trigger (attempt %d/%d), expanding buffer by %d ticks → new_price=%.6f",
                    symbol,
                    rounded_price,
                    attempt,
                    max_attempts,
                    buffer_ticks,
                    new_price,
                )
                adj_sl_price = new_price
                continue

        _logger.error(
            "[%s] GAVE UP placing SL after %d attempts. Position remains without hard SL. last_error=%s",
            symbol,
            max_attempts,
            last_error,
        )
        return None

    def close_position(self, symbol: str, direction: int, qty: float, reason: str = "unspecified"):
        """
        Cierra una posición específica usando reduceOnly.
        Siempre loguea la razón; no fuerza close_all salvo que se llame explícitamente.
        """
        try:
            normalized = self.normalize_qty(symbol, qty)
        except Exception as exc:
            _logger.warning("[%s] Failed to normalize qty %.8f: %s", symbol, qty, exc)
            normalized = qty

        if normalized <= 0:
            _logger.debug("[%s] close_position skipped; normalized qty=%.8f", symbol, normalized)
            return

        side_amt = normalized if direction == 1 else -normalized
        try:
            self._submit_close(symbol, side_amt, reason=reason)
            return
        except Exception as exc:
            _logger.warning("[%s] close_position failed (reason=%s): %s", symbol, reason, exc)
            # No forzamos close_all aquí para evitar cierres fantasma; que lo maneje el caller si es crítico.

    def close_all(self, symbol: str, reason: str = "manual_close_all"):
        try:
            _logger.info("[%s] Attempting to close all positions... reason=%s", symbol, reason)
            if hasattr(self.exchange, "fetch_positions"):
                positions = self.exchange.fetch_positions([symbol])
                for pos in positions:
                    amt = float(
                        pos.get("contracts")
                        or pos.get("positionAmt")
                        or pos.get("amount")
                        or 0.0
                    )
                    if abs(amt) < 1e-9:
                        continue
                    _logger.info("[%s] Position found: amt=%.6f, closing... reason=%s", symbol, amt, reason)
                    self._submit_close(symbol, amt, reason=reason)
            else:
                self._fallback_close(symbol, reason=reason)
        except Exception as exc:
            _logger.debug("[%s] close positions failed: %s", symbol, exc)

    def _extract_filled_qty(self, order: dict | None, default_qty: float) -> float:
        if not order or not isinstance(order, dict):
            return abs(float(default_qty))
        candidates = [
            order.get("filled"),
            order.get("executedQty"),
            order.get("amount"),
            order.get("origQty"),
        ]
        info = order.get("info") or {}
        candidates.extend(
            [
                info.get("executedQty"),
                info.get("origQty"),
                info.get("origQty") if isinstance(info, dict) else None,
            ]
        )
        for val in candidates:
            if val is None:
                continue
            try:
                qty = abs(float(val))
                if qty > 0:
                    return qty
            except (TypeError, ValueError):
                continue
        return abs(float(default_qty))

    def _submit_close(self, symbol: str, amt: float, reason: str = "unspecified"):
        side = "sell" if amt > 0 else "buy"
        try:
            order = self.exchange.create_order(
                symbol,
                "market",
                side,
                abs(amt),
                None,
                {"reduceOnly": True},
            )
            filled_qty = self._extract_filled_qty(order, abs(amt))
            _logger.info("[%s] Close order executed: side=%s, qty=%.6f, reason=%s", symbol, side.upper(), filled_qty, reason)
            return order
        except Exception as exc:
            _logger.warning("[%s] close order failed (reason=%s): %s", symbol, reason, exc)
            return None

    def _fallback_close(self, symbol: str, reason: str = "unspecified"):
        balance = self.exchange.fetch_balance()
        info = balance.get("info") or {}
        for pos in info.get("positions", []):
            if pos.get("symbol") != symbol:
                continue
            amt = float(pos.get("positionAmt") or 0.0)
            if abs(amt) < 1e-9:
                continue
            self._submit_close(symbol, amt, reason=reason)

    def _round_qty(self, symbol: str, qty: float) -> float:
        if qty <= 0:
            _logger.debug(f"[{symbol}] _round_qty received qty <= 0: {qty}")
            return 0.0
        market = self.exchange.markets.get(symbol, {})
        filters = market.get("info", {}).get("filters", [])

        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        min_qty = amount_limits.get("min")

        precision = market.get("precision", {}) or {}
        step = precision.get("amount") or precision.get("qty")
        if not step:
            step = limits.get("amount", {}).get("step")

        # Fallback a filters de Binance para step/min_qty si precision no lo trae
        if (step is None or step == 0) or min_qty is None:
            for f in filters:
                ftype = f.get("filterType")
                if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                    step = step or f.get("stepSize")
                    min_qty = min_qty or f.get("minQty")
                if ftype == "MIN_NOTIONAL":
                    # Se ignora aquí, sólo interesa para sizing, no para redondeo
                    pass

        _logger.debug(f"[{symbol}] _round_qty: initial_qty={qty}, step={step}, min_qty={min_qty}")

        if not step or step <= 0:
            _logger.debug(f"[{symbol}] _round_qty: no valid step found, returning original qty.")
            if min_qty is not None and qty < min_qty:
                return 0.0
            return float(qty)

        dstep = Decimal(str(step))
        dqty = Decimal(str(qty))
        try:
            units = (dqty // dstep) * dstep
            result = float(units)
            _logger.debug(f"[{symbol}] _round_qty: rounded_qty={result}")

            if min_qty is not None and result > 0 and result < min_qty:
                _logger.warning(f"[{symbol}] Rounded qty {result} is less than min_qty {min_qty}. Original was {qty}. Returning 0.")
                return 0.0

            return result
        except InvalidOperation:
            _logger.warning(f"[{symbol}] _round_qty: InvalidOperation for qty={qty}, step={step}")
            return float(qty)


def build_exchange() -> ccxt.BaseExchange:
    """
    Configura y devuelve una instancia del exchange ccxt.
    - Lee las credenciales de la API y el modo (testnet/mainnet) del entorno.
    - Establece el apalancamiento para los pares activos.
    - Sincroniza la hora con el servidor del exchange.
    """
    alias = {
        "binance_futures": "binanceusdm",
        "binance_future": "binanceusdm",
        "binance": "binanceusdm",
    }
    exchange_name = os.getenv("EXCHANGE", "binanceusdm")
    cls_name = alias.get(exchange_name, exchange_name)
    if not hasattr(ccxt, cls_name):
        raise RuntimeError(f"ccxt broker '{cls_name}' not found")
    
    # Cargar API keys ANTES de crear el exchange
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        _logger.warning("API Key or Secret not found in environment. Trading will fail.")
    else:
        _logger.info("API credentials loaded successfully")
    
    exchange_cls = getattr(ccxt, cls_name)
    params = {
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "recvWindow": 60000,  # 60 seconds window
        },
        "adjustForTimeDifference": True,
        "apiKey": api_key,
        "secret": api_secret,
        "timeout": 30000,  # 30 seconds
    }
    exchange = exchange_cls(params)

    api_mode = os.getenv("BINANCE_API_MODE", "testnet").lower()
    if api_mode == "testnet" and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
        _logger.info("Sandbox/Testnet mode enabled.")

    exchange.load_markets()
    
    # Sync time with Binance servers
    try:
        server_time = exchange.fetch_time()
        local_time = int(time.time() * 1000)
        time_diff = server_time - local_time
        _logger.info(f"Time difference with Binance: {time_diff}ms ({time_diff/1000:.2f}s)")
        if abs(time_diff) > 1000:
            _logger.warning(f"Significant time difference detected. Adjusting...")
            # Force recalculation on next request
            exchange.load_time_difference()
    except Exception as e:
        _logger.warning(f"Could not sync time with exchange: {e}")
    
    # Configurar apalancamiento para cada par
    leverage = int(os.getenv("LEVERAGE", "10"))
    for symbol in ACTIVE_PAIRS:
        try:
            exchange.set_leverage(leverage, symbol)
            _logger.info(f"[{symbol}] Leverage set to {leverage}x")
        except Exception as e:
            _logger.warning(f"[{symbol}] Could not set leverage: {e}")
    
    return exchange


def run():
    """
    Función principal que ejecuta el bucle de trading del bot.
    - Inicializa el exchange, data feed, y el estado para cada símbolo.
    - Entra en un bucle infinito para:
        1. Obtener nuevos datos de mercado para cada par.
        2. Ejecutar la lógica de trading para cada par (run_symbol_step).
        3. Planificar nuevas entradas basadas en los resultados (plan_entries).
        4. Ejecutar las órdenes de entrada para los candidatos seleccionados.
    - Maneja la interrupción por teclado para cerrar todas las posiciones de forma segura.
    """
    if not ACTIVE_PAIRS:
        _logger.error("No active pairs configured.")
        return

    # Limpiar archivo trades.csv al iniciar y crear uno nuevo con headers
    trades_file = Path(TRADE_LOG_PATH)
    trades_file.parent.mkdir(parents=True, exist_ok=True)
    
    if trades_file.exists():
        trades_file.unlink()
        _logger.info("Archivo trades.csv eliminado.")
    
    # Crear nuevo archivo con headers
    import csv
    with trades_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        writer.writeheader()
    
    _logger.info("Nuevo archivo trades.csv creado con historial limpio.")
    
    _logger.info("Starting Binance Scalper Bot")
    _logger.info("Pairs: %s", ", ".join(ACTIVE_PAIRS))
    _logger.info("Leverage: %sx", os.getenv("LEVERAGE", "10"))

    exchange = build_exchange()
    engine_settings = EngineSettings()
    score_settings = ScoreSettings()
    data_feed = DataFeed(exchange, engine_settings, logger=_logger)
    api = BinanceFuturesAPI(exchange, tuple(ACTIVE_PAIRS), StrategySettings())
    states: Dict[str, SymbolState] = {symbol: SymbolState() for symbol in ACTIVE_PAIRS}

    iteration_count = 0
    try:
        # Bucle principal del bot
        while True:
            iteration_count += 1
            snapshots: Dict[str, SimpleNamespace] = {}
            
            # 1. Procesar cada símbolo individualmente
            for symbol in ACTIVE_PAIRS:
                try:
                    # Obtener datos de mercado
                    signal_df = data_feed.signal_bars(symbol)
                    exec_df = data_feed.execution_bars(symbol)
                    macro_df = data_feed.macro_bars(symbol)

                    if signal_df.empty or exec_df.empty:
                        _logger.debug("[%s] missing bars, skipping iteration", symbol)
                        snapshots[symbol] = SimpleNamespace(position=states[symbol].position)
                        continue

                    api.update_market_data(symbol, exec_df, signal_df)
                    bar = exec_df.iloc[-1]

                    # 2. Ejecutar la lógica de trading para el símbolo
                    snapshot = run_symbol_step(
                        api,
                        symbol,
                        bar,
                        exec_df,
                        signal_df,
                        macro_df,
                        states[symbol],
                    )
                    snapshots[symbol] = snapshot or SimpleNamespace(position=states[symbol].position, close=float(bar.close))
                
                except NetworkError as exc:
                    _logger.warning("[%s] NetworkError fetching data: %s. Skipping symbol.", symbol, exc)
                    snapshots[symbol] = SimpleNamespace(position=states[symbol].position)
                    continue
                except Exception as exc:
                    _logger.exception("[%s] step failed: %s", symbol, exc)
                    snapshots[symbol] = SimpleNamespace(position=states[symbol].position)

            # 3. Planificar nuevas entradas con la visión global del mercado
            entry_candidates = plan_entries_with_min_open_symbols(
                symbols=ACTIVE_PAIRS,
                snapshots=snapshots,
                states=states,
                engine_settings=engine_settings,
                score_settings=score_settings,
                logger=_logger,
            )

            # 4. Ejecutar las entradas planificadas
            for candidate in entry_candidates:
                snap = snapshots.get(candidate.symbol) or SimpleNamespace(position=states[candidate.symbol].position)
                open_entry_from_candidate(
                    api=api,
                    symbol=candidate.symbol,
                    candidate=candidate,
                    snapshot=snap,
                    state=states[candidate.symbol],
                    score_settings=score_settings,
                    logger=_logger,
                )

            active_positions = sum(1 for s in states.values() if s.position is not None)
            _logger.info(
                "Iteration %d | Active positions: %d/%d | target_min=%d | max_open=%d",
                iteration_count,
                active_positions,
                len(ACTIVE_PAIRS),
                engine_settings.always_in_market_target_min,
                engine_settings.always_in_market_max_open,
            )

            time.sleep(engine_settings.poll_seconds)
            
    except KeyboardInterrupt:
        _logger.info("Bot stopped by user")
        _logger.info("Closing all open positions...")
        
        # Cerrar todas las posiciones de todos los pares
        for symbol in ACTIVE_PAIRS:
            try:
                if states[symbol].position:
                    _logger.info("[%s] Closing active position...", symbol)
                    api.close_all(symbol)
                else:
                    _logger.debug("[%s] No position to close", symbol)
            except Exception as exc:
                _logger.error("[%s] Error closing positions: %s", symbol, exc)
        
        _logger.info("All positions closed. Bot stopped.")


if __name__ == "__main__":
    run()
