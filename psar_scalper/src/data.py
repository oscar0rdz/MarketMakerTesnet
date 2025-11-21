
from __future__ import annotations

import time
from typing import Dict, Tuple, TYPE_CHECKING

import pandas as pd
from ccxt.base.errors import NetworkError, ExchangeNotAvailable

if TYPE_CHECKING:
    import ccxt
    from logging import Logger
    from psar_scalper.config import EngineSettings


class DataFeed:
    """
    Handles fetching of OHLCV data from the exchange.
    """

    def __init__(
        self,
        exchange: "ccxt.Exchange",
        engine_settings: "EngineSettings",
        logger: "Logger",
    ):
        self.exchange = exchange
        self.engine_settings = engine_settings
        self.logger = logger
        self._cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
        self.cache_ttl_seconds = engine_settings.poll_seconds / 2.0

    def _fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """
        Fetches OHLCV data from the exchange and caches it.
        """
        cache_key = (symbol, timeframe)
        current_time = time.time()

        if cache_key in self._cache:
            df, last_fetch_time = self._cache[cache_key]
            if current_time - last_fetch_time < self.cache_ttl_seconds:
                self.logger.debug(
                    "[%s] Returning cached %s data (age=%.2fs)",
                    symbol,
                    timeframe,
                    current_time - last_fetch_time,
                )
                return df

        self.logger.debug(
            "[%s] Fetching fresh %s data (limit=%d)", symbol, timeframe, limit
        )
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                self.logger.warning(
                    "[%s] fetch_ohlcv for %s returned no data.", symbol, timeframe
                )
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            
            # Binance returns from oldest to newest, no need to sort.
            # We keep the original timestamp as a column for compatibility with other parts
            # that might expect it.
            
            self._cache[cache_key] = (df, current_time)
            return df

        except (NetworkError, ExchangeNotAvailable) as e:
            self.logger.warning(
                "[%s] Network error fetching %s data: %s", symbol, timeframe, e
            )
            # Return stale data if available
            if cache_key in self._cache:
                self.logger.warning("[%s] Returning stale %s data due to network error.", symbol, timeframe)
                return self._cache[cache_key][0]
            raise  # Re-raise if no cache is available
        except Exception as e:
            self.logger.exception(
                "[%s] Unhandled error fetching %s data: %s", symbol, timeframe, e
            )
            # Return stale data if available
            if cache_key in self._cache:
                self.logger.warning("[%s] Returning stale %s data due to unhandled error.", symbol, timeframe)
                return self._cache[cache_key][0]
            return pd.DataFrame()

    def signal_bars(self, symbol: str) -> pd.DataFrame:
        """
        Fetches the bars for the signal timeframe (e.g., 30m).
        """
        return self._fetch_ohlcv(
            symbol,
            self.engine_settings.signal_timeframe,
            self.engine_settings.history_limit,
        )

    def execution_bars(self, symbol: str) -> pd.DataFrame:
        """
        Fetches the bars for the execution timeframe (e.g., 1m).
        """
        return self._fetch_ohlcv(
            symbol,
            self.engine_settings.execution_timeframe,
            self.engine_settings.monitor_history_limit,
        )

    def macro_bars(self, symbol: str) -> pd.DataFrame:
        """
        Fetches the bars for the macro timeframe (e.g., 4h).
        """
        return self._fetch_ohlcv(
            symbol,
            self.engine_settings.macro_timeframe,
            self.engine_settings.history_limit,
        )
