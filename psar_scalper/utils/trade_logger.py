# psar_scalper/utils/trade_logger.py

import csv
from pathlib import Path
from datetime import datetime
from typing import Optional


class TradeLogger:
    """
    Logger sencillo para escribir trades en CSV.
    Soporta:
      - log_open: cuando se abre una posición
      - log_close: cuando se cierra (TP / SL / manual)
    Puedes decidir si registras aperturas, cierres o ambos.
    """

    def __init__(
        self,
        csv_path: str = "logs/trades.csv",
        log_opens: bool = True,
        log_closes: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.log_opens = log_opens
        self.log_closes = log_closes

        # Asegura que exista el archivo con encabezados
        self._ensure_header()

    @property
    def fieldnames(self):
        return [
            "ts",
            "event",           # open | close
            "symbol",
            "side",            # LONG | SHORT
            "position_id",     # opcional, puede ir vacío
            "engine_mode",     # ALWAYS_IN_MARKET, etc.
            "cluster",         # LARGE_CAP, MEME, etc.

            "qty",
            "entry_price",
            "exit_price",
            "sl",
            "tp",

            "score",
            "macro",
            "micro",
            "regime",          # UP / DOWN / FLAT
            "rrr",             # risk-reward ratio

            "atr_pct",
            "atr5m_pct",

            "pnl",
            "pnl_pct",
            "reason",          # TP / SL / MANUAL / ...
        ]

    def _ensure_header(self) -> None:
        """
        Crea el CSV con encabezado si no existe o si está vacío.
        No borra datos existentes si el archivo ya tiene contenido.
        """
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def _write_row(self, row: dict) -> None:
        """
        Escribe una fila en el CSV, rellenando claves faltantes con "".
        """
        full_row = {k: row.get(k, "") for k in self.fieldnames}
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(full_row)

    def log_open(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        sl: float,
        tp: float,
        score: float = 0.0,
        macro: float = 0.0,
        micro: float = 0.0,
        regime: str = "UNK",
        rrr: float = 0.0,
        atr_pct: float = 0.0,
        atr5m_pct: float = 0.0,
        engine_mode: str = "ALWAYS_IN_MARKET",
        cluster: str = "",
        position_id: Optional[str] = None,
        reason: str = "ENTRY",
        ts: Optional[datetime] = None,
    ) -> None:
        """
        Registra la APERTURA de una posición.
        """
        if not self.log_opens:
            return

        if ts is None:
            ts = datetime.utcnow()

        row = {
            "ts": ts.isoformat(),
            "event": "open",
            "symbol": symbol,
            "side": side.upper(),
            "position_id": position_id or "",
            "engine_mode": engine_mode,
            "cluster": cluster,

            "qty": qty,
            "entry_price": entry_price,
            "exit_price": "",   # aún no hay
            "sl": sl,
            "tp": tp,

            "score": score,
            "macro": macro,
            "micro": micro,
            "regime": regime,
            "rrr": rrr,

            "atr_pct": atr_pct,
            "atr5m_pct": atr5m_pct,

            "pnl": "",
            "pnl_pct": "",
            "reason": reason,
        }

        self._write_row(row)

    def log_close(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,          # "TP" | "SL" | "MANUAL" | ...
        engine_mode: str = "ALWAYS_IN_MARKET",
        cluster: str = "",
        position_id: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """
        Registra el CIERRE de una posición (TP / SL / manual).
        """
        if not self.log_closes:
            return

        if ts is None:
            ts = datetime.utcnow()

        row = {
            "ts": ts.isoformat(),
            "event": "close",
            "symbol": symbol,
            "side": side.upper(),
            "position_id": position_id or "",
            "engine_mode": engine_mode,
            "cluster": cluster,

            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "sl": "",
            "tp": "",

            "score": "",
            "macro": "",
            "micro": "",
            "regime": "",
            "rrr": "",

            "atr_pct": "",
            "atr5m_pct": "",

            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
        }

        self._write_row(row)
