import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str = "sar_scalper", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Handler para terminal (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Handler para archivo de logs detallado
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(logging.DEBUG)  # Archivo más detallado

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)  # Logger acepta todos los niveles

    return logger
