# app/logging.py
from __future__ import annotations
import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_INITIALIZED = False

def setup_logging(level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """Gắn handler cho console + file, chỉ chạy 1 lần."""
    global _LOG_INITIALIZED
    if _LOG_INITIALIZED:
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _LOG_INITIALIZED = True

def get_logger(name: str | None = None) -> logging.Logger:
    """Helper: dùng ở mọi nơi sau khi setup_logging()."""
    return logging.getLogger(name or "app")

__all__ = ["setup_logging", "get_logger"]
