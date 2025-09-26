# app/core/config.py
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # optional
except Exception:
    yaml = None


class Config(dict):
    """Cho phép truy cập bằng dấu chấm: cfg.DB_URL, cfg.OUTPUT_DIR..."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


# ---- Helpers ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../human-action-detection
DEFAULT_CFG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not yaml:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _abspath_if_relative(p: str | None) -> str | None:
    if not p:
        return p
    try:
        q = Path(p)
        if q.is_absolute():
            return str(q)
        return str((PROJECT_ROOT / q).resolve())
    except Exception:
        return p


def _normalize_paths(cfg: Dict[str, Any]) -> None:
    """Chuyển các đường dẫn tương đối thành tuyệt đối (nếu có)."""
    path_keys = [
        # model/checkpoints
        "GENERAL_MODEL", "SMOKE_MODEL", "SMOKE_WEIGHT_MODEL", "DRINK_MODEL",
        # output/tmp
        "OUTPUT_DIR", "TMP_DIR", "STATIC_DIR",
        # feedback dirs
        "SMOKE_RAW_DIR", "SMOKE_PROC_DIR", "DRINK_RAW_DIR", "DRINK_PROC_DIR",
        # calibrator
        "SMOKE_CALIB_PATH", "DRINK_CALIB_PATH", "CALIB_LOG_DIR",
    ]
    for k in path_keys:
        if k in cfg:
            cfg[k] = _abspath_if_relative(cfg.get(k))


def _finalize_common(cfg: Dict[str, Any]) -> Config:
    """Merge ENV override + defaults cho AUTH/CORS; chuẩn hoá đường dẫn; trả Config."""
    # ---- DB: ưu tiên ENV.DB_URL > YAML.DB_URL > ghép từ mảnh DB_* ----
    db_user = os.getenv("DB_USER", str(cfg.get("DB_USER", "root")))
    db_pass = os.getenv("DB_PASS", str(cfg.get("DB_PASS", "")))
    db_host = os.getenv("DB_HOST", str(cfg.get("DB_HOST", "127.0.0.1")))
    db_port = int(os.getenv("DB_PORT", str(cfg.get("DB_PORT", 3306))))
    db_name = os.getenv("DB_NAME", str(cfg.get("DB_NAME", "behavior_detect")))

    db_url_env = os.getenv("DB_URL")
    if db_url_env:
        db_url = db_url_env
    else:
        db_url_yaml = cfg.get("DB_URL")
        if db_url_yaml:
            db_url = str(db_url_yaml)
        else:
            db_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"

    # ---- Auth / CORS ----
    secret_key = os.getenv("SECRET_KEY", str(cfg.get("SECRET_KEY") or secrets.token_hex(32)))
    tok_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",
                                str(cfg.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 30))))

    cors_env = os.getenv("CORS_ORIGINS")
    cors_raw = cors_env if cors_env is not None else cfg.get(
        "CORS_ORIGINS",
        ["http://localhost", "http://127.0.0.1:8000", "http://localhost:3000", "http://127.0.0.1:3000"]
    )
    if isinstance(cors_raw, str):
        cors_origins: List[str] = [x.strip() for x in cors_raw.split(",") if x.strip()]
    else:
        cors_origins = list(cors_raw) if isinstance(cors_raw, list) else ["*"]

    # ---- Chuẩn hoá đường dẫn ----
    _normalize_paths(cfg)

    # ---- Merge & return ----
    out: Dict[str, Any] = dict(cfg)
    out.update({
        "DB_USER": db_user,
        "DB_PASS": db_pass,
        "DB_HOST": db_host,
        "DB_PORT": db_port,
        "DB_NAME": db_name,
        "DB_URL": db_url,
        "SECRET_KEY": secret_key,
        "ACCESS_TOKEN_EXPIRE_MINUTES": tok_minutes,
        "CORS_ORIGINS": cors_origins,
        # tiện dụng
        "PROJECT_ROOT": str(PROJECT_ROOT),
    })
    return Config(out)


# ---- Public API ----
def get_config() -> Config:
    """
    Thứ tự ưu tiên:
    1) Nếu tồn tại app.config.load_cfg() => dùng kết quả đó (tức là giữ logic build DB_URL của bạn).
    2) Ngược lại: đọc YAML (CFG_PATH hoặc ./config.yaml) rồi áp ENV override.
    Tất cả được chuẩn hoá đường dẫn và trả về Config truy cập kiểu chấm.
    """
    # 1) Ưu tiên loader tuỳ biến nếu có
    try:
        from app.config import load_cfg  # type: ignore
        cfg = load_cfg(os.getenv("CFG_PATH") or str(DEFAULT_CFG_PATH))
        if not isinstance(cfg, dict):
            cfg = dict(cfg)
        return _finalize_common(cfg)
    except Exception:
        # Không có app.config hoặc lỗi → fallback YAML
        pass

    # 2) Fallback đọc YAML trực tiếp
    cfg_path = Path(os.getenv("CFG_PATH") or DEFAULT_CFG_PATH)
    data = _load_yaml(cfg_path)
    return _finalize_common(data)
