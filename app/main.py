# app/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import get_config
from app.core.db import create_all
from app.detector.config import DetectionConfig
from app.detector.behavior import BehaviorDetector
from app.logging import setup_logging

# Routers
from app.auth.router import router as auth_router, router_pub as auth_router_pub
from app.routers.feedback import router as feedback_router
from app.routers.predict import router as predict_router         # ✅ dùng router /predict của bạn
from app.routers.housekeeping import router as housekeeping_router
from app.workers.finetune import FineTuneManager

app = FastAPI(title="Human Behavior Detection API", version="1.0.0")
setup_logging("INFO")

# ===== Config & DB =====
cfg = get_config()
app.state.cfg = cfg
create_all()

# ===== Static (frontend) =====
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# (tuỳ chọn) root -> index.html cho tiện truy cập
@app.get("/")
def _index_redirect():
    return RedirectResponse(url="/static/index.html", status_code=307)

# ===== Include routers (đã login OK, giữ nguyên) =====
app.include_router(auth_router)             # /auth/*
app.include_router(auth_router_pub)         # /login, /register, /logout (alias)
app.include_router(feedback_router)         # /feedback/*
app.include_router(predict_router)          # /predict/*  <-- HTML đang gọi POST /predict
app.include_router(housekeeping_router)     # /housekeeping/download

# ===== Health fallback =====
@app.get("/_health_fallback")
def _health_fallback():
    try:
        det = getattr(app.state, "detector", None)
        return {"ok": True, "detector_error": None} if det else {"ok": False, "detector_error": "not init"}
    except Exception as e:
        return JSONResponse({"ok": False, "detector_error": repr(e)})

# ===== Debug routes list (tuỳ chọn) =====
@app.get("/_routes")
def _routes():
    return [getattr(r, "path", str(r)) for r in app.router.routes]

# ===== Startup: init detector =====
@app.on_event("startup")
def _startup_detector():
    # NOTE: DetectionConfig.from_app_config phải tồn tại theo project của bạn
    det_cfg = DetectionConfig.from_app_config(app.state.cfg)
    app.state.detector = BehaviorDetector(det_cfg)
    app.state.detector_ready = True

# ===== Startup: fine-tune worker (tuỳ chọn) =====
@app.on_event("startup")
def _startup_workers_once():
    app.state.ft_manager = FineTuneManager(app.state.detector, cfg)
