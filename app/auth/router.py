# app/auth/router.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, Request
from sqlalchemy.orm import Session
from typing import Optional

from app.core.db import get_session
from app.auth.service import register_user, authenticate, get_user_by_email  # adjust if needed
from app.auth.security import set_session_cookie, clear_session_cookie, get_current_user  # adjust if needed

router = APIRouter(prefix="/auth", tags=["auth"])
# Alias không prefix để /login, /register vẫn chạy nếu frontend gọi nhầm
router_pub = APIRouter(tags=["auth"])


# -------- Helpers để nhận JSON/Form --------
async def _read_fields(req: Request, *names: str) -> dict:
    """
    Trả về dict {name: value} cho các field yêu cầu, ưu tiên JSON, fallback Form.
    """
    data = {}
    ct = (req.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            j = await req.json()
            if isinstance(j, dict):
                data.update(j)
        except Exception:
            pass
    if not data:
        # fallback: form
        try:
            form = await req.form()
            for k in names:
                if k in form:
                    data[k] = form.get(k)
        except Exception:
            pass
    return {k: (data.get(k) if data.get(k) is not None else "") for k in names}


# ---------------- Register ----------------
@router.post("/register")
async def register(req: Request, db: Session = Depends(get_session)):
    fields = await _read_fields(req, "email", "password", "full_name")
    email = (fields.get("email") or "").strip().lower()
    password = fields.get("password") or ""
    full_name = fields.get("full_name") or ""

    if not email or not password:
        raise HTTPException(status_code=422, detail="email và password là bắt buộc")

    try:
        u = register_user(db, email=email, full_name=full_name, raw_password=password)
        return {"id": u.id, "email": u.email, "full_name": u.full_name}
    except ValueError as e:
        # ví dụ: email đã tồn tại
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- Login ----------------
@router.post("/login")
async def login(req: Request, response: Response, db: Session = Depends(get_session)):
    fields = await _read_fields(req, "email", "password")
    email = (fields.get("email") or "").strip().lower()
    password = fields.get("password") or ""

    if not email or not password:
        raise HTTPException(status_code=422, detail="Thiếu email hoặc password")

    user = authenticate(db, email=email, raw_password=password)
    if not user:
        # Không tiết lộ lý do cụ thể (sai email hay mật khẩu)
        raise HTTPException(status_code=401, detail="Sai thông tin đăng nhập")

    # Thiết lập cookie phiên
    set_session_cookie(response, user.email)
    return {"ok": True, "email": user.email}


# ---------------- Me ----------------
@router.get("/me")
def me(user=Depends(get_current_user)):
    # get_current_user đọc từ cookie (hoặc header) -> trả user object/row
    if not user:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    return {"email": user.email, "full_name": getattr(user, "full_name", None)}


# ---------------- Logout ----------------
@router.post("/logout")
def logout(response: Response):
    clear_session_cookie(response)
    return {"ok": True}


# ---------------- Aliases (không prefix) ----------------
@router_pub.post("/register")
async def register_alias(req: Request, db: Session = Depends(get_session)):
    return await register(req, db)

@router_pub.post("/login")
async def login_alias(req: Request, response: Response, db: Session = Depends(get_session)):
    return await login(req, response, db)

@router_pub.post("/logout")
async def logout_alias(response: Response):
    return logout(response)
