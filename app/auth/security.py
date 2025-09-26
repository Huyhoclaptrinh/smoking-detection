# app/auth/security.py
from __future__ import annotations
from datetime import timedelta, datetime
import hashlib, hmac, os
from fastapi import HTTPException, Request, Response, Depends
from sqlalchemy.orm import Session
from app.core.db import get_session
from .models import User

# ====== password ======
def hash_password(raw: str) -> str:
    salt = os.environ.get("PASS_SALT", "smoke_salt").encode()
    return hashlib.pbkdf2_hmac("sha256", raw.encode(), salt, 100_000).hex()

def verify_password(raw: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_password(raw), hashed)

# ====== cookie session (đơn giản) ======
COOKIE_NAME = "session"
SESSION_TTL = timedelta(days=7)

def _sign(data: str) -> str:
    key = os.environ.get("SESSION_KEY", "very-secret-key").encode()
    return hmac.new(key, data.encode(), hashlib.sha256).hexdigest()

def make_session_value(email: str, exp: datetime | None = None) -> str:
    exp = exp or (datetime.utcnow() + SESSION_TTL)
    payload = f"{email}|{int(exp.timestamp())}"
    sig = _sign(payload)
    return f"{payload}|{sig}"

def parse_session(value: str) -> tuple[str, int] | None:
    try:
        email, exp_ts, sig = value.split("|")
        payload = f"{email}|{exp_ts}"
        if not hmac.compare_digest(_sign(payload), sig):
            return None
        if int(exp_ts) < int(datetime.utcnow().timestamp()):
            return None
        return email, int(exp_ts)
    except Exception:
        return None

def set_session_cookie(resp: Response, email: str) -> None:
    resp.set_cookie(
        key=COOKIE_NAME,
        value=make_session_value(email),
        httponly=True,
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
        path="/",
    )

def clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(COOKIE_NAME, path="/")

# ====== dependencies ======
def get_current_user(request: Request, db: Session = Depends(get_session)) -> User:
    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        raise HTTPException(401, "unauthorized")
    parsed = parse_session(raw)
    if not parsed:
        raise HTTPException(401, "invalid session")
    email, _ = parsed
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(401, "user not found")
    return user

def require_user(user: User = Depends(get_current_user)) -> User:
    return user
