# app/auth/service.py
from __future__ import annotations

from sqlalchemy.orm import Session
from app.auth.models import User
from app.auth.security import hash_password, verify_password

def get_user_by_email(db: Session, email: str) -> User | None:
    if not email:
        return None
    return db.query(User).filter(User.email == email.lower().strip()).first()

# --- giữ nguyên các hàm bạn đang có; nếu chưa có, dùng mẫu dưới ---

def register_user(db: Session, email: str, full_name: str, raw_password: str) -> User:
    email_norm = (email or "").lower().strip()
    if not email_norm or not raw_password:
        raise ValueError("Email và mật khẩu là bắt buộc")
    if get_user_by_email(db, email_norm):
        raise ValueError("Email đã tồn tại")

    user = User(
        email=email_norm,
        full_name=full_name or None,
        password_hash=hash_password(raw_password),
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate(db: Session, email: str, raw_password: str) -> User | None:
    user = get_user_by_email(db, (email or "").lower().strip())
    if not user:
        return None
    if not verify_password(raw_password, user.password_hash):
        return None
    # ✳️ phòng khi model/DB thiếu field, dùng getattr cho chắc
    if not getattr(user, "is_active", True):
        return None
    return user
