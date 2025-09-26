# app/repositories/user_repo.py
from __future__ import annotations
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.user import User

def get_by_email(db: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email)
    return db.scalar(stmt)

def create_user(db: Session, email: str, full_name: str, password_hash: str, role: str = "user") -> User:
    u = User(email=email, full_name=full_name, password_hash=password_hash, role=role)
    db.add(u)
    db.flush()   # có id
    return u
