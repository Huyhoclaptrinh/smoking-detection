# app/auth/models.py
from __future__ import annotations
from sqlalchemy import Column, Integer, String, Boolean, DateTime, text
from sqlalchemy.sql import func
from app.core.db import Base

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(191), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name     = Column(String(191))
    role          = Column(String(32), nullable=False, server_default=text("'user'"))
    # ✳️ thêm trường này để map đúng với cột trong DB (tinyint(1))
    is_active     = Column(Boolean, nullable=False, server_default=text("1"))
    created_at    = Column(DateTime, nullable=False, server_default=func.now())
