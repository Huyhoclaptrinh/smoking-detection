# app/core/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# đọc config an toàn
try:
    from app.core.config import get_config
    DB_URL = get_config().get("DB_URL", "sqlite:///./app.db")
except Exception:
    DB_URL = "sqlite:///./app.db"

engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

class Base(DeclarativeBase):
    pass

def create_all():
    # Import models ở đây để chắc chắn đã nạp class trước khi create_all
    from app.models import user, event  # noqa: F401
    Base.metadata.create_all(bind=engine)

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
