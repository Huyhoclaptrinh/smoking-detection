# app/models/event.py
from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, func, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

class Event(Base):
    """
    1 event = 1 frame có phát hiện hành vi (smoke/drink/none/unknown...)
    """
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_sec: Mapped[float] = mapped_column(Float, nullable=False)  # timestamp (s)
    typ: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # smoke/drink/unknown...
    obj_name: Mapped[str] = mapped_column(String(64), nullable=True)         # bottle/cup/cigarette...
    score: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

Index("ix_events_video_frame", Event.video_name, Event.frame_idx, unique=False)

class Segment(Base):
    """
    Gộp chuỗi events cùng loại liên tục thành 1 segment (để xem nhanh).
    """
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    typ: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    peak_score: Mapped[float] = mapped_column(Float, nullable=True)
    num_frames: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

Index("ix_segments_video_typ", Segment.video_name, Segment.typ, unique=False)

class Feedback(Base):
    """
    Lưu góp ý người dùng để finetune (ảnh/label YOLO lưu ở file; DB giữ metadata).
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    orig_type: Mapped[str] = mapped_column(String(32), nullable=False)
    correct_type: Mapped[str] = mapped_column(String(32), nullable=False)
    bbox_xyxy: Mapped[str] = mapped_column(String(128), nullable=True)  # "x1,y1,x2,y2"
    img_path: Mapped[str] = mapped_column(String(512), nullable=True)   # ảnh crop hoặc gốc
    label_path: Mapped[str] = mapped_column(String(512), nullable=True) # file .txt yolo
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
