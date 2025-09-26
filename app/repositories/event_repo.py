# app/repositories/event_repo.py
from __future__ import annotations
from typing import Iterable, List, Optional
from sqlalchemy import select, delete, func
from sqlalchemy.orm import Session
from app.models.event import Event, Segment

# -------- Events ----------
def bulk_insert_events(db: Session, items: Iterable[dict]) -> int:
    """
    items: iterable của dict {video_name, frame_idx, ts_sec, typ, obj_name, score}
    """
    objs = [Event(**it) for it in items]
    db.add_all(objs)
    # flush để lấy id nếu cần, nhưng thường không cần.
    return len(objs)

def list_events_by_video(db: Session, video_name: str, limit: int = 1000) -> List[Event]:
    stmt = (select(Event)
            .where(Event.video_name == video_name)
            .order_by(Event.ts_sec.asc())
            .limit(limit))
    return list(db.scalars(stmt))

def delete_events_of_video(db: Session, video_name: str) -> int:
    res = db.execute(delete(Event).where(Event.video_name == video_name))
    return res.rowcount or 0

def count_events_rate_per_minute(db: Session, video_name: str) -> float:
    # Ví dụ đơn giản: tổng events / (end_ts - start_ts)/60 (nếu bạn có duration)
    # Ở đây demo: dùng số lượng / 1 để không lỗi; bạn có thể thay bằng duration thật.
    stmt = select(func.count(Event.id)).where(Event.video_name == video_name)
    total = db.scalar(stmt) or 0
    return float(total)  # thay bằng logic thật

# -------- Segments ----------
def insert_segments(db: Session, items: Iterable[dict]) -> int:
    """
    items: iterable của dict {video_name, typ, start_ts, end_ts, peak_score, num_frames}
    """
    objs = [Segment(**it) for it in items]
    db.add_all(objs)
    return len(objs)

def list_segments_by_video(db: Session, video_name: str, limit: int = 200) -> List[Segment]:
    stmt = (select(Segment)
            .where(Segment.video_name == video_name)
            .order_by(Segment.start_ts.asc())
            .limit(limit))
    return list(db.scalars(stmt))

def delete_segments_of_video(db: Session, video_name: str) -> int:
    res = db.execute(delete(Segment).where(Segment.video_name == video_name))
    return res.rowcount or 0
