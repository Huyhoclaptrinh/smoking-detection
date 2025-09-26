# app/repositories/feedback_repo.py
from __future__ import annotations
from typing import Iterable, List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.event import Feedback

def insert_feedback(db: Session, **kwargs) -> Feedback:
    """
    kwargs: video_name, frame_idx, orig_type, correct_type, bbox_xyxy, img_path, label_path
    """
    fb = Feedback(**kwargs)
    db.add(fb)
    db.flush()
    return fb

def list_feedback_by_video(db: Session, video_name: str) -> List[Feedback]:
    stmt = select(Feedback).where(Feedback.video_name == video_name).order_by(Feedback.frame_idx.asc())
    return list(db.scalars(stmt))
