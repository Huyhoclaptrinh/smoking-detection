from fastapi import APIRouter, Query, Depends
from sqlalchemy import text
from app.core.db import engine
from app.auth.security import require_user

router = APIRouter(tags=["videos"], dependencies=[Depends(require_user)])

@router.get("/videos")
def get_videos():
    sql = text("""
        SELECT video_name, MAX(timestamp_sec) AS last_time
        FROM events GROUP BY video_name ORDER BY last_time DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return {"videos": [r["video_name"] for r in rows]}

@router.get("/events")
def get_events(video_name: str = Query(...)):
    sql = text("""
        SELECT frame_idx, timestamp_sec, timestamp_hms, event_type, obj_name, score
        FROM events WHERE video_name=:v ORDER BY frame_idx
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"v": video_name}).mappings().all()
    return rows

@router.get("/segments")
def get_segments(video_name: str = Query(...)):
    sql = text("""
        SELECT event_type, start_ts, end_ts, start_hms, end_hms, peak_score, num_frames
        FROM segments WHERE video_name=:v ORDER BY start_ts
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"v": video_name}).mappings().all()
    return rows
