from typing import List, Dict, Optional
from pydantic import BaseModel

class PredictResponse(BaseModel):
    events: List[Dict]
    overlay_video: Optional[str] = None
    transcript_en: Optional[str] = None
    transcript_vi: Optional[str] = None
    csv: Optional[str] = None
    segments: Optional[List[Dict]] = None
    segments_csv: Optional[str] = None
    video_name: Optional[str] = None

class FeedbackItem(BaseModel):
    video_name: str
    frame_idx: int
    orig_type: str
    correct_type: Optional[str] = None
    correct_obj_name: Optional[str] = None
    bbox: Optional[List[float]] = None
    error_type: str
