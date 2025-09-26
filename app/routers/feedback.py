# app/routers/feedback.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from pathlib import Path
from ..services import feedback_store
from .. import config

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackIn(BaseModel):
    video_name: str
    timestamp: float = Field(..., ge=0)
    frame_idx: int | None = None
    pred_type: str | None = None
    pred_obj: str | None = None
    correct_type: str              # "smoke" | "drink" | "none"
    correct_obj: str | None = None # "cigarette" | "bottle" | "cup" | "wine glass" | None
    # optional boxes if available (pixels in original frame)
    box_xyxy: list[int] | None = None    # prefer user-provided
    pred_box_xyxy: list[int] | None = None


def _maybe_enqueue_finetune(request: Request, typ: str):
    """
    Hàm chạy ở nền, tách khỏi vòng đời HTTP.
    TUYỆT ĐỐI không raise để khỏi spam log.
    """
    try:
        mgr = getattr(request.app.state, "ft_manager", None)
        if mgr is None:
            print("[feedback] ft skip: manager_not_ready")
            return

        # chỉ chạy khi đủ ngưỡng & đủ ảnh (đã có check trong manager)
        ok = mgr.run_once(typ)  # không force; để manager tự quyết
        print(f"[feedback] finetune enqueue result typ={typ} ok={ok}")
    except Exception as e:
        print("[feedback] finetune enqueue error:", repr(e))


@router.post("")
def submit_feedback(payload: FeedbackIn):
    # 1) resolve video path (you already serve downloads from housekeeping path)
    video_path = (Path(config.PROJ_ROOT) / "uploads" / payload.video_name)  # adjust to your storage
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {video_path}")

    # 2) store sample
    dst_img = feedback_store.save_feedback_sample(
        video_path=video_path,
        timestamp=payload.timestamp,
        correct_type=payload.correct_type,
        correct_obj=payload.correct_obj,
        pred_box_xyxy=tuple(payload.pred_box_xyxy) if payload.pred_box_xyxy else None,
        user_box_xyxy=tuple(payload.box_xyxy) if payload.box_xyxy else None,
    )

    # 3) (optional) trigger fine-tune if enough new files exist
    # simple heuristic: count new images in feedback days
    new_count = sum(1 for _ in (config.FEEDBACK_RAW_DIR).glob("*/*.jpg"))
    fine_tune_triggered = False
    reason = None
    if new_count >= config.MIN_SAMPLES_TO_TRAIN:
        fine_tune_triggered = True
        reason = f"threshold {config.MIN_SAMPLES_TO_TRAIN}+ samples"
        # fire-and-forget (thread/process) – see worker below
        try:
            from ..workers.fine_tune import launch_finetune_async
            launch_finetune_async()
        except Exception as e:
            fine_tune_triggered = False
            reason = f"launch failed: {e}"

    return {"ok": True, "saved": str(dst_img), "fine_tune_triggered": fine_tune_triggered, "fine_tune_reason": reason}