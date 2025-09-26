# app/routers/predict.py
from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, HTTPException, File, Request
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Any, Mapping, Union
from uuid import uuid4
from shutil import copyfileobj

from app.core.db import get_session
from app.repositories.event_repo import bulk_insert_events, insert_segments
from app.detector.behavior import BehaviorDetector
from app.detector.events import events_to_segments  # nếu có

router = APIRouter(prefix="/predict", tags=["predict"])

# === Đường dẫn gốc & thư mục uploads ổn định ===
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helpers về path ----------
def _to_posix_rel(p: Union[str, Path, None]) -> str | None:
    """
    Chuẩn hoá 'p' về đường dẫn TƯƠNG ĐỐI so với BASE_DIR, dạng POSIX (dùng '/').
    - Nếu 'p' đã là tương đối: hợp thức hoá và trả về.
    - Nếu 'p' là tuyệt đối (Windows hoặc POSIX): cố gắng relative_to(BASE_DIR).
    - Nếu relative_to thất bại, cắt chuỗi kể từ 'outputs/' hoặc 'uploads/' hoặc 'static/'.
    - Cuối cùng, nếu vẫn không được, trả về phần tên file (p.name) như phương án cuối.
    """
    if not p:
        return None
    s = str(p).replace("\\", "/")
    try:
        cand = Path(s)
        # Nếu là tương đối (không có anchor), cứ chuẩn hoá về posix
        if not cand.is_absolute():
            # dẹp '..' nếu có
            rel = (BASE_DIR / cand).resolve().relative_to(BASE_DIR)
            return rel.as_posix()

        # Tuyệt đối: thử relative_to(BASE_DIR)
        rel = Path(s).resolve().relative_to(BASE_DIR)
        return rel.as_posix()
    except Exception:
        # Thử cắt phần sau các thư mục quen thuộc
        for key in ("outputs/", "uploads/", "static/"):
            idx = s.lower().rfind(key)
            if idx != -1:
                return s[idx:]
        # Cuối cùng: trả về chỉ tên file
        try:
            return Path(s).name
        except Exception:
            return s


def _save_upload_to_project(f: UploadFile) -> Path:
    """
    Lưu file upload vào <BASE_DIR>/uploads với tên an toàn, giữ đuôi.
    Trả về đường dẫn TUYỆT ĐỐI trong máy chủ.
    """
    suffix = Path(f.filename or "").suffix or ".bin"
    safe_name = f"{uuid4().hex}{suffix}"
    dst = UPLOAD_DIR / safe_name
    with dst.open("wb") as out:
        copyfileobj(f.file, out)
    return dst


# ---------- Chuẩn hoá sự kiện ----------
def _norm_event(e: Any) -> dict:
    if isinstance(e, dict):
        get = e.get
        return {
            "frame_idx": get("frame_idx") or get("frame"),
            "timestamp": get("timestamp") or get("ts_sec") or get("ts"),
            "type":      get("type") or get("typ") or get("label"),
            "obj_name":  get("obj_name") or get("obj") or get("name"),
            "score":     float(get("score") or get("conf") or get("confidence") or 0.0),
        }
    return {
        "frame_idx": getattr(e, "frame_idx", getattr(e, "frame", None)),
        "timestamp": getattr(e, "timestamp", getattr(e, "ts_sec", getattr(e, "ts", None))),
        "type":      getattr(e, "type", getattr(e, "typ", getattr(e, "label", None))),
        "obj_name":  getattr(e, "obj_name", getattr(e, "obj", getattr(e, "name", None))),
        "score":     float(getattr(e, "score", getattr(e, "conf", getattr(e, "confidence", 0.0)))),
    }


# ---------- Chuẩn hoá kết quả detector ----------
def _normalize_result(res: Any, video_path: str) -> dict:
    """
    Chuẩn hoá mọi kiểu trả về của detector về dict có keys:
    events(list), overlay_video(Optional[str]), csv(Optional[str]), video_name(str)
    """
    out = {"events": [], "overlay_video": None, "csv": None, "video_name": Path(video_path).stem}
    # dict
    if isinstance(res, dict):
        out["events"]        = res.get("events") or []
        out["overlay_video"] = res.get("overlay_video")
        out["csv"]           = res.get("csv")
        out["video_name"]    = res.get("video_name") or out["video_name"]
        return out
    # tuple/list: đoán (events, overlay?, csv?)
    if isinstance(res, (list, tuple)):
        if res and isinstance(res[0], (list, tuple)):
            out["events"] = list(res[0])
        if len(res) > 1 and isinstance(res[1], (str, type(None))):
            out["overlay_video"] = res[1]
        if len(res) > 2 and isinstance(res[2], (str, type(None))):
            out["csv"] = res[2]
        return out
    # fallback: nếu detector trả list events trực tiếp
    if isinstance(res, list):
        out["events"] = res
        return out
    return out


def _call_detector(det: BehaviorDetector, video_path: str) -> dict:
    """
    Gọi hàm infer bất kể nó tên gì: process_video / detect_video / run_video / predict / process
    Sau đó chuẩn hoá kết quả.
    """
    last_err: Exception | None = None
    for name in ("process_video", "detect_video", "run_video", "predict", "process"):
        if hasattr(det, name):
            try:
                res = getattr(det, name)(video_path)
                return _normalize_result(res, video_path)
            except TypeError as e:
                # có thể hàm này nhận bytes thay vì path -> thử bytes
                last_err = e
                try:
                    with open(video_path, "rb") as fh:
                        res = getattr(det, name)(fh.read())
                    return _normalize_result(res, video_path)
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e
    raise AttributeError(
        f"BehaviorDetector không có hàm infer phù hợp "
        f"(đã thử: process_video, detect_video, run_video, predict, process). "
        f"Lỗi cuối: {last_err!r}"
    )


# ---------- Endpoint ----------
@router.post("")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
):
    # Lấy detector đã init ở app.state
    det: BehaviorDetector | None = getattr(request.app.state, "detector", None)
    if det is None:
        raise HTTPException(500, "Detector not ready")

    # 1) Lưu upload vào uploads/
    uploaded_abs = _save_upload_to_project(file)  # tuyệt đối
    uploaded_rel = _to_posix_rel(uploaded_abs)    # tương đối POSIX cho client

    # 2) Gọi detector
    try:
        res = _call_detector(det, str(uploaded_abs))
    except Exception as e:
        raise HTTPException(500, f"Detector error: {e!r}")

    # 3) Chuẩn hoá events
    events_raw = res.get("events") or []
    video_name = res.get("video_name") or Path(file.filename or uploaded_abs.name).stem
    events = [_norm_event(e) for e in events_raw]

    # 4) Ghi DB
    try:
        rows = [
            dict(
                video_name=video_name,
                frame_idx=e["frame_idx"],
                ts_sec=e["timestamp"],
                typ=e["type"],          # nếu cột là 'type' thì đổi thành: type=e["type"]
                obj_name=e["obj_name"],
                score=e["score"],
            )
            for e in events
        ]
        if rows:
            bulk_insert_events(db, rows)
    except Exception as e:
        raise HTTPException(500, detail=f"DB error: {e!r}")

    # 5) Segments (tuỳ chọn)
    segs = []
    try:
        segs = events_to_segments(events, gap_s=1.0, min_len_s=0.30) or []
        if segs:
            seg_rows = [
                dict(
                    video_name=video_name,
                    typ=(s.get("type") if isinstance(s, dict) else getattr(s, "type", None)),
                    start_ts=(s.get("start_ts") if isinstance(s, dict) else getattr(s, "start_ts", None)),
                    end_ts=(s.get("end_ts") if isinstance(s, dict) else getattr(s, "end_ts", None)),
                    peak_score=(s.get("peak_score") if isinstance(s, dict) else getattr(s, "peak_score", None)),
                    num_frames=(s.get("num_frames") if isinstance(s, dict) else getattr(s, "num_frames", None)),
                )
                for s in segs
            ]
            insert_segments(db, seg_rows)
    except Exception:
        segs = []

    # 6) Chuẩn hoá đường dẫn trả về (tương đối POSIX)
    overlay_rel = _to_posix_rel(res.get("overlay_video"))
    csv_rel     = _to_posix_rel(res.get("csv"))

    return {
        "video_name": video_name,
        "original_video": uploaded_rel,   # để front-end có thể phát bản gốc
        "n_events": len(events),
        "n_segments": len(segs),
        "overlay_video": overlay_rel,
        "csv": csv_rel,
        "events": events,
    }
