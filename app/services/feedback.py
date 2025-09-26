from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import cv2, numpy as np
from app.detector.behavior import BehaviorDetector  # for model handles via app.state.detector
from app.utils.vision_basic import iou_xyxy
from app.logging import get_logger

logger = get_logger()

DRINK_CLASSES = ["bottle","cup","wine glass"]
DRINK_CLASS2ID = {n:i for i,n in enumerate(DRINK_CLASSES)}

VIDEO_SEARCH_DIRS = ["tmp","uploads","static","outputs"]
VIDEO_EXTS = [".mp4",".mov",".avi",".mkv",".webm"]

def _find_video_by_stem(stem: str) -> Optional[str]:
    for d in VIDEO_SEARCH_DIRS:
        for ext in VIDEO_EXTS:
            p = Path(d) / f"{stem}{ext}"
            if p.exists(): return str(p)
    p_overlay = Path("outputs") / f"{stem}_overlay.mp4"
    return str(p_overlay) if p_overlay.exists() else None

def extract_frame_by_index_or_ts(video_name: str, frame_idx: Optional[int]=None, timestamp_sec: Optional[float]=None):
    stem = Path(video_name).stem
    vpath = _find_video_by_stem(stem)
    if not vpath: return None
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx is None and timestamp_sec is not None:
        frame_idx = int(round(timestamp_sec * fps))
    if frame_idx is None: frame_idx = 0
    frame_idx = max(0, min(int(frame_idx), max(0, total-1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read(); cap.release()
    if not ok or frame is None: return None
    return frame

def _save_yolo_label(lbl_path: str, cls_id: int, xyxy, W: int, H: int) -> bool:
    x1, y1, x2, y2 = map(float, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1: return False
    cx = ((x1 + x2)/2.0) / W; cy = ((y1 + y2)/2.0) / H
    ww = (x2 - x1) / W;      hh = (y2 - y1) / H
    Path(lbl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")
    return True

def autobox_for_feedback(detector: BehaviorDetector, video_name: str,
                         frame_idx: int, correct_type: Optional[str],
                         correct_obj_name: Optional[str]=None) -> Optional[List[int]]:
    img = extract_frame_by_index_or_ts(video_name, frame_idx)
    if img is None: return None
    if (correct_type or "").lower() == "smoke":
        res = detector.smoke_model.predict(source=img, imgsz=detector.cfg.imgsz, conf=0.10, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            i = int(np.argmax(res.boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = map(int, res.boxes.xyxy.cpu().numpy()[i])
            return [x1, y1, x2, y2]
        return None
    if (correct_type or "").lower() == "drink":
        res_g = detector.general_model.predict(source=img, imgsz=detector.cfg.imgsz, conf=0.10, verbose=False)[0]
        cands = []
        if res_g.boxes is not None and len(res_g.boxes):
            xyxy = res_g.boxes.xyxy.cpu().numpy()
            cls  = res_g.boxes.cls.cpu().numpy().astype(int)
            conf = res_g.boxes.conf.cpu().numpy()
            for (b, ci, cf) in zip(xyxy, cls, conf):
                name = res_g.names[ci]
                if name in (detector.cfg.drink_objs or DRINK_CLASSES):
                    cands.append({"src":"yolo","name":name,"box":tuple(map(int,b)),"score":float(cf)})
        from app.detector.drink_roi import detect_drink_rois_multi, classify_drink_roi
        for lab, box, src in detect_drink_rois_multi(img, detector.general_model, detector.cfg.imgsz,
                                                     0.10, detector.cfg.drink_objs, foam_params=getattr(detector,"_foam_params",{})):
            label_hint, beer_score = classify_drink_roi(img, box, getattr(detector,"_foam_params",{}))
            cands.append({"src":src, "name":label_hint, "box":tuple(map(int,box)), "score":float(beer_score)})
        if not cands: return None
        tgt = (correct_obj_name or "").lower()
        def rank(c):
            x1,y1,x2,y2 = c["box"]; ar = (y2-y1) / max(1,(x2-x1)); bonus = 0.0
            if tgt == "wine glass": bonus += (0.8 if c["name"]=="wine glass" else 0) + max(0.0, ar-1.2)*0.3
            elif tgt == "bottle":  bonus += (0.8 if c["name"]=="bottle" else 0) + max(0.0, ar-1.3)*0.2
            elif tgt == "cup":     bonus += (0.8 if c["name"]=="cup" else 0) + max(0.0, 1.2-ar)*0.2
            if c["src"] == "yolo": bonus += 0.1
            return c["score"] + bonus
        best = max(cands, key=rank)
        return list(map(int, best["box"]))
    return None

def report_feedback(detector: BehaviorDetector,
                    video_name: str, frame_idx: int,
                    orig_type: str, correct_type: str | None,
                    correct_obj_name: str | None, bbox: list[float] | None):
    img = extract_frame_by_index_or_ts(video_name, frame_idx)
    if img is None: raise RuntimeError(f"Không trích được frame {frame_idx} từ video {video_name}")
    H, W = img.shape[:2]
    stem = f"{video_name}_f{frame_idx:06d}.jpg"
    if (correct_type or "").strip().lower() == "smoke":
        img_out = Path("feedback_raw/smoke")/"images"/stem
        lbl_out = Path("feedback_raw/smoke")/"labels"/stem.replace(".jpg",".txt")
        img_out.parent.mkdir(parents=True, exist_ok=True); lbl_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out), img)
        if bbox:
            ok = _save_yolo_label(str(lbl_out), 0, bbox, W=W, H=H)
            if not ok: lbl_out.write_text("", encoding="utf-8")
        else:
            lbl_out.write_text("", encoding="utf-8")
        return "smoke"
    elif (correct_type or "").strip().lower() == "drink":
        img_out = Path("feedback_raw/drink")/"images"/stem
        lbl_out = Path("feedback_raw/drink")/"labels"/stem.replace(".jpg",".txt")
        img_out.parent.mkdir(parents=True, exist_ok=True); lbl_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_out), img)
        if bbox and correct_obj_name:
            mapping = DRINK_CLASS2ID
            name = (correct_obj_name or "").lower().strip().replace("_"," ")
            cls_id = mapping.get(name, mapping.get("cup",1))
            ok = _save_yolo_label(str(lbl_out), cls_id, bbox, W=W, H=H)
            if not ok: lbl_out.write_text("", encoding="utf-8")
        else:
            lbl_out.write_text("", encoding="utf-8")
        return "drink"
    else:
        for base in ("feedback_raw/smoke","feedback_raw/drink"):
            img_out = Path(base)/"images"/stem
            lbl_out = Path(base)/"labels"/stem.replace(".jpg",".txt")
            img_out.parent.mkdir(parents=True, exist_ok=True)
            lbl_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_out), img)
            lbl_out.write_text("", encoding="utf-8")
        return "none"
