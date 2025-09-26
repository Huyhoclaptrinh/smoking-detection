import cv2, json, time
from pathlib import Path
from typing import Optional, Tuple
from .. import config
from urllib.parse import unquote
import os

# ---------- utils ----------
def _ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

PROJ = config.PROJ_ROOT
UPLOADS = PROJ / "uploads"
OUTPUTS = PROJ / "outputs"

def resolve_media_path(p: str | os.PathLike | None) -> Path | None:
    """Turn a browser-reported path (possibly URL-encoded or relative) into a real local path."""
    if not p:
        return None
    s = unquote(str(p)).strip()

    # normalize slashes
    s = s.replace("\\", os.sep).replace("/", os.sep)

    # absolute path given?
    cand = Path(s)
    if cand.is_absolute():
        return cand

    # common relative prefixes we return from /predict
    if s.startswith("uploads" + os.sep) or s.startswith("outputs" + os.sep):
        return (PROJ / s).resolve()

    # sometimes only a file name comes back
    name = Path(s).name
    for base in (OUTPUTS, UPLOADS, PROJ):
        cand = (base / name)
        if cand.exists():
            return cand.resolve()

    # last resort: treat relative to project
    return (PROJ / s).resolve()

def extract_frame(video_path: Path, timestamp: float, out_dir: Path) -> Tuple[Path, Tuple[int,int]]:
    """Grab a frame (nearest second) from video using OpenCV."""
    _ensure(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = max(0, int(round(timestamp * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame at {timestamp}s from {video_path}")
    H, W = frame.shape[:2]
    out = out_dir / f"frame_{int(time.time())}_{frame_index}.jpg"
    cv2.imwrite(str(out), frame)
    return out, (W, H)

def _yolo_xyxy_to_txt(x1, y1, x2, y2, W, H):
    # normalized xywh
    cx = ((x1+x2)/2) / W
    cy = ((y1+y2)/2) / H
    ww = (x2-x1) / W
    hh = (y2-y1) / H
    return cx, cy, ww, hh

# ---------- main entry ----------
def save_feedback_sample(
    *,
    video_path: Path,
    timestamp: float,
    correct_type: str,      # "smoke" | "drink" | "none"
    correct_obj: Optional[str],  # "cigarette" | "bottle" | "cup" | "wine glass" | None
    pred_box_xyxy: Optional[Tuple[int,int,int,int]] = None,
    # For missed detections, you can pass a box drawn from UI:
    user_box_xyxy: Optional[Tuple[int,int,int,int]] = None,
) -> Path:
    """
    Returns path to the saved image. Also writes YOLO label if applicable.
    - smoke + cigarette: goes to Cigarette.v1i.yolov8/{train|valid}/images with NO .txt (negative class)
      (your cigarette dataset treats empty label as “non-smoking” frame)
    - drink + {bottle|cup|wine glass}: goes to coco_drink_yolo/images + .txt label
    - none: push to negatives (no .txt) to help reduce FPs
    """
    # 1) Dump raw frame
    day = time.strftime("%Y%m%d")
    raw_dir = config.FEEDBACK_RAW_DIR / day
    img_path, (W, H) = extract_frame(video_path, timestamp, raw_dir)

    # 2) Decide destination + label writing
    if correct_type == "smoke":
        # Your convention: "non_smoking" images -> Cigarette dataset with empty label file.
        # If this feedback is to correct a SMOKING prediction to NON, set correct_type="none".
        # If instead you really want a POSITIVE 'cigarette' class, write a label file here.
        # By your previous pipeline, you're using empty .txt for negatives.
        dst_img = (config.CIG_DATASET / "train" / "images" / img_path.name)
        _ensure(dst_img.parent)
        dst_lbl = (config.CIG_DATASET / "train" / "labels" / (img_path.stem + ".txt"))
        # If you want “positive cigarette” use the box (prefer user_box, else predicted)
        if correct_obj == "cigarette" and (user_box_xyxy or pred_box_xyxy):
            x1,y1,x2,y2 = (user_box_xyxy or pred_box_xyxy)
            cx,cy,ww,hh = _yolo_xyxy_to_txt(x1,y1,x2,y2,W,H)
            _ensure(dst_lbl.parent)
            dst_lbl.write_text(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")
        else:
            # keep it negative for cigarette → empty file or no .txt (choose one)
            # here we write empty to be explicit
            _ensure(dst_lbl.parent)
            dst_lbl.write_text("")
        img_path.replace(dst_img)
        return dst_img

    elif correct_type == "drink" and correct_obj in {"bottle","cup","wine glass"}:
        # map your drink classes to IDs
        cls_map = {"bottle":0, "wine glass":1, "cup":2}  # adjust to your YAML order
        cls_id = cls_map[correct_obj]
        box = user_box_xyxy or pred_box_xyxy
        if not box:
            # No box? Park it for annotation later
            pending = config.FINETUNE_WORKDIR / "needs_box"
            _ensure(pending)
            (pending / f"{img_path.stem}.todo.json").write_text(json.dumps({
                "img": str(img_path), "W": W, "H": H,
                "reason": "no_box_for_drink", "obj": correct_obj
            }, ensure_ascii=False, indent=2))
            return img_path

        x1,y1,x2,y2 = box
        cx,cy,ww,hh = _yolo_xyxy_to_txt(x1,y1,x2,y2,W,H)
        dst_img = config.DRINK_DATASET / "images" / img_path.name
        dst_lbl = config.DRINK_DATASET / "labels" / (img_path.stem + ".txt")
        _ensure(dst_img.parent); _ensure(dst_lbl.parent)
        img_path.replace(dst_img)
        dst_lbl.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")
        return dst_img

    else:
        # 'none' → negative sample to fight FP (no label)
        dst_img = (config.DRINK_DATASET / "images" / img_path.name)
        _ensure(dst_img.parent)
        img_path.replace(dst_img)
        # no label file
        return dst_img
