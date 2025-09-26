from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np, cv2
from ultralytics import YOLO
from app.utils.beer_foam import detect_foam_bands_oriented
from app.utils.beer_foam import beer_mask_hsv
from app.utils.vision_basic import iou_xyxy, _clip_xyxy, _roi_core, _amber_ratio, _foam_top_ratio

def detect_drink_rois_multi(frame_bgr, general_model: YOLO, imgsz: int, conf_thres: float, drink_objs=None, foam_params=None):
    H, W = frame_bgr.shape[:2]
    cands = []
    res = general_model.predict(source=frame_bgr, imgsz=imgsz, conf=conf_thres, verbose=False)[0]
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls  = res.boxes.cls.cpu().numpy().astype(int)
        names = [res.names[i] for i in cls]
        for (x1,y1,x2,y2), n in zip(xyxy, names):
            if n in (drink_objs or ["bottle","cup","wine glass"]):
                cands.append((n, (int(x1),int(y1),int(x2),int(y2)), "yolo"))

    for d in detect_foam_bands_oriented(frame_bgr, **(foam_params or {})):
        x,y,w,h = d["bbox"]
        padx,pady = int(0.25*w), int(0.4*h)
        box = (max(0,x-padx), max(0,y-pady), min(W-1,x+w+padx), min(H-1,y+h+pady))
        cands.append(("cup", box, "foam"))

    amber = beer_mask_hsv(frame_bgr)
    cnts,_ = cv2.findContours(amber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in (cnts or []):
        if cv2.contourArea(c) < max(400, (H*W)//3000): continue
        x,y,w,h = cv2.boundingRect(c)
        padx,pady = int(0.08*w), int(0.10*h)
        box = _clip_xyxy((x-padx, y-pady, x+w+padx, y+h+pady), W,H)
        if (box[3]-box[1]) < 0.7*(box[2]-box[0]):
            add = int(0.7*(box[2]-box[0]) - (box[3]-box[1]))
            box = (box[0], max(0, box[1]-add//2), box[2], min(H-1, box[3]+add-add//2))
        cands.append(("cup", box, "color"))

    prio = {'yolo':0, 'foam':1, 'color':2}
    cands = sorted(cands, key=lambda t: (prio.get(t[2],3), -((t[1][2]-t[1][0])*(t[1][3]-t[1][1]))))
    merged=[]
    for lab, box, src in cands:
        if all(iou_xyxy(box, b2) < 0.55 for _, b2, _ in merged):
            merged.append((lab, box, src))
    return merged

def classify_drink_roi(frame_bgr, box_xyxy, foam_params) -> tuple[Optional[str], float]:
    x1,y1,x2,y2 = map(int, box_xyxy)
    H, W = frame_bgr.shape[:2]
    x1,y1,x2,y2 = max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return (None, 0.0)  # <-- KHÔNG trả 'cup' ở đây nữa

    core = _roi_core(roi)
    amber = _amber_ratio(core)
    foam_top = _foam_top_ratio(
        roi, top_ratio=0.22,
        s_max=foam_params.get("s_max",110),
        v_min=foam_params.get("v_min",170)
    )
    h, w = roi.shape[:2]
    ar   = h / float(w + 1e-6)
    geom = np.clip((ar - 1.0) / 2.5, 0, 1)

    score_beer = 0.5*amber + 0.35*foam_top + 0.15*geom
    must_and = (amber >= 0.12) and (foam_top >= 0.10)

    if must_and and score_beer >= 0.36:
        subtype = "beer_like"
    else:
        subtype = None
        score_beer = max(0.0, score_beer - 0.10)  # phạt nhẹ khi không đủ điều kiện

    return (subtype, float(score_beer))

