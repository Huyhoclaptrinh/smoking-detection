from __future__ import annotations
import numpy as np, cv2
from typing import List, Dict, Optional, Tuple
from app.utils.vision_basic import _skin_mask_ycrcb

def foam_mask_hsv(img_bgr: np.ndarray, s_max: int = 90, v_min: int = 180) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, v_min], dtype=np.uint8)
    upper = np.array([179, s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def beer_mask_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([12, 90, 60], dtype=np.uint8)
    upper1 = np.array([36, 255, 255], dtype=np.uint8)
    lower2 = np.array([8, 110, 50], dtype=np.uint8)
    upper2 = np.array([22, 255, 200], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    skin = _skin_mask_ycrcb(img_bgr)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def _connected_components(mask: np.ndarray, min_area: int):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [0]
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep.append(i)
    filtered = np.isin(labels, keep).astype(np.uint8) * 255
    return cv2.connectedComponentsWithStats(filtered, connectivity=8)

def _rotate_roi(img, center, angle_deg, size):
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _clip_rect(x, y, w, h, W, H, pad=0):
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(W - x, w + 2*pad); h = min(H - y, h + 2*pad)
    return x, y, w, h

def _bbox_iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa, ya = max(x1,x2), max(y1,y2)
    xb, yb = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter + 1e-6
    return inter / union

def detect_foam_bands_oriented(
    img_bgr: np.ndarray,
    s_max: int = 90,
    v_min: int = 180,
    require_amber_below: bool = False,
    min_area: Optional[int] = None,
    max_band_thickness_ratio: float = 1.0,
    min_aspect: float = 1.05,
    below_offset_px: int = 10,
    score_thresh: float = 0.42,
    fallback_top_of_amber: bool = True
) -> List[Dict]:
    H, W = img_bgr.shape[:2]
    if min_area is None:
        min_area = max(200, (H * W) // 2000)

    foam = foam_mask_hsv(img_bgr, s_max=s_max, v_min=v_min)
    cnts, _ = cv2.findContours(foam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets: List[Dict] = []
    for c in cnts or []:
        if cv2.contourArea(c) < min_area: continue

        (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
        cx, cy, rw, rh = float(cx), float(cy), float(rw), float(rh)
        if rw < 1 or rh < 1: continue
        long_side, short_side = (max(rw, rh), min(rw, rh))
        ang = angle + (90.0 if rw < rh else 0.0)

        x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = _clip_rect(x, y, w, h, W, H, pad=int(0.2 * max(w, h)))
        roi = img_bgr[y:y + h, x:x + w]
        if roi.size == 0: continue

        single = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(single, [c - [x, y]], -1, 255, thickness=cv2.FILLED)

        center = (w // 2, h // 2)
        size = (w, h)
        roi_r    = _rotate_roi(roi,    center, -ang, size)
        single_r = _rotate_roi(single, center, -ang, size)

        amber_r = None
        if require_amber_below:
            amber_roi = beer_mask_hsv(roi)
            amber_r   = _rotate_roi(amber_roi, center, -ang, size)

        ys, xs = np.where(single_r > 0)
        if xs.size == 0 or ys.size == 0: continue
        x2, y2 = int(xs.min()), int(ys.min())
        w2, h2 = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if w2 <= 0 or h2 <= 0: continue

        aspect = (w2 + 1e-6) / (h2 + 1e-6)
        thickness_ratio = h2 / (w2 + 1e-6)
        if aspect < (min_aspect * 0.8) or thickness_ratio > max_band_thickness_ratio:
            continue

        roi_band = roi_r[y2:y2 + h2, x2:x2 + w2]
        band_msk = (single_r[y2:y2 + h2, x2:x2 + w2] > 0)
        if roi_band.size == 0 or band_msk.sum() == 0: continue

        hsv_band = cv2.cvtColor(roi_band, cv2.COLOR_BGR2HSV)
        S = hsv_band[:, :, 1].astype(np.float32) / 255.0
        V = hsv_band[:, :, 2].astype(np.float32) / 255.0
        whiteness  = float((1.0 - S[band_msk]).mean() * 0.6 + V[band_msk].mean() * 0.4)
        row_cov    = (band_msk.sum(axis=1) / max(1, band_msk.shape[1])).astype(np.float32)
        continuity = float(np.percentile(row_cov, 60))

        adjacency = 0.0
        if require_amber_below and amber_r is not None:
            y_beg = min(roi_r.shape[0] - 1, y2 + h2 + 1)
            y_end = min(roi_r.shape[0],     y2 + h2 + below_offset_px)
            if y_end > y_beg:
                strip = (amber_r[y_beg:y_end, x2:x2 + w2] > 0)
                adjacency = float(strip.any(axis=0).mean()) if strip.size else 0.0

        score = 0.55 * whiteness + 0.35 * continuity + 0.10 * adjacency
        if score < score_thresh: continue

        dets.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "rbox": [cx, cy, long_side, short_side, float(ang)],
            "score": round(float(score), 3),
            "features": {
                "aspect_w_over_h": round(float(aspect), 3),
                "thickness_ratio": round(float(thickness_ratio), 3),
                "whiteness":       round(float(whiteness), 3),
                "continuity":      round(float(continuity), 3),
                "adjacency":       round(float(adjacency), 3),
            }
        })

    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    final: List[Dict] = []
    for d in dets:
        if all(_bbox_iou(d["bbox"], f["bbox"]) <= 0.3 for f in final):
            final.append(d)

    if fallback_top_of_amber and not final:
        amber_full = beer_mask_hsv(img_bgr)
        num_a, labels_a, stats_a, _ = _connected_components(amber_full, max(200, (H * W) // 2500))
        for i in range(1, num_a):
            x = int(stats_a[i, cv2.CC_STAT_LEFT])
            y = int(stats_a[i, cv2.CC_STAT_TOP])
            w = int(stats_a[i, cv2.CC_STAT_WIDTH])
            h = int(stats_a[i, cv2.CC_STAT_HEIGHT])
            if w * h < max(400, (H * W) // 3000): continue

            y0 = max(0, y - 10)
            y1 = min(H, y + 5)
            if y1 <= y0: continue
            strip = img_bgr[y0:y1, x:x + w]
            foam_local = foam_mask_hsv(strip, s_max=s_max, v_min=v_min) > 0
            coverage = float(foam_local.any(axis=0).mean()) if foam_local.size else 0.0
            if coverage >= 0.35:
                vis_score = float(foam_local.mean())
                final.append({
                    "bbox":  [x, y0, w, y1 - y0],
                    "rbox":  None,
                    "score": round(0.4 + min(0.6, vis_score), 3),
                    "features": {"fallback_top_of_amber": True, "coverage": round(coverage, 3)}
                })
    return final
