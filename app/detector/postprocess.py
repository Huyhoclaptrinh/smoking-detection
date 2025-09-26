from typing import Dict, List, Tuple
import numpy as np

def gaussian_soft_nms_single_class(
    boxes: np.ndarray,  # (N,4) xyxy
    scores: np.ndarray, # (N,)
    sigma: float = 0.5,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.001,
) -> List[int]:
    """
    Gaussian Soft-NMS for a *single class* set of boxes.
    Returns kept indices relative to the input order.
    """
    if len(boxes) == 0:
        return []
    b = boxes.copy().astype(np.float32)
    s = scores.copy().astype(np.float32)
    N = b.shape[0]
    keep = []

    for i in range(N):
        max_pos = i + np.argmax(s[i:])
        b[[i, max_pos]] = b[[max_pos, i]]
        s[[i, max_pos]] = s[[max_pos, i]]

        cur_box = b[i]
        cur_score = s[i]
        if cur_score < score_thresh:
            continue
        keep.append(i)

        rest = b[i+1:]
        if rest.shape[0] == 0:
            continue
        xx1 = np.maximum(cur_box[0], rest[:,0])
        yy1 = np.maximum(cur_box[1], rest[:,1])
        xx2 = np.minimum(cur_box[2], rest[:,2])
        yy2 = np.minimum(cur_box[3], rest[:,3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area1 = (cur_box[2]-cur_box[0]) * (cur_box[3]-cur_box[1])
        area2 = (rest[:,2]-rest[:,0]) * (rest[:,3]-rest[:,1])
        iou = inter / (area1 + area2 - inter + 1e-9)

        weight = np.exp(-(iou**2)/sigma)
        s[i+1:] = s[i+1:] * np.where(iou > iou_thresh, weight, 1.0)

    return keep

def classwise_filter_softnms(
    xyxy: np.ndarray,        # (N,4)
    conf: np.ndarray,        # (N,)
    cls: np.ndarray,         # (N,)
    names: Dict[int, str],   # model.names or result.names
    conf_cfg: Dict[str, float],   # per-class conf min
    soft_cfg: Dict[str, Dict],    # per-class soft-nms params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply per-class thresholds and class-wise Soft-NMS.
    Returns filtered xyxy, conf, cls arrays.
    """
    out_boxes, out_scores, out_cls = [], [], []
    for cid, cname in names.items():
        mask = (cls == cid)
        if not mask.any():
            continue
        b = xyxy[mask]
        s = conf[mask]
        # Per-class threshold
        th = conf_cfg.get(cname, conf_cfg.get("_default", 0.35))
        keep1 = s >= th
        b, s = b[keep1], s[keep1]
        if len(s) == 0:
            continue
        # Per-class soft-nms params
        params = soft_cfg.get(cname, soft_cfg.get("_default", {}))
        sigma = params.get("sigma", 0.5)
        iou_t = params.get("iou", 0.5)
        score_t = params.get("score", 0.001)

        kept_rel_idx = gaussian_soft_nms_single_class(b, s, sigma=sigma, iou_thresh=iou_t, score_thresh=score_t)
        if len(kept_rel_idx) == 0:
            continue
        kept_rel_idx = np.array(kept_rel_idx, dtype=int)
        out_boxes.append(b[kept_rel_idx])
        out_scores.append(s[kept_rel_idx])
        out_cls.append(np.full(kept_rel_idx.shape[0], cid, dtype=cls.dtype))
    if len(out_scores) == 0:
        return (np.empty((0,4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=cls.dtype))
    return np.vstack(out_boxes), np.concatenate(out_scores), np.concatenate(out_cls)
