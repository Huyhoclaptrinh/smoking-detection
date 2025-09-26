from __future__ import annotations
import numpy as np, cv2

def iou_xyxy(b1, b2):
    x1,y1,x2,y2 = b1; X1,Y1,X2,Y2 = b2
    xa, ya = max(x1,X1), max(y1,Y1)
    xb, yb = min(x2,X2), min(y2,Y2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = max(1,(x2-x1)*(y2-y1)) + max(1,(X2-X1)*(Y2-Y1)) - inter
    return inter / union

def _clip_xyxy(box, W, H):
    x1,y1,x2,y2 = map(int, box)
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def _expand_from_foam_bbox(x,y,w,h, W,H, grow_down=1.8, grow_side=0.25, grow_up=0.40, min_ar=0.70):
    x1 = int(x - grow_side*w); x2 = int(x + w + grow_side*w)
    y1 = int(y - grow_up*h);   y2 = int(y + h + grow_down*h)
    x1,y1,x2,y2 = _clip_xyxy((x1,y1,x2,y2), W,H)
    h2, w2 = (y2-y1), (x2-x1)
    if h2 < min_ar * w2:
        add = int(min_ar * w2 - h2)
        y1 = max(0, y1 - add//2); y2 = min(H-1, y2 + add - add//2)
    return (x1,y1,x2,y2)

def _skin_mask_ycrcb(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, np.array([0,133,77],np.uint8), np.array([255,173,127],np.uint8))

def _roi_core(roi, vmin=0.25, vmax=0.85, hmin=0.15, hmax=0.85):
    H,W = roi.shape[:2]
    ya,yb = int(H*vmin), int(H*vmax)
    xa,xb = int(W*hmin), int(W*hmax)
    if ya>=yb or xa>=xb: return roi
    return roi[ya:yb, xa:xb]

def _amber_ratio(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    m = (H>=10)&(H<=38)&(S>60)&(V>50)
    return float(np.mean(m.astype(np.float32)))

def _foam_top_ratio(roi_bgr, top_ratio=0.22, s_max=120, v_min=170):
    H,W = roi_bgr.shape[:2]
    yb = max(2, int(H*top_ratio))
    top = roi_bgr[:yb, :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,v_min],np.uint8), np.array([179,s_max,255],np.uint8))
    return float((mask>0).mean())
