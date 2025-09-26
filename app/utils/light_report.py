# app/utils/light_report.py
from __future__ import annotations
import cv2, csv, numpy as np
from pathlib import Path

def _center_crop(img, frac: float = 0.8):
    h, w = img.shape[:2]
    dw, dh = int((1-frac)*w/2), int((1-frac)*h/2)
    return img[dh:h-dh, dw:w-dw] if (dh>0 and dw>0) else img

def frame_light_stats(bgr, center_frac: float = 0.8) -> dict:
    """
    Trả về các chỉ số ánh sáng/màu cho 1 frame:
      - v_mean, v_std (HSV V-channel)
      - s_mean (độ bão hòa)
      - r_mean,g_mean,b_mean
      - red_ratio = R_mean / ((G_mean+B_mean)/2)
      - under_frac: % pixel V<28 (tối)
      - over_frac:  % pixel V>235 (cháy)
    Lưu ý: cắt vùng giữa để bỏ viền đen/letterbox.
    """
    roi = _center_crop(bgr, center_frac)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    V = hsv[...,2].astype(np.float32); S = hsv[...,1].astype(np.float32)
    b, g, r = [roi[...,i].astype(np.float32) for i in (0,1,2)]

    v_mean = float(V.mean()); v_std = float(V.std())
    s_mean = float(S.mean())
    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
    red_ratio = float(r_mean / max(1.0, (g_mean + b_mean)/2.0))
    under_frac = float((V < 28).mean())     # frame rất tối
    over_frac  = float((V > 235).mean())    # cháy sáng

    return dict(
        v_mean=v_mean, v_std=v_std, s_mean=s_mean,
        r_mean=r_mean, g_mean=g_mean, b_mean=b_mean,
        red_ratio=red_ratio, under_frac=under_frac, over_frac=over_frac
    )

def video_light_report(video_path: str,
                       stride: int = 1,
                       center_frac: float = 0.8,
                       save_csv_path: str | None = None) -> dict:
    """
    Quét toàn bộ video theo bước 'stride' (1 = mọi frame).
    Ghi CSV nếu save_csv_path != None.
    Trả về summary (min/median/max cho v_mean & red_ratio, % khung tối/cháy).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % stride != 0:
            i += 1; continue
        t = i / fps
        st = frame_light_stats(frame, center_frac=center_frac)
        st_row = {
            "frame_idx": i,
            "time_s": round(t, 3),
            **{k: round(v, 6) for k,v in st.items()}
        }
        rows.append(st_row)
        i += 1
    cap.release()

    if save_csv_path:
        p = Path(save_csv_path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame_idx","time_s"])
            w.writeheader()
            for r in rows: w.writerow(r)

    # Summary
    if not rows:
        return {"frames": 0}
    v_all = np.array([r["v_mean"] for r in rows], dtype=np.float32)
    red_all = np.array([r["red_ratio"] for r in rows], dtype=np.float32)
    under_all = np.array([r["under_frac"] for r in rows], dtype=np.float32)
    over_all  = np.array([r["over_frac"]  for r in rows], dtype=np.float32)

    def s(a):  # stats helper
        return dict(min=float(a.min()), p5=float(np.percentile(a,5)),
                    median=float(np.median(a)), p95=float(np.percentile(a,95)),
                    max=float(a.max()))

    return {
        "frames": len(rows),
        "v_mean_stats": s(v_all),
        "red_ratio_stats": s(red_all),
        "pct_frames_dark": float((under_all > 0.10).mean()),  # >10% pixel rất tối
        "pct_frames_bright": float((over_all > 0.05).mean()), # >5% pixel rất sáng
        "csv": save_csv_path,
        "fps": float(fps), "stride": stride
    }
