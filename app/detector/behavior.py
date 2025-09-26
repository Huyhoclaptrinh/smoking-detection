from __future__ import annotations
import os, time, datetime as dt
from pathlib import Path
from typing import List
import numpy as np, cv2
from collections import deque
from ultralytics import YOLO

from app.logging import get_logger
from app.utils.fs import ensure_dir, video_writer
from app.detector.config import DetectionConfig
from app.detector.events import Event
from app.detector.calibrator import CalibratorHelper
from app.detector.drink_roi import detect_drink_rois_multi, classify_drink_roi
from app.detector.postprocess import classwise_filter_softnms
from app.detector.config_postprocess import CONF_CFG, SOFT_CFG
from app.utils.vision_basic import iou_xyxy
from app.services.autoft import evaluate_autoft_from_events, try_save_last_autoft_ts
from app.workers.triggers import try_trigger_finetune
from app.core.training import run_yolo_finetune
from app.utils.light_report import video_light_report, frame_light_stats

logger = get_logger(__name__)

# === light-preproc helpers (module-level) ===
import cv2, numpy as np

def _center_stats(bgr):
    H, W = bgr.shape[:2]
    roi = bgr[H//10:9*H//10, W//10:9*W//10]  # bỏ viền đen/đen nền
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Vm = float(hsv[:, :, 2].mean())
    Bm, Gm, Rm = [float(roi[:, :, i].mean()) for i in (0, 1, 2)]
    return Vm, Rm, Gm, Bm

def light_fix_balanced(bgr, red_ratio_thr=1.25, v_dark_thr=150, desat_factor=0.85):
    # 1) Gray-world WB
    b,g,r = cv2.split(bgr.astype(np.float32))
    mB, mG, mR = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    m = (mB+mG+mR)/3.0
    b = np.clip(b*(m/mB), 0, 255); g = np.clip(g*(m/mG), 0, 255); r = np.clip(r*(m/mR), 0, 255)
    bgr = cv2.merge([b,g,r]).astype(np.uint8)
    # 2) LAB-CLAHE (kênh L)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    bgr = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
    # 3) Desat nếu đỏ áp đảo
    Vm, Rm, Gm, Bm = _center_stats(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if Rm > red_ratio_thr * ((Gm + Bm)/2.0):
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * desat_factor, 0, 255).astype(np.uint8)
    # 4) Gamma nhẹ nếu còn tối
    if Vm < v_dark_thr:
        gamma = 1.12
        table = np.clip(((np.arange(256)/255.0)**(1.0/gamma))*255, 0, 255).astype(np.uint8)
        return cv2.LUT(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), table)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class BehaviorDetector:
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.log = get_logger(__name__)

        self.general_model = YOLO(cfg.general_model_path)
        self.smoke_model   = YOLO(cfg.smoke_model_path)
        self.smoke_weight_model = YOLO(cfg.smoke_weight_path) if cfg.smoke_weight_path else None
        self.drink_class2id = {"bottle":0, "cup":1, "wine glass":2}

        ensure_dir(cfg.output_dir)

        # tham số foam cho bia
        self._foam_params = dict(
            s_max=110, v_min=170,
            require_amber_below=True,
            min_area=150,
            min_aspect=1.2,
            score_thresh=0.45,
            fallback_top_of_amber=True,
        )

        self._smoke_votes = deque(maxlen=5)
        self._drink_votes = deque(maxlen=5)
        self._hand_near_votes = deque(maxlen=5)
        self._last_smoke_frame = -10**9

        # trạng thái “drink” gần nhất để refill
        self._last_drink_state = None  # dict(box, name, score, frame_idx)

        self.calib = CalibratorHelper(cfg)
        self._last_mouth_pt = None

    # ---------- helpers ----------
    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return (0.5*(x1+x2), 0.5*(y1+y2))


    def _hand_mouth_debug(self, boxes_g, names_g, img_w, img_h):
        """
        Trả về dict: {mx,my,hx,hy,d_px,d_norm} của tay gần miệng nhất.
        None nếu thiếu tay/miệng trong khung.
        """
        mouth = self._estimate_mouth_point(boxes_g, names_g)
        if mouth is None:
            return None
        hands = self._hand_points(boxes_g, names_g)
        if not hands:
            return None

        mx, my = mouth
        hx, hy = min(hands, key=lambda p: (p[0]-mx)**2 + (p[1]-my)**2)
        d_px = ((hx-mx)**2 + (hy-my)**2) ** 0.5
        diag = (img_w**2 + img_h**2) ** 0.5
        d_norm = d_px / (diag + 1e-6)
        return dict(mx=float(mx), my=float(my), hx=float(hx), hy=float(hy),
                    d_px=float(d_px), d_norm=float(d_norm))
    
    def _bottle_tip_from_kept(self, boxes_g, names_g):
        """Lấy đỉnh (top-center) của ROI đồ uống làm tay giả. Ưu tiên 'bottle'."""
        if not len(boxes_g): return None
        pri = {"bottle": 0, "cup": 1, "wine glass": 2}
        cand = []
        for b, n in zip(boxes_g, names_g):
            if n in ("bottle", "cup", "wine glass"):
                x1,y1,x2,y2 = map(float, b)
                tip = (0.5*(x1+x2), y1)  # top-center
                cand.append((pri.get(n, 9), tip))
        if not cand: return None
        cand.sort(key=lambda x: x[0])
        return cand[0][1]



    def _estimate_mouth_point(self, boxes_g, names_g):
        """mouth -> face/head -> person -> fallback last mouth. Trả về (x,y) hoặc None."""
        if not len(boxes_g):
            return self._last_mouth_pt

        lower = [n.lower() for n in names_g] if names_g else []

        # mouth
        for i, n in enumerate(lower):
            if "mouth" in n:
                x1, y1, x2, y2 = boxes_g[i]
                pt = (0.5*(x1+x2), 0.5*(y1+y2))
                self._last_mouth_pt = pt
                return pt
        # face/head
        for key in ("face", "head"):
            for i, n in enumerate(lower):
                if key in n:
                    x1, y1, x2, y2 = boxes_g[i]
                    pt = (0.5*(x1+x2), y1 + 0.4*(y2-y1))
                    self._last_mouth_pt = pt
                    return pt
        # person
        for i, n in enumerate(lower):
            if "person" in n:
                x1, y1, x2, y2 = boxes_g[i]
                pt = (0.5*(x1+x2), y1 + 0.35*(y2-y1))
                self._last_mouth_pt = pt
                return pt

        # fallback: dùng miệng cũ nếu có
        return self._last_mouth_pt


    def _hand_points(self, boxes_g, names_g):
        pts = []
        if not len(boxes_g) or not names_g: return pts
        for i, n in enumerate(names_g):
            if "hand" in n.lower():
                x1, y1, x2, y2 = boxes_g[i]
                pts.append((0.5*(x1+x2), 0.5*(y1+y2)))
        return pts

    def _hand_mouth_dist_norm(self, boxes_g, names_g, img_w, img_h):
        """Khoảng cách ngắn nhất tay-miệng, đã chuẩn hóa theo đường chéo ảnh."""
        mouth = self._estimate_mouth_point(boxes_g, names_g)
        hands = self._hand_points(boxes_g, names_g)
        if (mouth is None) or (not hands): return None
        mx, my = mouth
        dmin = min(((hx-mx)**2 + (hy-my)**2)**0.5 for (hx,hy) in hands)
        diag = (img_w**2 + img_h**2) ** 0.5
        return float(dmin / (diag + 1e-6))

    def _hand_near_mouth_bool(self, boxes_g, names_g, img_w, img_h):
        d = self._hand_mouth_dist_norm(boxes_g, names_g, img_w, img_h)
        if d is None: return False, None
        thr = float(getattr(self.cfg, "hand_near_dist_frac", 0.09))
        return (d <= thr), d

    def _vote_and_emit(self, q: deque, passed: bool, need: int, ev_maker) -> bool:
        q.append(1 if passed else 0)
        if sum(q) >= need and passed:
            ev_maker(); q.clear(); return True
        return False

    # ---------- main ----------
    def process(self, video_path: str, save_overlay: bool = True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Báo cáo ánh sáng tổng hợp (tùy chọn)
        if getattr(self.cfg, "light_report_csv", None):
            try:
                rep = video_light_report(
                    video_path,
                    stride=getattr(self.cfg, "light_report_stride", 2),
                    center_frac=getattr(self.cfg, "light_report_center_frac", 0.85),
                    save_csv_path=self.cfg.light_report_csv,
                )
                self.log.info(f"[light_report] {rep}")
            except Exception as e:
                self.log.warning(f"[light_report] failed: {e}")

        events: List[Event] = []
        out_path, writer = None, None
        if save_overlay:
            out_path = str(Path(self.cfg.output_dir) / (Path(video_path).stem + "_overlay.mp4"))
            writer = video_writer(out_path, fps, w, h)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.cfg.frame_skip != 0:
                frame_idx += 1
                continue

            ts_sec = frame_idx / fps
            ts_hms = str(dt.timedelta(seconds=ts_sec))
            overlay = frame.copy() if writer is not None else None
            emitted_this_frame = False

            # --- Light diag per frame ---
            if getattr(self.cfg, "light_diag_enable", False):
                stride = max(1, int(getattr(self.cfg, "light_diag_stride", 15)))
                if (frame_idx % stride) == 0:
                    st = frame_light_stats(frame, center_frac=getattr(self.cfg, "light_diag_center_frac", 0.85))
                    self._last_light = st
                    self.log.info(
                        "[light_diag] f=%d t=%.2fs V=%.1f±%.1f S=%.1f R×=%.2f U=%d%% O=%d%%",
                        frame_idx, ts_sec,
                        st["v_mean"], st["v_std"], st["s_mean"], st["red_ratio"],
                        int(st["under_frac"]*100), int(st["over_frac"]*100)
                    )
                    if overlay is not None and getattr(self.cfg, "light_diag_overlay", True):
                        txt = (f"V:{st['v_mean']:.0f} S:{st['s_mean']:.0f} "
                               f"R×:{st['red_ratio']:.2f} U:{st['under_frac']*100:.0f}% "
                               f"O:{st['over_frac']*100:.0f}%")
                        cv2.putText(overlay, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(overlay, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # ---- YOLO call ----
            frame_g = light_fix_balanced(frame)
            rg = self.general_model(frame_g, imgsz=self.cfg.imgsz, conf=0.05, iou=0.90, verbose=False)[0]
            rs = self.smoke_model(frame,   imgsz=self.cfg.imgsz, conf=0.05, iou=0.90, verbose=False)[0]

            # === SMOKE postproc ===
            xyxy_s = rs.boxes.xyxy.cpu().numpy() if rs.boxes is not None else np.zeros((0, 4))
            conf_s_raw = rs.boxes.conf.cpu().numpy() if rs.boxes is not None else np.zeros((0,))
            cls_s_raw  = rs.boxes.cls.cpu().numpy().astype(int) if rs.boxes is not None else np.zeros((0,), dtype=int)
            names_map_s = getattr(rs, 'names', getattr(self.smoke_model, 'names', {}))
            if xyxy_s.size > 0:
                xyxy_s_f, conf_s_f, cls_s_f = classwise_filter_softnms(xyxy_s, conf_s_raw, cls_s_raw, names_map_s, CONF_CFG, SOFT_CFG)
            else:
                xyxy_s_f, conf_s_f, cls_s_f = np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)
            boxes_s, conf_s = xyxy_s_f, conf_s_f

            # === GENERAL postproc ===
            xyxy_g = rg.boxes.xyxy.cpu().numpy() if rg.boxes is not None else np.zeros((0, 4))
            conf_g_raw = rg.boxes.conf.cpu().numpy() if rg.boxes is not None else np.zeros((0,))
            cls_g_raw  = rg.boxes.cls.cpu().numpy().astype(int) if rg.boxes is not None else np.zeros((0,), dtype=int)
            names_map_g = getattr(rg, 'names', getattr(self.general_model, 'names', {}))
            if xyxy_g.size > 0:
                xyxy_g_f, conf_g_f, cls_g_f = classwise_filter_softnms(xyxy_g, conf_g_raw, cls_g_raw, names_map_g, CONF_CFG, SOFT_CFG)
            else:
                xyxy_g_f, conf_g_f, cls_g_f = np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)

            boxes_g, conf_g, cls_g = xyxy_g_f, conf_g_f, cls_g_f
            names_g = [names_map_g[int(i)] for i in cls_g] if len(cls_g) else []
            _keep = getattr(self.cfg, 'keep_classes', ['bottle','cup','wine glass','cigarette'])
            if len(boxes_g):
                mask = [names_g[i] in _keep for i in range(len(names_g))] if len(names_g)==len(boxes_g) else [True]*len(boxes_g)
                boxes_g = np.array([b for b,m in zip(boxes_g,mask) if m], float) if len(boxes_g) else np.zeros((0,4))
                conf_g  = np.array([c for c,m in zip(conf_g,mask) if m], float) if len(conf_g) else np.zeros((0,))
                cls_g   = np.array([c for c,m in zip(cls_g,mask) if m], int) if len(cls_g) else np.zeros((0,), int)
                names_g = [n for n,m in zip(names_g,mask) if m] if names_g else []

            # --- tay gần miệng (có tay/miệng thật hoặc tay-giả/miệng-nhớ) ---
            names_raw = [names_map_g[int(i)] for i in cls_g_raw] if xyxy_g.size > 0 else []
            mouth_pt = self._estimate_mouth_point(xyxy_g, names_raw)

            # tay thực
            hand_pts = self._hand_points(xyxy_g, names_raw)

            # nếu không có tay -> thử tay giả từ đỉnh chai/cốc (sau merge kept)
            if (not hand_pts) and bool(getattr(self.cfg, "use_bottle_tip_as_hand", True)):
                tip = self._bottle_tip_from_kept(boxes_g, names_g)
                if tip is not None:
                    hand_pts = [tip]

            # tính khoảng cách (nếu có đủ 2 điểm)
            d_hm = None
            if mouth_pt is not None and hand_pts:
                mx, my = mouth_pt
                hx, hy = min(hand_pts, key=lambda p: (p[0]-mx)**2 + (p[1]-my)**2)
                d_px = ((hx-mx)**2 + (hy-my)**2) ** 0.5
                diag = (w**2 + h**2) ** 0.5
                d_hm = float(d_px / (diag + 1e-6))

            thr = float(getattr(self.cfg, "hand_near_dist_frac", 0.09))
            hand_near = (d_hm is not None) and (d_hm <= thr)
            self._hand_near_votes.append(1 if hand_near else 0)

            # log tay-miệng
            if getattr(self.cfg, "hand_diag_enable", True) and (frame_idx % int(getattr(self.cfg, "hand_diag_stride", 5)) == 0):
                self.log.info("[hand_diag] f=%d t=%.2fs d_norm=%s thr=%.3f near=%s",
                            frame_idx, ts_sec,
                            f"{d_hm:.4f}" if d_hm is not None else "None",
                            thr, hand_near)
                if overlay is not None and getattr(self.cfg, "hand_diag_overlay", True) and (d_hm is not None):
                    cv2.circle(overlay, (int(mx), int(my)), 5, (0,255,0), -1)
                    cv2.circle(overlay, (int(hx), int(hy)), 5, (255,0,0), -1)
                    cv2.line(overlay, (int(mx), int(my)), (int(hx), int(hy)), (255,255,255), 1)
                    cv2.putText(overlay, f"d={d_hm:.3f}", (int(mx)+6, max(0, int(my)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Ghi JSONL (tùy chọn)
                if getattr(self.cfg, "hand_diag_log_jsonl", False):
                    import json
                    if not getattr(self, "_hand_jsonl_path", None):
                        jpath = getattr(self.cfg, "hand_diag_jsonl_path", "") or str(
                            Path(self.cfg.output_dir) / "hand" / f"{Path(video_path).stem}_hand.jsonl"
                        )
                        Path(jpath).parent.mkdir(parents=True, exist_ok=True)
                        self._hand_jsonl_path = jpath

                    rec = {"frame": int(frame_idx), "time_s": float(ts_sec),
                        "near": bool(hand_near), "thr": thr}
                    if dbg: rec.update(dbg)
                    try:
                        with open(self._hand_jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception as e:
                        self.log.warning(f"[hand_diag] jsonl write failed: {e}")


            # === PRIORITY MERGE general+smoke (như bạn đang dùng) ===
            def _iou_xyxy(a, b):
                ax1, ay1, ax2, ay2 = [float(x) for x in a]; bx1, by1, bx2, by2 = [float(x) for x in b]
                inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
                iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
                inter = iw * ih
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                den = area_a + area_b - inter
                return (inter / den) if den > 0 else 0.0

            merge_iou = float(getattr(self.cfg, 'merge_iou', 0.5))
            _prio_w = getattr(self.cfg, 'priority_weights', {'cigarette': 1.10})

            items = []
            for b, s, c in zip(boxes_g, conf_g, cls_g):
                n = names_map_g[int(c)]; wgt = float(_prio_w.get(n, 1.0))
                items.append({'bbox': b, 'score': float(s)*wgt, 'raw': float(s), 'name': n, 'source': 'general'})
            if 'xyxy_s_f' in locals() and len(xyxy_s_f):
                _names_s = [names_map_s[int(i)] for i in cls_s_f] if len(cls_s_f) else []
                for b, s, n in zip(xyxy_s_f, conf_s_f, _names_s):
                    wgt = float(_prio_w.get(n, 1.0))
                    items.append({'bbox': b, 'score': float(s)*wgt, 'raw': float(s), 'name': n, 'source': 'smoke'})
            items.sort(key=lambda d: d['score'], reverse=True)
            kept = []
            for it in items:
                if all(_iou_xyxy(it['bbox'], kt['bbox']) <= merge_iou for kt in kept):
                    kept.append(it)

            # rebuild outputs
            g_boxes, g_confs, names_g2 = [], [], []
            s_boxes, s_confs = [], []
            for it in kept:
                if it['source'] == 'general':
                    g_boxes.append(it['bbox']); g_confs.append(it['raw']); names_g2.append(it['name'])
                else:
                    s_boxes.append(it['bbox']); s_confs.append(it['raw'])
            boxes_g = np.array(g_boxes, float).reshape((-1,4)) if g_boxes else np.zeros((0,4))
            conf_g  = np.array(g_confs, float) if g_confs else np.zeros((0,))
            names_g = names_g2
            boxes_s = np.array(s_boxes, float).reshape((-1,4)) if s_boxes else np.zeros((0,4))
            conf_s  = np.array(s_confs, float) if s_confs else np.zeros((0,))

            # Smoke weight (optional)
            has_sw, conf_sw = False, np.array([])
            if self.smoke_weight_model is not None:
                r_sw = self.smoke_weight_model(frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf_thres, verbose=False)[0]
                conf_sw = r_sw.boxes.conf.cpu().numpy() if r_sw.boxes is not None else np.zeros((0,))
                has_sw = (conf_sw.size > 0)

            # ---- SMOKE ---- (giữ logic cũ)
            if len(boxes_s) > 0:
                i = int(np.argmax(conf_s))
                cig_conf = float(conf_s[i])
                sx1, sy1, sx2, sy2 = boxes_s[i]
                bw = max(1.0, float(sx2 - sx1)); bh = max(1.0, float(sy2 - sy1))
                area_norm_s = float((bw * bh) / max(1.0, w * h))
                asp = float(bh / (bw + 1e-6))
                sw_c = float(conf_sw.max()) if conf_sw.size > 0 else 0.0
                hand_flag = 1.0 if hand_near else 0.0

                heur = 0.70 * cig_conf + 0.20 * (sw_c if has_sw else 0.0) + 0.10 * hand_flag
                feats = dict(cig_conf=cig_conf, sw_conf=sw_c, hand_flag=hand_flag,
                             area_norm=area_norm_s, aspect=asp)
                score = self.calib.apply_smoke(feats, heur)
                self.calib.log_smoke(feats, 1 if score >= self.cfg.unknown_min_conf else 0)

                thr = float(self.cfg.unknown_min_conf)
                if has_sw: thr -= 0.03
                if not has_sw and not hand_near: thr += 0.07
                fast_pass = cig_conf >= max(self.cfg.conf_thres + 0.25, 0.65)
                need_votes = 2 if cig_conf < 0.75 else 1
                passed = (score >= thr) or fast_pass

                emitted = self._vote_and_emit(
                    self._smoke_votes, passed, need=need_votes,
                    ev_maker=lambda: events.append(
                        Event(frame_idx, ts_sec, ts_hms, "smoke", "cigarette", float(score))
                    )
                )
                if emitted:
                    emitted_this_frame = True
                    self._last_smoke_frame = frame_idx
                elif self.cfg.emit_unknown:
                    events.append(Event(frame_idx, ts_sec, ts_hms, "unknown", "smoke_like", float(score)))
                    emitted_this_frame = True

            # ---- DRINK ----
            existing_types = {e.type for e in events if e.frame_idx == frame_idx}
            recent_smoke = (frame_idx - self._last_smoke_frame) <= int(fps * 1.5)

            emitted_drink = False
            yolo_drink_names = (self.cfg.drink_objs or ["bottle", "cup", "wine glass"])
            yolo_drink_boxes = [tuple(map(int, b)) for b, n in zip(boxes_g, names_g) if n in yolo_drink_names]

            if ("smoke" not in existing_types) and (not recent_smoke):
                # 1) smoke mạnh?
                smoke_conf_max = float(conf_s.max()) if len(conf_s) else 0.0
                sw_aux_max = float(conf_sw.max()) if (has_sw and conf_sw.size > 0) else 0.0
                strict_thr = float(getattr(self.cfg, "smoke_strict_conf", 0.72))
                scene_has_sw = (smoke_conf_max >= strict_thr) or (sw_aux_max >= strict_thr)

                # 2) ROI: foam + fallback YOLO (trên frame_g)
                rois = detect_drink_rois_multi(
                    frame_bgr=frame_g,
                    general_model=self.general_model,
                    imgsz=self.cfg.imgsz,
                    conf_thres=self.cfg.conf_thres,
                    drink_objs=self.cfg.drink_objs,
                    foam_params=self._foam_params
                )
                if not rois:
                    area_min = float(getattr(self.cfg, "drink_min_area_frac", 1/1200)) * w * h
                    rois = []
                    for (bxyxy, name, sc) in zip(boxes_g, names_g, conf_g):
                        if name in yolo_drink_names:
                            x1, y1, x2, y2 = map(int, bxyxy)
                            if (x2-x1)*(y2-y1) >= area_min:
                                rois.append((name, (x1, y1, x2, y2), "yolo_fallback"))

                kept = []
                for lab, box, *_ in rois:
                    x1, y1, x2, y2 = box
                    if (x2 - x1) * (y2 - y1) < (w * h) // 400:
                        continue
                    if any(iou_xyxy(box, b2) > 0.55 for _, b2 in kept):
                        continue
                    kept.append((lab, box))

                for lab, box in kept[:2]:
                    # 3) phân loại foam
                    subtype, beer_score = classify_drink_roi(frame, box, self._foam_params)

                    # 4) container & yolo_conf_max
                    container = "unknown"; container_conf = 0.0; yolo_conf_max = 0.0
                    for (bxyxy, name, sc) in zip(boxes_g, names_g, conf_g):
                        if name in yolo_drink_names and iou_xyxy(tuple(map(int, box)), tuple(map(int, bxyxy))) > 0.30:
                            if sc > container_conf:
                                container_conf = float(sc); container = name
                            yolo_conf_max = max(yolo_conf_max, float(sc))
                    if container == "unknown":
                        container = "cup"

                    x1, y1, x2, y2 = map(int, box)
                    area_norm = ((x2-x1)*(y2-y1)) / float(w*h + 1e-6)

                    # 5) hai score: foam & yolo-shortcut (nới theo ánh sáng/bottle to)
                    base = 0.45 if subtype == "beer_like" else 0.25
                    heur = float(min(0.95, base + 0.45 * beer_score))
                    feats = dict(yolo_conf_max=float(yolo_conf_max), beer_score=float(beer_score))
                    score_foam = self.calib.apply_drink(feats, heur)

                    shortcut_conf = float(getattr(self.cfg, "drink_yolo_shortcut_conf", 0.62))
                    st_local = getattr(self, "_last_light", None)
                    if st_local and ((st_local["red_ratio"] >= 1.35) or (st_local["v_mean"] <= 110)):
                        shortcut_conf = max(0.48, shortcut_conf - 0.10)
                    if container == "bottle" and area_norm >= 0.03:
                        shortcut_conf = min(shortcut_conf, 0.55)

                    score_yolo = 0.0
                    if yolo_conf_max >= shortcut_conf:
                        score_yolo = 0.55 + 0.40 * float(yolo_conf_max) + (0.05 if hand_near else 0.0)

                    if scene_has_sw:
                        overlaps = any(iou_xyxy(box, yb) > 0.20 for yb in yolo_drink_boxes)
                        score = max(score_foam, score_yolo) if (overlaps and subtype=="beer_like" and beer_score>=0.45) else 0.0
                    else:
                        score = max(score_foam, score_yolo)

                    passed = (score >= self.cfg.unknown_min_conf)
                    display_name = f"{container}(beer)" if subtype == "beer_like" else container

                    votes_need = 2
                    if (container == "bottle" and (area_norm >= 0.03 or hand_near)) or (hand_near and sum(self._hand_near_votes)>=2):
                        votes_need = 1

                    self.log.info(
                        "[drink_dbg] f=%d cont=%s yolo=%.3f area=%.4f beer=%.3f thr=%.2f score=%.3f pass=%s",
                        frame_idx, container, yolo_conf_max, area_norm, beer_score, shortcut_conf, score, passed
                    )

                    emitted = self._vote_and_emit(
                        self._drink_votes, passed, need=votes_need,
                        ev_maker=lambda: events.append(
                            Event(frame_idx, ts_sec, ts_hms, "drink", display_name, float(score))
                        )
                    )
                    if emitted:
                        emitted_drink = True
                        emitted_this_frame = True
                        # lưu trạng thái cho REFILL
                        self._last_drink_state = dict(
                            box=(x1,y1,x2,y2), name=display_name, score=float(score), frame_idx=frame_idx
                        )
                        if overlay is not None:
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,165,255), 2)
                            cv2.putText(overlay, f"{display_name}:{score:.2f}", (x1, max(0, y1-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

            # ---- REFILL (propagate) nếu khung hiện tại hụt detect ----
            if (not emitted_drink):
                st = self._last_drink_state
                if st is not None:
                    max_gap = int(getattr(self.cfg, "refill_miss_frames", 6))
                    persist_need = int(getattr(self.cfg, "hand_near_persist_frames", 2))
                    allow_no_hand = bool(getattr(self.cfg, "refill_allow_no_hand", True))

                    # tri-state: True/False/Unknown
                    hand_unknown = (d_hm is None)

                    if (frame_idx - st["frame_idx"]) <= max_gap:
                        # nếu còn box đồ uống hiện tại thì yêu cầu chút IoU để chắc ăn
                        iou_ok = True
                        if len(yolo_drink_boxes):
                            iou_ok = max(iou_xyxy(st["box"], b) for b in yolo_drink_boxes) >= float(getattr(self.cfg, "refill_min_iou", 0.10))

                        cond_votes = (sum(self._hand_near_votes) >= persist_need)
                        cond_nohand = (hand_unknown and allow_no_hand)

                        if iou_ok and (cond_votes or cond_nohand):
                            decay = float(getattr(self.cfg, "refill_decay", 0.92))
                            gap = frame_idx - st["frame_idx"]
                            score = max(0.0, min(1.0, st["score"] * (decay ** gap)))
                            if score >= float(self.cfg.unknown_min_conf) - 0.05:
                                events.append(Event(frame_idx, ts_sec, ts_hms, "drink", st["name"], float(score)))
                                emitted_this_frame = True
                                if overlay is not None:
                                    x1,y1,x2,y2 = st["box"]
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 120, 0), 1)
                                    cv2.putText(overlay, f"{st['name']} (refill):{score:.2f}", (x1, max(0,y1-6)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,120,0), 1)


            # ---- Suspicious ----
            existing_types = {e.type for e in events if e.frame_idx == frame_idx}
            if hand_near and not ({"smoke","drink"} & existing_types):
                events.append(Event(frame_idx, ts_sec, ts_hms, "suspicious", None, 1.0))
                emitted_this_frame = True

            # Unknown giảm spam (2 giây/lần)
            if getattr(self.cfg, "emit_unknown_for_empty_frames", True) and not emitted_this_frame:
                if (frame_idx // max(1, int(fps))) % 2 == 0:
                    events.append(Event(frame_idx, ts_sec, ts_hms, "unknown", None, 1.0))

            if writer is not None and overlay is not None:
                writer.write(overlay)
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Calibrator auto-train
        self.calib.maybe_retrain(force=False)

        # AutoFT evaluation
        try:
            total_frames = frame_idx
            logger.info(f"[autoft] ENTER evaluate | fps={fps:.2f}, total_frames={total_frames}")
            triggered = evaluate_autoft_from_events(events, fps, total_frames, self.cfg, try_trigger_finetune)
            if triggered:
                try_save_last_autoft_ts(time.time())
            logger.info(f"[autoft] EXIT evaluate | triggered={triggered}")
        except Exception as e:
            logger.warning(f"[autoft] ERROR during evaluate: {e}")

        return {"events": [e.__dict__ for e in events], "overlay_video": out_path}
