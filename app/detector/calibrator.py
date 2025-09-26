from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import csv, time, joblib, numpy as np
from app.logging import get_logger

logger = get_logger()

class CalibratorHelper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.smoke_calib = None
        self.drink_calib = None
        self._last_ts = 0.0
        self._load_if_any()

    def _load_if_any(self):
        try:
            p = Path(self.cfg.smoke_calib_path)
            if p.exists(): self.smoke_calib = joblib.load(p); logger.info(f"[calib] Loaded smoke {p}")
        except Exception as e:
            logger.info(f"[calib] skip load smoke: {e}")
        try:
            p = Path(self.cfg.drink_calib_path)
            if p.exists(): self.drink_calib = joblib.load(p); logger.info(f"[calib] Loaded drink {p}")
        except Exception as e:
            logger.info(f"[calib] skip load drink: {e}")

    # --- logging ---
    def _log_row(self, csv_path: Path, header: List[str], row: List):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file: w.writerow(header)
            w.writerow(row)

    def log_smoke(self, features: Dict, label: int):
        hdr = ["cig_conf","sw_conf","hand_flag","area_norm","aspect","label"]
        row = [float(features.get("cig_conf",0.0)),
               float(features.get("sw_conf",0.0)),
               float(features.get("hand_flag",0.0)),
               float(features.get("area_norm",0.0)),
               float(features.get("aspect",1.0)),
               int(label)]
        self._log_row(Path(self.cfg.calib_log_dir)/"smoke_log.csv", hdr, row)

    def log_drink(self, features: Dict, label: int):
        hdr = ["yolo_conf_max","beer_score","amber","foam_top","ar","label"]
        row = [float(features.get("yolo_conf_max",0.0)),
               float(features.get("beer_score",0.0)),
               float(features.get("amber",0.0)),
               float(features.get("foam_top",0.0)),
               float(features.get("ar",1.0)),
               int(label)]
        self._log_row(Path(self.cfg.calib_log_dir)/"drink_log.csv", hdr, row)

    # --- train if enough data ---
    def maybe_retrain(self, force: bool=False):
        if not (self.cfg.calib_autotrain or force): return
        now = time.time()
        if not force and (now - self._last_ts) < self.cfg.calib_retrain_interval_sec: return
        try:
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
        except Exception as e:
            logger.warning(f"[calib] pandas/sklearn missing: {e}"); return

        def _safe_train(csv_path: Path, feat_cols: List[str], out_pkl: Path, kind: str):
            if not csv_path.exists(): return False
            import pandas as pd, numpy as np
            df = pd.read_csv(csv_path)
            if "label" not in df.columns: return False
            df = df.replace([np.inf,-np.inf], np.nan).dropna()
            if len(df) < self.cfg.calib_min_samples: return False
            y = df["label"].astype(int).values
            pos, neg = int((y==1).sum()), int((y==0).sum())
            if len(np.unique(y))<2 or pos < self.cfg.calib_min_pos or neg < self.cfg.calib_min_neg:
                logger.info(f"[calib] Skip {kind}: need pos/neg, got pos={pos}, neg={neg}, n={len(y)}")
                return False
            X = df[feat_cols].astype(np.float32).values
            try:
                clf = LogisticRegression(max_iter=400, class_weight="balanced")
                clf.fit(X, y)
                out_pkl.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(clf, out_pkl)
                if kind=="smoke": self.smoke_calib = clf
                else: self.drink_calib = clf
                logger.info(f"[calib] Trained {kind} -> {out_pkl}")
                return True
            except Exception as e:
                logger.warning(f"[calib] train {kind} failed: {e}")
                return False

        s_ok = _safe_train(Path(self.cfg.calib_log_dir)/"smoke_log.csv",
                           ["cig_conf","sw_conf","hand_flag","area_norm","aspect"],
                           Path(self.cfg.smoke_calib_path), "smoke")
        d_ok = _safe_train(Path(self.cfg.calib_log_dir)/"drink_log.csv",
                           ["yolo_conf_max","beer_score","amber","foam_top","ar"],
                           Path(self.cfg.drink_calib_path), "drink")
        if s_ok or d_ok: self._last_ts = now

    # --- apply ---
    def apply_smoke(self, features: Dict, fallback_score: float) -> float:
        if self.smoke_calib is None: return float(fallback_score)
        import numpy as np
        x = np.array([[features.get("cig_conf",0.0),
                       features.get("sw_conf",0.0),
                       features.get("hand_flag",0.0),
                       features.get("area_norm",0.0),
                       features.get("aspect",1.0)]], dtype=np.float32)
        try:
            return float(self.smoke_calib.predict_proba(x)[0,1])
        except Exception:
            return float(fallback_score)

    def apply_drink(self, features: Dict, fallback_score: float) -> float:
        if self.drink_calib is None: return float(fallback_score)
        import numpy as np
        x = np.array([[features.get("yolo_conf_max",0.0),
                       features.get("beer_score",0.0),
                       features.get("amber",0.0),
                       features.get("foam_top",0.0),
                       features.get("ar",1.0)]], dtype=np.float32)
        try:
            return float(self.drink_calib.predict_proba(x)[0,1])
        except Exception:
            return float(fallback_score)
