from __future__ import annotations
import glob, json, logging, os, threading, time
from datetime import datetime as dt
from pathlib import Path

log = logging.getLogger("finetune")
FEEDBACK_ROOT = Path("feedback_raw").resolve()   # bạn đổi nếu đường dẫn khác
# Cấu trúc mong đợi: feedback_raw/<typ>/images/*.jpg|*.png
MIN_IMAGES = 1  # số ảnh tối thiểu để train 1 lần (bạn chỉnh)

class FineTuneManager:
    def __init__(self, detector, cfg):
        self.detector = detector          # chỗ bạn gọi train thực tế
        self.cfg = cfg
        self.log = logging.getLogger("finetune")
        self.lock = threading.Lock()
        self.state = {
            "running": False,
            "last_start": None,
            "last_end": None,
            "last_error": None,
            "last_type": None,
            "last_model_path": None,
            "last_n_samples": 0,
        }
        os.makedirs("logs", exist_ok=True)
        self._state_path = "logs/finetune_state.json"

    # --- quant ---
    def _count_files(self, folder: str) -> int:
        if not folder: return 0
        return len(glob.glob(os.path.join(folder, "*.*")))  # jpg/png…

    def counts(self):
        return {
            "smoke": self._count_files(self.cfg.get("SMOKE_PROC_DIR")),
            "drink": self._count_files(self.cfg.get("DRINK_PROC_DIR")),
        }

    def thresholds(self):
        return {
            "smoke": int(self.cfg.get("MIN_FB_SMOKE", 1)),
            "drink": int(self.cfg.get("MIN_FB_DRINK", 1)),
        }

    def _dump(self):
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump({**self.state, "counts": self.counts(), "thresholds": self.thresholds()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log.warning("Dump state failed: %s", e)

    # --- public ---
    def start_periodic(self, interval_sec: int = 60):
        t = threading.Thread(target=self._loop, args=(interval_sec,), daemon=True)
        t.start()

    def maybe_run(self):
        th = self.thresholds()
        ct = self.counts()
        if not self.state["running"]:
            if ct["smoke"] >= th["smoke"]:
                return self._run_async("smoke")
            if ct["drink"] >= th["drink"]:
                return self._run_async("drink")
        return False
    
    def _count_raw_images(self, typ: str) -> int:
        img_dir = (FEEDBACK_ROOT / typ / "images").resolve()
        n = 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            n += len(glob.glob(str(img_dir / ext)))
        return n

    def run_once(self, typ: str, *, force: bool = False) -> bool:
        # (giữ logic ngưỡng của bạn nếu có)
        if not force:
            th = self.thresholds()
            ct = self.counts()
            need = int(th.get(typ, 999999))
            have = int(ct.get(typ, 0))
            if have < need:
                log.info("Fine-tune skip: not enough feedback count (%s: %s/%s)", typ, have, need)
                return False

        # ✅ NEW: phải có dataset thô tối thiểu
        raw_imgs = self._count_raw_images(typ)
        if raw_imgs < MIN_IMAGES:
            log.info("Fine-tune skip: not enough RAW images in feedback_raw/%s/images (have=%d, need=%d)",
                     typ, raw_imgs, MIN_IMAGES)
            return False

        # chạy thật
        return self._run(typ)  # gọi hàm nội bộ của bạn

    # --- internals ---
    def _loop(self, interval):
        self.log.info("FineTuneManager loop started, every %s s", interval)
        while True:
            try:
                self.maybe_run()
            except Exception as e:
                self.log.exception("fine-tune periodic check error: %s", e)
            time.sleep(interval)

    def _run_async(self, typ: str):
        if self.state["running"]:
            return False
        t = threading.Thread(target=self._run, args=(typ,), daemon=True)
        t.start()
        return True

    def _run(self, typ: str):
        with self.lock:
            self.state.update({
                "running": True,
                "last_start": dt.utcnow().isoformat(),
                "last_end": None,
                "last_error": None,
                "last_type": typ,
                "last_model_path": None,
            })
            self._dump()
            counts = self.counts()
            self.log.info("==> Fine-tune START type=%s  samples=%s", typ, counts.get(typ, 0))
            try:
                # >>> CHỖ GỌI TRAIN THỰC TẾ <<<
                # Thay dòng dưới bằng hàm train của bạn.
                # Ví dụ nếu bạn có detector.fine_tune(task, epochs, lr0):
                if hasattr(self.detector, "fine_tune"):
                    epochs = int(self.cfg.get("EPOCHS_FINETUNE_SMOKE" if typ=="smoke" else "EPOCHS_FINETUNE_DRINK", 5))
                    lr0    = float(self.cfg.get("LR0_SMOKE" if typ=="smoke" else "LR0_DRINK", 1e-4))
                    out_model_path = self.detector.fine_tune(task=typ, epochs=epochs, lr0=lr0)  # <-- đổi cho đúng chữ ký của bạn
                else:
                    # Nếu chưa có hàm, tạm log mô phỏng 5s
                    time.sleep(5)
                    out_model_path = f"training/epochs_finetuned/{typ}_{int(time.time())}.pt"

                self.state.update({
                    "running": False,
                    "last_end": dt.utcnow().isoformat(),
                    "last_model_path": out_model_path,
                    "last_n_samples": counts.get(typ, 0),
                })
                self._dump()
                self.log.info("<== Fine-tune DONE type=%s  model=%s", typ, out_model_path)
            except Exception as e:
                self.state.update({
                    "running": False,
                    "last_end": dt.utcnow().isoformat(),
                    "last_error": str(e),
                })
                self._dump()
                self.log.exception("Fine-tune ERROR type=%s: %s", typ, e)
