# app/core/training.py
from __future__ import annotations
import os, shutil, random, yaml, time, logging
from pathlib import Path
from typing import Tuple, List
from ultralytics import YOLO

logger = logging.getLogger("ft")

# ===== Base dataset (nếu bạn có sẵn các thư mục này) =====
SMOKE_BASE_DIRS = [
    ("Cigarette.v1i.yolov8/train/images", "Cigarette.v1i.yolov8/train/labels"),
    ("Cigarette.v1i.yolov8/valid/images", "Cigarette.v1i.yolov8/valid/labels"),
]
DRINK_BASE_DIRS = [
    ("coco_drink_yolo/images/train", "coco_drink_yolo/labels/train"),
    ("coco_drink_yolo/images/val",   "coco_drink_yolo/labels/val"),
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _iter_images(folder: Path):
    for p in folder.glob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def _handle_remove_readonly(func, path, exc_info):
    import stat, errno
    excvalue = exc_info[1]
    if func in (os.rmdir, os.remove, os.unlink) and getattr(excvalue, "errno", None) == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        func(path)
    else:
        raise

def _prepare_dataset(raw_dir: str, proc_dir: str, base_dirs: List[tuple[str,str]],
                     nc: int, names: List[str]) -> str:
    """Dọn proc_dir, copy base + feedback vào train/val, sinh data.yaml."""
    proc = Path(proc_dir)
    for subset in ("train", "val"):
        for group in ("images", "labels"):
            d = proc / group / subset
            if d.exists():
                shutil.rmtree(d, onerror=_handle_remove_readonly)
            d.mkdir(parents=True, exist_ok=True)

    # copy base
    for img_dir, lbl_dir in base_dirs:
        img_dir = Path(img_dir); lbl_dir = Path(lbl_dir)
        target = "val" if any(s in str(img_dir).lower() for s in ("val","valid")) else "train"
        if img_dir.exists():
            for ip in _iter_images(img_dir):
                shutil.copy(ip, proc / "images" / target / ip.name)
                lp = lbl_dir / (ip.stem + ".txt")
                if lp.exists():
                    shutil.copy(lp, proc / "labels" / target / lp.name)

    # copy feedback
    raw_img = Path(raw_dir) / "images"
    raw_lbl = Path(raw_dir) / "labels"
    imgs = [p for p in _iter_images(raw_img)] if raw_img.exists() else []
    random.shuffle(imgs)
    n = len(imgs)
    if n == 0:
        raise RuntimeError(f"Không tìm thấy ảnh feedback trong {raw_img}")

    if n == 1:
        train_list, val_list = imgs, imgs
    else:
        split = max(1, int(0.8 * n))
        train_list = imgs[:split]
        val_list   = imgs[split:] or imgs[:1]

    for img_list, subset in ((train_list,"train"), (val_list,"val")):
        for ip in img_list:
            shutil.copy(ip, proc / "images" / subset / ip.name)
            lp = raw_lbl / (ip.stem + ".txt")
            out_lbl = proc / "labels" / subset / (ip.stem + ".txt")
            if lp.exists():
                shutil.copy(lp, out_lbl)
            else:
                out_lbl.write_text("", encoding="utf-8")

    data_cfg = {
        "path": str(proc.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    int(nc),
        "names": list(names),
    }
    yml = proc / "data.yaml"
    yml.write_text(yaml.dump(data_cfg, allow_unicode=True), encoding="utf-8")
    return str(yml)

def _load_cfg():
    p = Path(os.getenv("CFG_PATH", "config.yaml"))
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_yolo_finetune(part: str, det_cfg) -> Tuple[bool, str]:
    """
    Worker sẽ gọi hàm này.
    Trả về (ok, ckpt_path).
    """
    try:
        cfg = _load_cfg()
        # đường dẫn feedback/raw/proc (có trong config.yaml của bạn)
        SMOKE_RAW_DIR  = cfg.get("SMOKE_RAW_DIR", "feedback_raw/smoke")
        SMOKE_PROC_DIR = cfg.get("SMOKE_PROC_DIR", "feedback_proc/smoke")
        DRINK_RAW_DIR  = cfg.get("DRINK_RAW_DIR", "feedback_raw/drink")
        DRINK_PROC_DIR = cfg.get("DRINK_PROC_DIR", "feedback_proc/drink")

        # model gốc
        smoke_ckpt   = cfg.get("SMOKE_MODEL", "training/epochs_25/smoke.pt")
        general_ckpt = cfg.get("GENERAL_MODEL", "training/epochs_25/best.pt")

        # tham số train
        epochs = int(cfg.get("EPOCHS_FINETUNE_SMOKE" if part=="smoke" else "EPOCHS_FINETUNE_DRINK", 25))
        imgsz  = int(cfg.get("IMGSZ", 640))
        device = cfg.get("FT_DEVICE", 0)
        nc     = int(cfg.get("NC", 3))
        names  = list(cfg.get("NAMES", ["smoke","drink","none"]))

        if part == "smoke":
            data_yaml = _prepare_dataset(SMOKE_RAW_DIR, SMOKE_PROC_DIR, SMOKE_BASE_DIRS, nc, names)
            base_ckpt = smoke_ckpt
        elif part == "drink":
            data_yaml = _prepare_dataset(DRINK_RAW_DIR, DRINK_PROC_DIR, DRINK_BASE_DIRS, nc, names)
            base_ckpt = general_ckpt
        else:
            logger.error(f"[ft] Unknown part: {part}")
            return False, ""

        logger.info(f"[ft] start training {part} | base={base_ckpt} epochs={epochs} imgsz={imgsz} device={device}")
        model = YOLO(base_ckpt)
        run_name = f"{part}_ft_{int(time.time())}"
        results = model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz,
            project="training", name=run_name, verbose=True, device=device
        )
        ckpt = Path(results.save_dir) / "weights" / "best.pt"
        return (ckpt.exists(), str(ckpt) if ckpt.exists() else "")
    except Exception as e:
        logger.exception(f"[ft] run_yolo_finetune fatal: {e}")
        return False, ""
