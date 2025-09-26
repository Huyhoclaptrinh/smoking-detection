from __future__ import annotations
from pathlib import Path
import shutil, random, yaml
from typing import List, Tuple
from app.utils.fs import handle_remove_readonly
from app.logging import get_logger

logger = get_logger()
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def ensure_dirs(raw_dirs: list[str], proc_dirs: list[str]):
    for raw_dir in raw_dirs:
        (Path(raw_dir)/"images").mkdir(parents=True, exist_ok=True)
        (Path(raw_dir)/"labels").mkdir(parents=True, exist_ok=True)
    for proc_dir in proc_dirs:
        for subset in ("train","val"):
            (Path(proc_dir)/"images"/subset).mkdir(parents=True, exist_ok=True)
            (Path(proc_dir)/"labels"/subset).mkdir(parents=True, exist_ok=True)

def _iter_images(folder: Path):
    for ext in IMG_EXTS:
        yield from folder.glob(f"*{ext}")

def prepare_dataset(raw_dir: str, proc_dir: str, base_dirs: List[Tuple[str,str]], NC=3, NAMES=("smoke","drink","none")) -> str:
    # 1) clean proc
    for subset in ("train","val"):
        for group in ("images","labels"):
            folder = Path(proc_dir)/group/subset
            if folder.exists():
                shutil.rmtree(folder, onerror=handle_remove_readonly)
            folder.mkdir(parents=True, exist_ok=True)

    # 2) copy base
    for img_dir, lbl_dir in base_dirs:
        img_dir = Path(img_dir); lbl_dir = Path(lbl_dir)
        target = "val" if any(s in str(img_dir).lower() for s in ("val","valid")) else "train"
        for img_path in _iter_images(img_dir):
            shutil.copy(img_path, Path(proc_dir)/"images"/target/img_path.name)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(lbl_path, Path(proc_dir)/"labels"/target/lbl_path.name)

    # 3) feedback
    raw_img_dir = Path(raw_dir)/"images"
    raw_lbl_dir = Path(raw_dir)/"labels"
    raw_imgs = [p for p in _iter_images(raw_img_dir)]
    random.shuffle(raw_imgs)
    n = len(raw_imgs)
    if n == 0:
        raise RuntimeError(f"Không tìm thấy ảnh feedback trong {raw_img_dir}")

    if n == 1:
        train_list, val_list = raw_imgs, raw_imgs
    else:
        split = max(1, int(0.8*n))
        train_list = raw_imgs[:split]
        val_list   = raw_imgs[split:] or raw_imgs[:1]

    for img_list, subset in ((train_list,"train"),(val_list,"val")):
        for img_path in img_list:
            shutil.copy(img_path, Path(proc_dir)/"images"/subset/img_path.name)
            lbl_src = raw_lbl_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy(lbl_src, Path(proc_dir)/"labels"/subset/lbl_src.name)
            else:
                (Path(proc_dir)/"labels"/subset/(img_path.stem + ".txt")).write_text("", encoding="utf-8")

    # 4) data.yaml
    data_cfg = {"path": str(Path(proc_dir)),
                "train":"images/train", "val":"images/val",
                "nc": int(NC), "names": list(NAMES)}
    yaml_path = Path(proc_dir)/"data.yaml"
    yaml_path.write_text(yaml.dump(data_cfg, allow_unicode=True), encoding="utf-8")
    return str(yaml_path)

def _archive_feedback(raw_dir: str):
    raw = Path(raw_dir)
    if not raw.exists(): return
    arc = raw.parent / (raw.name + "_archive")
    (arc/"images").mkdir(parents=True, exist_ok=True)
    (arc/"labels").mkdir(parents=True, exist_ok=True)
    for p in (raw/"images").glob("*"):
        shutil.move(str(p), str(arc/"images"/p.name))
    for p in (raw/"labels").glob("*.txt"):
        shutil.move(str(p), str(arc/"labels"/p.name))
