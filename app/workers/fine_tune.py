import threading, subprocess, sys
from pathlib import Path
from .. import config

def _train_smoke():
    # Cigarette dataset (your yaml already exists in that folder)
    yaml_path = config.PROJ_ROOT / "Cigarette.v1i.yolov8" / "data.yaml"
    cmd = [
        sys.executable, "-m", "ultralytics",
        "train",
        f"model={config.FINETUNE_WEIGHTS_SMOKE}",
        f"data={yaml_path}",
        "imgsz=640", "epochs=60",
        "close_mosaic=10", "patience=20",
        "project=runs/fine_tune", "name=smoke_vft"
    ]
    subprocess.run(cmd, check=True, cwd=config.PROJ_ROOT)

def _train_drink():
    yaml_path = config.PROJ_ROOT / "coco_drink_yolo" / "data.yaml"
    cmd = [
        sys.executable, "-m", "ultralytics",
        "train",
        f"model={config.FINETUNE_WEIGHTS_DRINK}",
        f"data={yaml_path}",
        "imgsz=768", "epochs=80",
        "close_mosaic=10", "patience=30",
        "project=runs/fine_tune", "name=drink_vft"
    ]
    subprocess.run(cmd, check=True, cwd=config.PROJ_ROOT)

def launch_finetune_async():
    def _run():
        try:
            _train_smoke()
        except Exception as e:
            print("[fine-tune] smoke failed:", e)
        try:
            _train_drink()
        except Exception as e:
            print("[fine-tune] drink failed:", e)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
