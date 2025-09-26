# app/config.py
import os, yaml

def load_cfg(path: str | None = None):
    cfg_path = os.getenv("CFG_PATH", path or "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Nếu YAML đã có DB_URL thì giữ nguyên, nếu không mới build từ mảnh ghép
    if not cfg.get("DB_URL"):
        db_user = cfg.get("DB_USER","root")
        db_pass = cfg.get("DB_PASS","1")
        db_host = cfg.get("DB_HOST","127.0.0.1")
        db_port = cfg.get("DB_PORT",3306)
        db_name = cfg.get("DB_NAME","behavior_detect")
        cfg["DB_NAME"] = db_name
        cfg["DB_URL"]  = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"

    return cfg

from pathlib import Path

# where your base project lives (adjust if needed)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# incoming user feedback frames (raw)
FEEDBACK_RAW_DIR = PROJ_ROOT / "feedback_raw"

# processed training shards (go straight into YOLO datasets)
#  - cigarette (non-smoking negatives = NO .txt file)
CIG_DATASET = PROJ_ROOT / "Cigarette.v1i.yolov8"
#  - drinks (bottle / cup / wine_glass), standard YOLO layout
DRINK_DATASET = PROJ_ROOT / "coco_drink_yolo"

# fine-tune control
MIN_SAMPLES_TO_TRAIN = 60   # threshold to trigger a fine-tune
FINETUNE_WORKDIR = PROJ_ROOT / "runs" / "finetune_queue"
FINETUNE_WEIGHTS_SMOKE = PROJ_ROOT / "smoke_best.pt"   # set your latest weights
FINETUNE_WEIGHTS_DRINK = PROJ_ROOT / "drink_best.pt"   # set your latest weights
