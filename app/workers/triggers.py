from pathlib import Path
import json, time
from app.logging import get_logger
logger = get_logger()

TRIGGER_DIR = Path("training/triggers")
TRIGGER_DIR.mkdir(parents=True, exist_ok=True)

def try_trigger_finetune(part_hint: str, reason: str = "feedback") -> bool:
    if part_hint not in ("smoke","drink"):
        logger.info(f"[autoft] skip enqueue part={part_hint}")
        return False
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = TRIGGER_DIR / f"{stamp}_{part_hint}_{reason}.json"
    path.write_text(json.dumps({"part": part_hint, "reason": reason, "time": stamp}, ensure_ascii=False))
    logger.info(f"[autoft] enqueued: {path}")
    return True
