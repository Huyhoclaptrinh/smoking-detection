from __future__ import annotations
from pathlib import Path
import json, time
from app.logging import get_logger

logger = get_logger()
_LAST = Path("training/.last_autoft.json")

def _load_last_ts() -> float:
    try:
        return json.loads(_LAST.read_text()).get("ts", 0.0) if _LAST.exists() else 0.0
    except Exception:
        return 0.0

def try_save_last_autoft_ts(ts: float):
    _LAST.parent.mkdir(parents=True, exist_ok=True)
    _LAST.write_text(json.dumps({"ts": ts}, ensure_ascii=False))

def _unique_frames_with(events, typ_prefix: str) -> int:
    frames = {e.frame_idx for e in events if getattr(e, "type", "").startswith(typ_prefix)}
    return len(frames)

def _enough_feedback(root_dir: str, min_n: int) -> bool:
    p = Path(root_dir) / "images"
    tot = sum(1 for x in p.glob("*") if x.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"})
    return tot >= min_n

def evaluate_autoft_from_events(events, fps: float, total_frames: int, cfg, try_trigger_finetune) -> bool:
    if not getattr(cfg, "autoft_enabled", True): return False
    duration_s = total_frames / max(1e-6, fps)
    if duration_s < getattr(cfg, "autoft_min_duration_s", 20): return False

    now = time.time()
    last_ts = _load_last_ts()
    cooldown_ok = (now - last_ts) >= getattr(cfg, "autoft_cooldown_hours", 6.0) * 3600

    mins = max(1e-6, duration_s / 60.0)
    n_drink = sum(1 for e in events if getattr(e, "type", "") == "drink")
    n_smoke = sum(1 for e in events if getattr(e, "type", "") == "smoke")
    drink_rate = n_drink / mins
    smoke_rate = n_smoke / mins

    unk_frames = _unique_frames_with(events, "unknown")
    unknown_rate = unk_frames / max(1, total_frames)

    min_ev_drink = getattr(cfg, "min_events_per_min_drink", 0.5)
    min_ev_smoke = getattr(cfg, "min_events_per_min_smoke", 0.5)
    max_unk_rate = getattr(cfg, "max_unknown_rate", 0.25)
    min_events_abs = getattr(cfg, "min_events_abs", 1)
    force = getattr(cfg, "autoft_force_trigger", False)

    need_drink = (n_drink < min_events_abs) or (drink_rate < min_ev_drink) or (unknown_rate > max_unk_rate)
    need_smoke = (n_smoke < min_events_abs) or (smoke_rate < min_ev_smoke) or (unknown_rate > max_unk_rate)

    # Read feedback dirs from common defaults (can override in your config/usage)
    DRINK_RAW_DIR = "feedback_raw/drink"
    SMOKE_RAW_DIR = "feedback_raw/smoke"
    fb_min = getattr(cfg, "min_feedback_per_branch", 10)
    fb_drink_ok = _enough_feedback(DRINK_RAW_DIR, fb_min)
    fb_smoke_ok = _enough_feedback(SMOKE_RAW_DIR, fb_min)

    logger.info(f"[autoft] stats | dur={duration_s:.2f}s, drink={n_drink} ({drink_rate:.2f}/min), "
                f"smoke={n_smoke} ({smoke_rate:.2f}/min), unknown_rate={unknown_rate:.2%}")
    logger.info(f"[autoft] gates | drink_need={need_drink}, smoke_need={need_smoke}, "
                f"fb_drink_ok={fb_drink_ok}, fb_smoke_ok={fb_smoke_ok}, cooldown_ok={cooldown_ok}, force={force}")

    triggered = False
    if cooldown_ok and need_drink and (fb_drink_ok or force):
        triggered |= bool(try_trigger_finetune("drink", reason="low_detection_or_high_unknown"))
    if cooldown_ok and need_smoke and (fb_smoke_ok or force):
        triggered |= bool(try_trigger_finetune("smoke", reason="low_detection_or_high_unknown"))
    return triggered
