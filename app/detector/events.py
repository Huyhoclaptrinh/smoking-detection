from dataclasses import dataclass
from typing import List

@dataclass
class Event:
    frame_idx: int
    timestamp: float
    timestamp_hms: str
    type: str                 # "smoke" | "drink" | "suspicious" | "unknown"
    obj_name: str | None
    score: float

def events_to_segments(events: List[dict], gap_s: float = 1.0, min_len_s: float = 0.30) -> List[dict]:
    ev = [e for e in events if e.get("type") in ("smoke","drink")]
    ev.sort(key=lambda x: (x["type"], x["timestamp"]))
    segs = []
    cur = None
    for e in ev:
        t  = e["timestamp"]; th = e["timestamp_hms"]
        s  = float(e.get("score",0.0)); typ = e["type"]
        if cur and typ == cur["type"] and (t - cur["end_ts"] <= gap_s):
            cur["end_ts"] = t; cur["end_hms"] = th
            cur["peak_score"] = max(cur["peak_score"], s)
            cur["num_frames"] += 1
        else:
            if cur and (cur["end_ts"] - cur["start_ts"] >= min_len_s):
                segs.append(cur)
            cur = {"type": typ, "start_ts": t, "end_ts": t,
                   "start_hms": th, "end_hms": th,
                   "peak_score": s, "num_frames": 1}
    if cur and (cur["end_ts"] - cur["start_ts"] >= min_len_s):
        segs.append(cur)
    return segs
