from __future__ import annotations
import subprocess, os, tempfile
from typing import Optional, Tuple
import whisper
from app.utils.translate import translate_m2m100_en_to_vi

def extract_audio(video_path: str) -> Optional[str]:
    fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return wav_path if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1024 else None

# Safe wrapper for predict
def transcribe_and_translate_safe(wav_path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        w = whisper.load_model("base")
        res = w.transcribe(wav_path, language="en")
        txt_en = res.get("text", "").strip() or None
    except Exception:
        return None, None
    txt_vi = translate_m2m100_en_to_vi(txt_en) if txt_en else None
    return txt_en, txt_vi
