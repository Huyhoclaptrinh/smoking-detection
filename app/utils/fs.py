from __future__ import annotations
from pathlib import Path
import os, stat, errno, cv2

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_temp_upload(upload_file, tmp_dir="tmp") -> str:
    ensure_dir(tmp_dir)
    dest = Path(tmp_dir) / upload_file.filename
    with open(dest, "wb") as f:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)
    return str(dest)

def video_writer(out_path: str, fps: float, w: int, h: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))

def handle_remove_readonly(func, path, exc_info):
    excvalue = exc_info[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        func(path)
    else:
        raise

# Download guard
def _is_in_dir(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False

ALLOWED_ROOTS = [Path("outputs").resolve(), Path("tmp").resolve()]
def is_allowed_download(path: str) -> bool:
    p = Path(path).resolve()
    return any(_is_in_dir(p, root) for root in ALLOWED_ROOTS)
