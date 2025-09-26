# app/routers/housekeeping.py
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
import mimetypes

router = APIRouter(prefix="/housekeeping", tags=["housekeeping"])
BASE_DIR = Path(__file__).resolve().parents[2]  # gốc project

def safe_join(base: Path, rel: str) -> Path:
    """
    Join base + rel (tương đối). Ngăn path traversal.
    """
    rel = rel.replace("\\", "/")
    cand = (base / rel).resolve()
    if not str(cand).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return cand

def iter_file_range(fp: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    with fp.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_len = min(chunk_size, remaining)
            data = f.read(read_len)
            if not data:
                break
            yield data
            remaining -= len(data)

@router.get("/download")
def download(path: str = Query(...), request: Request = None):
    """
    Tải file trong project theo đường dẫn tương đối (vd: uploads/abc.mp4, outputs/xyz_overlay.mp4).
    Hỗ trợ HTTP Range để <video> có thể tua.
    """
    file_path = safe_join(BASE_DIR, path)
    print(f"[download] BASE_DIR={BASE_DIR}")
    print(f"[download] req path(raw)={path!r}")
    print(f"[download] resolved path 1={file_path}")
    try:
        parent = file_path.parent
        if parent.exists():
            print(f"[download] listdir({parent}):", [p.name for p in parent.iterdir()])
    except Exception as e:
        print("[download] listdir error:", repr(e))

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    file_size = file_path.stat().st_size
    ctype, _ = mimetypes.guess_type(str(file_path))
    if not ctype:
        ctype = "application/octet-stream"

    range_header = request.headers.get("range") if request else None
    if range_header:
        # ví dụ: "bytes=0-"
        unit, _, spec = range_header.partition("=")
        if unit.strip().lower() != "bytes":
            return Response(status_code=416)
        s, _, e = spec.partition("-")
        try:
            start = int(s) if s else 0
            end   = int(e) if e else file_size - 1
        except ValueError:
            return Response(status_code=416)
        start = max(0, start); end = min(file_size - 1, end)
        if start > end:
            return Response(status_code=416)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(
            iter_file_range(file_path, start, end),
            status_code=206,
            media_type=ctype,
            headers=headers,
        )

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    return StreamingResponse(file_path.open("rb"), media_type=ctype, headers=headers)
