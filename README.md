# Smoking/Drinking Behavior Detection – API + UI + MySQL

Phát hiện hành vi **hút thuốc** và **uống nước/đồ uống** trên video bằng YOLO, đóng gói thành **API FastAPI**, có **UI tải video**, và **lưu log thẳng vào MySQL** (không còn CSV nếu bạn chọn chế độ DB-only).

---

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Kiến trúc & luồng xử lý](#kiến-trúc--luồng-xử-lý)
- [Cấu trúc thư mục đề xuất](#cấu-trúc-thư-mục-đề-xuất)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
  - [1. Tạo env](#1-tạo-env)
  - [2. Cài dependencies](#2-cài-dependencies)
  - [3. Chuẩn bị model YOLO](#3-chuẩn-bị-model-yolo)
  - [4. Chuẩn bị MySQL](#4-chuẩn-bị-mysql)
- [Cấu hình](#cấu-hình)
- [Chạy dự án](#chạy-dự-án)
  - [Chạy bằng Python file](#chạy-bằng-python-file)
  - [Chạy trong Jupyter Notebook](#chạy-trong-jupyter-notebook)
- [API Endpoints](#api-endpoints)
- [Giao diện người dùng (UI)](#giao-diện-người-dùng-ui)
- [Tích hợp MySQL](#tích-hợp-mysql)
  - [Auto create table](#auto-create-table)
  - [Query ngược dữ liệu](#query-ngược-dữ-liệu)
- [Troubleshooting](#troubleshooting)
- [Roadmap / Gợi ý phát triển](#roadmap--gợi-ý-phát-triển)
- [License](#license)

---

## Tính năng chính

- **Detect hút thuốc** bằng model YOLO chuyên biệt (cigarette class).
- **Detect uống** bằng YOLO COCO (bottle, cup, wine glass, can, mug...). Heuristic hiện tại đơn giản; có thể nâng cấp logic tay–miệng.
- **API FastAPI**: endpoint để upload video, trả JSON log + link tải video overlay.
- **UI HTML/JS**: form upload, progress bar, bảng event kết quả, link tải file.
- **Lưu log vào MySQL**: mỗi event được ghi trực tiếp vào DB (bỏ qua CSV nếu muốn).
- **Tùy chọn xuất CSV**: vẫn còn code demo nếu cần (có thể bỏ).

---

## Kiến trúc & luồng xử lý

```
Browser/UI  -> POST /predict (multipart/form-data: file)
                  |-- save_temp_upload()  -> lưu video tạm
                  |-- detector.process()  -> YOLO detect + overlay
                  |-- insert events vào MySQL (INSERT IGNORE)
                  |<-- JSON {events[], overlay_video}
Browser/UI  -> GET /download?path=... -> FastAPI trả file overlay
```

**detector.process()**:

1. Mở video với OpenCV, đọc từng frame theo `FRAME_SKIP`.
2. YOLO tổng quát (COCO) tìm đối tượng đồ uống.
3. YOLO smoke tìm thuốc lá.
4. Thêm event (frame, timestamp giây & HH\:MM\:SS, type, obj\_name, score).
5. Vẽ overlay (nếu bật), ghi ra `outputs/<video>_overlay.mp4`.
6. Trả kết quả cho API.

---

## Cấu trúc thư mục đề xuất

```
.
├── config.yaml                 # cấu hình model & tham số
├── requirements.txt
├── human_behavior_detection.ipynb            # FastAPI app (nếu không dùng notebook)
├── static/                     # chứa index.html (UI upload)
├── outputs/                    # video overlay
├── tmp/                        # file upload tạm (Windows friendly)
├── training/                   # folder chứa weights YOLO custom
```

---

## Yêu cầu hệ thống

- Python 3.10+
- (Tuỳ chọn) GPU + CUDA để tăng tốc YOLO.
- MySQL 8+ (hoặc MariaDB) nếu dùng DB (khuyên dùng).

---

## Cài đặt môi trường

### 1. Tạo env

```bash
conda create -n smoke-detect-env python=3.10 -y
conda activate smoke-detect-env
```

### 2. Cài dependencies

`requirements.txt` (đã hợp nhất phần API + ML + DB):

```txt
# Core
numpy>=1.24
opencv-python
pillow
requests
pandas

# Torch + YOLO
ultralytics>=8.2.0
# (Tuỳ GPU) cài torch phù hợp: https://pytorch.org/get-started/locally/
# ví dụ CUDA 11.8:
# torch==2.1.0+cu118
# torchvision==0.16.0+cu118
# torchaudio==2.1.0+cu118
# --extra-index-url https://download.pytorch.org/whl/cu118

# FastAPI
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
pydantic>=2.7.0
pyyaml>=6.0.1
loguru>=0.7.2

# DB
sqlalchemy>=2.0
pymysql>=1.1.0

# Notebook (tuỳ chọn)
notebook
```

Cài:

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị model YOLO

- `GENERAL_MODEL`: ví dụ `yolov8n.pt` (COCO).
- `SMOKE_MODEL`: đường dẫn tới model thuốc lá bạn đã train, ví dụ `training/epochs_50/smoke.pt`.

### 4. Chuẩn bị MySQL

- Tạo DB & user (MySQL Workbench hoặc CLI):

```sql
CREATE DATABASE behavior_detect CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'behav_user'@'%' IDENTIFIED BY 'StrongPass!2025';
GRANT ALL PRIVILEGES ON behavior_detect.* TO 'behav_user'@'%';
FLUSH PRIVILEGES;
```

- Bảng `events` (nếu không dùng auto-create):

```sql
CREATE TABLE events (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    video_name     VARCHAR(255) NOT NULL,
    frame_idx      INT NOT NULL,
    timestamp_sec  DOUBLE NOT NULL,
    timestamp_hms  VARCHAR(32) NOT NULL,
    event_type     ENUM('smoke','drink') NOT NULL,
    obj_name       VARCHAR(64),
    score          FLOAT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_video_frame (video_name, frame_idx, event_type, obj_name),
    KEY idx_video_ts (video_name, timestamp_sec)
);
```

- Hoặc dùng đoạn **auto-create table** trong code (xem mục dưới).

---

## Cấu hình

`config.yaml` ví dụ:

```yaml
GENERAL_MODEL: "yolov8n.pt"
SMOKE_MODEL:   "training/epochs_50/smoke.pt"
IMGSZ: 640
CONF_THRES: 0.25
DRINK_OBJS: ["bottle", "cup", "wine glass", "can", "mug"]
FRAME_SKIP: 3
OUTPUT_DIR: "outputs"
```

Env DB trong code (hoặc `.env`):

```python
DB_USER = "root"
DB_PASS = "1"
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "behavior_detect"
```

---

## Chạy dự án

### Chạy bằng Python file

```bash
export CFG_PATH=config.yaml   # Windows: set CFG_PATH=config.yaml
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Chạy trong Jupyter Notebook

- Gom code thành các cell: Config → Utils → Detector → DB → FastAPI → run uvicorn thread.
- Start server:

```python
import threading, uvicorn

def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if 'server_thread' not in globals():
    server_thread = threading.Thread(target=run_app, daemon=True)
    server_thread.start()
    print("Server running at http://127.0.0.1:8000")
```

---

## API Endpoints

| Method | Path        | Mô tả                                   |
| ------ | ----------- | --------------------------------------- |
| GET    | `/health`   | Kiểm tra tình trạng service             |
| POST   | `/predict`  | Upload video (`multipart/form-data`)    |
| GET    | `/download` | Tải file overlay `?path=outputs/...mp4` |

**Request mẫu (curl):**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@/path/to/video.mp4"
```

**Response mẫu:**

```json
{
  "events": [
    {"frame_idx": 450, "timestamp": 15.0, "timestamp_hms": "0:00:15", "type": "drink", "obj_name": "bottle", "score": 0.82}
  ],
  "overlay_video": "outputs/Drinking_v2_overlay.mp4"
}
```

---

## Giao diện người dùng (UI)

- File `static/index.html` (đã kèm form upload, progress bar, bảng kết quả).
- Mount trong FastAPI:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path

STATIC_DIR = Path.cwd()/"static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

@app.get("/")
def root():
    return RedirectResponse("/ui/index.html")
```

- Truy cập: `http://127.0.0.1:8000/ui/`.

---

## Tích hợp MySQL

### Auto create table

Ngay sau khi tạo `engine`:

```python
from sqlalchemy import text, inspect

SQL_CREATE_EVENTS = text("""
CREATE TABLE IF NOT EXISTS events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL,
    frame_idx INT NOT NULL,
    timestamp_sec DOUBLE NOT NULL,
    timestamp_hms VARCHAR(32) NOT NULL,
    event_type ENUM('smoke','drink') NOT NULL,
    obj_name VARCHAR(64),
    score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_video_frame (video_name, frame_idx, event_type, obj_name),
    KEY idx_video_ts (video_name, timestamp_sec)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")

insp = inspect(engine)
if not insp.has_table("events", schema=DB_NAME):
    with engine.begin() as conn:
        conn.execute(SQL_CREATE_EVENTS)
```

### Query ngược dữ liệu

Ví dụ endpoint:

```python
@app.get("/events")
def get_events(video_name: str):
    sql = text("SELECT * FROM events WHERE video_name=:v ORDER BY frame_idx")
    with engine.connect() as conn:
        rows = conn.execute(sql, {"v": video_name}).mappings().all()
    return rows
```

---

## Troubleshooting

- \`\`\*\* (500)\*\*: dùng block `try/except` để phân loại `SAVE_FILE_ERROR`, `DETECTOR_ERROR`, `DB_ERROR`, `RESPONSE_ERROR`.
- **Port 8000 bị chiếm**: đổi port hoặc restart kernel. Lỗi: `[Errno 10048] only one usage...`.
- **404 /favicon.ico, /**: bình thường; thêm redirect.
- \`\`: tạo thư mục trước khi `app.mount`.
- \*\*Windows & \*\*\`\`: dùng `tmp/` local thay vì `/tmp`.
- **Không JSON được numpy types**: ép `int()`, `float()` khi build rows/response.
- **Lỗi MySQL**: kiểm tra bảng/tên cột/ENUM, user/pass, quyền, `INSERT IGNORE` vs duplicate key.

---

## Roadmap / Gợi ý phát triển

- Hoàn thiện logic uống: sử dụng MediaPipe Hands/Face để kiểm tra tay gần miệng + hướng vật thể.
- Thêm audio: tách âm thanh, ASR (whisper/vosk) và dịch.
- Celery/Redis để xử lý video dài (job queue, status endpoint).
- Dockerize toàn bộ (API + DB), CI/CD.
- Dashboard thống kê hành vi (chart, filter theo thời gian/video) – có thể dùng Streamlit/Grafana.

---

## License

MIT (tuỳ chỉnh theo dự án của bạn).

