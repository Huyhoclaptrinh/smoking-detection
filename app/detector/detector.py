# app/detector/detector.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any

# Pydantic v2
from pydantic import BaseModel, ConfigDict, field_validator


class DetectionConfig(BaseModel):
    """
    Cấu hình detector. Định nghĩa các field theo đúng những gì main.py đang truyền:
    - general_model
    - smoke_model_path / smoke_model
    - drink_model_path / drink_model
    - imgsz
    - conf_thres
    - frame_skip
    - output_dir
    - smoke_objs / drink_objs
    - unknown_min_conf
    - emit_unknown
    """
    # Cho phép nhận thêm key lạ mà không bị lỗi (tránh TypeError Unexpected keyword)
    model_config = ConfigDict(extra="ignore")

    # Các đường dẫn model
    general_model: Optional[str] = None
    smoke_model_path: Optional[str] = None
    smoke_model: Optional[str] = None  # alias tương thích
    drink_model_path: Optional[str] = None
    drink_model: Optional[str] = None  # alias tương thích

    # Tham số chung
    imgsz: int = 640
    conf_thres: float = 0.25
    frame_skip: int = 1
    output_dir: str = "outputs"

    # Danh sách object
    smoke_objs: List[str] = ["cigarette"]
    drink_objs: List[str] = ["bottle", "cup", "wine glass"]

    # Unknown handling
    unknown_min_conf: float = 0.6
    emit_unknown: bool = False

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    # Các helper để thống nhất tên (dùng cái nào có giá trị)
    @property
    def smoke_weight(self) -> Optional[str]:
        return self.smoke_model_path or self.smoke_model

    @property
    def drink_weight(self) -> Optional[str]:
        return self.drink_model_path or self.drink_model


class BehaviorDetector:
    """
    Lớp detector đơn giản hoá. Bạn có thể map sang pipeline thật của bạn ở đây.
    Hiện tại mình giữ stub tối thiểu để hệ thống chạy được, tránh lỗi startup.
    Sau đó, bạn thay phần _run_inference(...) bằng code detect thật.
    """
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        # TODO: load các model nếu cần, ví dụ:
        # if self.cfg.smoke_weight: ... load
        # if self.cfg.drink_weight: ... load

    def process(self, video_path: str, save_overlay: bool = True) -> Dict[str, Any]:
        """
        Nhận video_path -> trả về dict chứa:
          - events: list các event (frame_idx, timestamp, timestamp_hms, type, obj_name, score)
          - overlay_video: đường dẫn video có vẽ overlay (nếu tạo)
        TẠM THỜI trả về khung rỗng để backend và UI hoạt động.
        Bạn thay thế phần TODO phía dưới bằng code suy luận thực tế của bạn.
        """
        video_path = str(video_path)

        # TODO: chạy suy luận thật tại đây và dựng 'events' + 'overlay_video'
        result = {
            "events": [],
            "overlay_video": None,
        }

        # Tối thiểu trả về tên file cho UI và /download
        result["video_name"] = Path(video_path).stem
        return result
