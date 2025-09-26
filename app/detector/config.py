from dataclasses import dataclass
from typing import Optional, List, Any, Mapping
import inspect

@dataclass
class DetectionConfig:
    general_model_path: str
    smoke_model_path: str
    smoke_weight_path: Optional[str] = None

    imgsz: int = 640
    conf_thres: float = 0.25
    drink_objs: List[str] | None = None
    frame_skip: int = 3
    output_dir: str = "outputs"

    feedback_dir: str = "feedback"
    min_feedback_for_retrain: int = 30
    rehearsal_per_class: int = 50

    smoke_calib_path: str = "training/calibrators/smoke_calib.pkl"
    drink_calib_path: str = "training/calibrators/drink_calib.pkl"
    calib_log_dir: str = "training/cal_data"
    calib_min_rows: int = 80
    calib_retrain_interval_sec: int = 1800
    unknown_min_conf: float = 0.60
    emit_unknown: bool = False
    calib_autotrain: bool = True
    calib_min_samples: int = 150
    calib_min_pos: int = 30
    calib_min_neg: int = 30

    autoft_enabled: bool = True
    autoft_min_duration_s: int = 5
    autoft_cooldown_hours: float = 0.0
    min_events_per_min_drink: float = 100.0
    min_events_per_min_smoke: float = 100.0
    autoft_force_trigger: bool = True
    min_obj_presence_frames_drink: int = 30
    min_obj_presence_frames_smoke: int = 30
    max_unknown_rate: float = 0.25
    min_feedback_per_branch: int = 10
    autoft_window_frames: int = 300
    emit_unknown_for_empty_frames: bool = True
    # Log ánh sáng theo thời gian thực
    light_diag_enable: bool = True          # bật/tắt log trong lúc detect
    light_diag_stride: int = 15             # mỗi 15 khung log 1 lần (tùy FPS)
    light_diag_center_frac: float = 0.85    # cắt vùng giữa để bỏ viền/letterbox
    light_diag_overlay: bool = True         # có vẽ text lên overlay hay không
    light_diag_log_jsonl: bool = False      # có ghi JSONL hay không
    light_diag_jsonl_path: str = ""         # để trống => auto theo tên video


    @classmethod
    def from_app_config(cls, cfg: Mapping[str, Any]) -> "DetectionConfig":
        """
        Dò tên tham số thật của __init__ rồi map các key từ YAML/ENV vào đúng tên,
        tránh lỗi unexpected keyword (vd: general_model vs general_model_path).
        """
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())  # {'self', 'general_model_path', 'imgsz', ...}

        def has(*names: str) -> str | None:
            for n in names:
                if n in params:
                    return n
            return None

        def val(*keys: str, default=None):
            for k in keys:
                if k in cfg:  # hỗ trợ cả UPPER/lower
                    return cfg[k]
                if k.upper() in cfg:
                    return cfg[k.upper()]
            return default

        kwargs: dict[str, Any] = {}

        # ---- model paths ----
        if (p := has("general_model", "general_model_path", "general_model_file")):
            kwargs[p] = val("GENERAL_MODEL")

        if (p := has("smoke_model", "smoke_model_path")):
            kwargs[p] = val("SMOKE_MODEL")

        if (p := has("smoke_weight_model", "smoke_weight_model_path", "smoke_weights_path", "smoke_weights")):
            kwargs[p] = val("SMOKE_WEIGHT_MODEL")

        if (p := has("drink_model", "drink_model_path")):
            kwargs[p] = val("DRINK_MODEL")

        if (p := has("light_diag_enable")):       kwargs[p] = val("LIGHT_DIAG_ENABLE", default=False)
        if (p := has("light_diag_stride")):       kwargs[p] = val("LIGHT_DIAG_STRIDE", default=15)
        if (p := has("light_diag_center_frac")):  kwargs[p] = val("LIGHT_DIAG_CENTER_FRAC", default=0.85)
        if (p := has("light_diag_overlay")):      kwargs[p] = val("LIGHT_DIAG_OVERLAY", default=True)
        if (p := has("light_diag_log_jsonl")):    kwargs[p] = val("LIGHT_DIAG_LOG_JSONL", default=False)
        if (p := has("light_diag_jsonl_path")):   kwargs[p] = val("LIGHT_DIAG_JSONL_PATH", default="")


        # ---- common runtime ----
        if (p := has("imgsz")):         kwargs[p] = val("IMGSZ", default=640)
        if (p := has("conf_thres")):    kwargs[p] = val("CONF_THRES", default=0.25)
        if (p := has("frame_skip")):    kwargs[p] = val("FRAME_SKIP", default=0)
        if (p := has("output_dir")):    kwargs[p] = val("OUTPUT_DIR", default="outputs")
        if (p := has("names")):         kwargs[p] = val("NAMES", default=["smoke","drink","none"])
        if (p := has("nc")):            kwargs[p] = val("NC", default=3)

        # ---- unknown / calibrator ----
        if (p := has("emit_unknown")):          kwargs[p] = val("EMIT_UNKNOWN", default=False)
        if (p := has("unknown_min_conf")):      kwargs[p] = val("UNKNOWN_MIN_CONF", default=0.60)
        if (p := has("smoke_calib_path")):      kwargs[p] = val("SMOKE_CALIB_PATH")
        if (p := has("drink_calib_path")):      kwargs[p] = val("DRINK_CALIB_PATH")
        if (p := has("calib_log_dir")):         kwargs[p] = val("CALIB_LOG_DIR")
        if (p := has("calib_min_rows")):        kwargs[p] = val("CALIB_MIN_ROWS", default=0)
        if (p := has("calib_retrain_interval_sec")):
            kwargs[p] = val("CALIB_RETRAIN_INTERVAL_SEC", default=0)

        # ---- drink config ----
        if (p := has("drink_objs", "drink_object_names")):
            kwargs[p] = val("DRINK_OBJS", default=["bottle","cup","wine glass"])

        # ---- feedback dirs (nếu __init__ có) ----
        if (p := has("smoke_raw_dir")):   kwargs[p] = val("SMOKE_RAW_DIR")
        if (p := has("smoke_proc_dir")):  kwargs[p] = val("SMOKE_PROC_DIR")
        if (p := has("drink_raw_dir")):   kwargs[p] = val("DRINK_RAW_DIR")
        if (p := has("drink_proc_dir")):  kwargs[p] = val("DRINK_PROC_DIR")

        # Bỏ các key None để không chèn rác
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**kwargs)
