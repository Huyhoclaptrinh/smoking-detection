# Default per-class thresholds and Soft-NMS params.
# Adjust to your dataset; keys must match model.names values.

CONF_CFG = {
    "bottle": 0.35,
    "wine glass": 0.30,
    "cup": 0.40,
    "cigarette": 0.25,
    "person": 0.50,
    "_default": 0.35,
}

SOFT_CFG = {
    "bottle":    {"sigma": 0.5, "iou": 0.60, "score": 0.001},
    "wine glass":{"sigma": 0.5, "iou": 0.60, "score": 0.001},
    "cup":       {"sigma": 0.5, "iou": 0.55, "score": 0.001},
    "cigarette": {"sigma": 0.4, "iou": 0.50, "score": 0.001},
    "person":    {"sigma": 0.5, "iou": 0.65, "score": 0.001},
    "_default":  {"sigma": 0.5, "iou": 0.55, "score": 0.001},
}
