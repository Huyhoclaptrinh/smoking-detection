from ultralytics import YOLO
import os
import cv2
from pathlib import Path

# Load YOLO model
model = YOLO("yolov8n.pt")

# Folder paths
DATASET_DIR = "dataset_test/images"
LABELS_DIR = "dataset_test/labels"
LOG_FILE = "person_label_log.txt"
splits = ["train", "val"]

log_lines = []

for split in splits:
    input_folder = os.path.join(DATASET_DIR, split)
    label_output_folder = os.path.join(LABELS_DIR, split)
    os.makedirs(label_output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in image_files:
        image_path = os.path.join(input_folder, img_name)

        # Check image
        img = cv2.imread(image_path)
        if img is None:
            line = f"⚠️ Không đọc được ảnh: {img_name}, bỏ qua."
            print(line)
            log_lines.append(line)
            continue

        results = model.predict(source=img, conf=0.4)[0]
        label_lines = []
        height, width = results.orig_shape

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            label_file = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_output_folder, label_file)
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
            line = f"[{split}] ✅ Đã tạo label cho: {img_name} → {label_file}"
        else:
            line = f"[{split}] ❌ Không phát hiện person trong: {img_name} → Không tạo label"

        print(line)
        log_lines.append(line)

# Ghi file log
with open(LOG_FILE, "w", encoding="utf-8") as logf:
    logf.write("\n".join(log_lines))

print(f"\n📄 Log đã được lưu vào: {LOG_FILE}")
