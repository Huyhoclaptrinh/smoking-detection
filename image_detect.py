import os
import cv2
import csv
from ultralytics import YOLO

# Configuration
FRAME_DIR = "frames"
LOG_PATH = "detection_log.csv"
MODEL_PATH = "yolov8n.pt"      # or your custom model
CONF_THRESHOLD = 0.4
TARGET_CLASS = "person"

# Load model
model = YOLO(MODEL_PATH)

# Prepare log file
with open(LOG_PATH, mode='w', newline='') as log_file:
    writer = csv.writer(log_file)
    writer.writerow(["Frame", "Num_of_Detections", "Boxes", "Confidences"])

    for filename in sorted(os.listdir(FRAME_DIR)):
        if filename.endswith(".jpg"):
            path = os.path.join(FRAME_DIR, filename)
            frame = cv2.imread(path)

            results = model(frame)[0]
            detections = results.boxes

            boxes = []
            confidences = []
            count = 0

            for box in detections:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                conf = float(box.conf.item())

                if class_name == TARGET_CLASS and conf >= CONF_THRESHOLD:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append(f"{x1},{y1},{x2},{y2}")
                    confidences.append(f"{conf:.2f}")

            writer.writerow([filename, count, "; ".join(boxes), "; ".join(confidences)])
            print(f"[{filename}] {count} {TARGET_CLASS}(s) detected.")
