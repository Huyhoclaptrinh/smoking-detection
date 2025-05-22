
# Smoking Detection Project

This project focuses on detecting whether a person is smoking in an image or webcam feed using a YOLOv5 object detection model. It supports real-time detection with OpenCV, automatic logging when smoking is detected, and is inspired by the research at https://github.com/AarnoStormborn/Smoking-Detection.

---

## 🔧 Environment Setup

### 1. Create Conda Environment (Python 3.10)
```bash
conda create -n smoke-detect-env python=3.10 -y
conda activate smoke-detect-env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install YOLOv5 (Ultralytics version)
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

## 📷 Run Real-Time Detection with Logging

Make sure you have your trained model file (e.g. `weights.pt`) and update the path accordingly:

```python
import cv2
import torch
import logging

# Logging configuration
logging.basicConfig(filename='smoking_detections.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/epochs_25/weights.pt')

# Open webcam
cap = cv2.VideoCapture(0)
window_name = 'Smoking Detection'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results.render()[0]
    cv2.imshow(window_name, annotated)

    # Log detections
    for _, row in results.pandas().xyxy[0].iterrows():
        if row['name'].lower() in ['smoking', 'smoke', 'cigarette']:
            logging.info(f"Detected {row['name']} with confidence {row['confidence']:.2f}")

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

This will open your webcam, run detection in real time, and create a log file `smoking_detections.log` whenever smoking is detected.

---

## 📊 Output

- Webcam stream with bounding boxes around smoking actions
- Logged detections in `smoking_detections.log`, e.g.:
  ```
  2025-05-22 09:45:12 - Detected smoking with confidence 0.91
  2025-05-22 09:47:30 - Detected cigarette with confidence 0.86
  ```

---

## 📚 Reference

This project uses data and inspiration from:
> AarnoStormborn. *Smoking Detection Using YOLOv5*. GitHub Repository: https://github.com/AarnoStormborn/Smoking-Detection

Please cite this work if you use the project in academic or commercial settings.

---

## ✅ Additional Notes

- If you encounter errors related to CUDA or PyTorch versioning, consider reinstalling `torch` with your specific CUDA version.
- Compatible with both CPU and GPU setups (GPU preferred for real-time performance).
