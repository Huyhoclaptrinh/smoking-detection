
# Smoking Detection Project

This project focuses on detecting whether a person is smoking in an image or video frame. The system uses YOLOv8 and OpenCV for object detection, and is designed to later support realtime detection and server deployment.

---

## Environment Setup

### 1. Create Conda Environment (Python 3.10)

```bash
conda create -n smoke-detect-env python=3.10 -y
conda activate smoke-detect-env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure you have a compatible GPU (e.g., NVIDIA 1650) with CUDA 11.8. If you don’t use GPU, modify `requirements.txt` accordingly.

### 3. Verify Installation

Test OpenCV:
```python
import cv2
print(cv2.__version__)
```

Test YOLOv8:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)
```

You should see a prediction image with bounding boxes saved locally.

---

## 📦 Dependencies

Main libraries (from `requirements.txt`):
- `opencv-python`
- `torch`, `torchvision`, `torchaudio` (CUDA 11.8)
- `ultralytics` (YOLOv8)
- `notebook`, `numpy`, `matplotlib`

---

## 💻 Optional: Jupyter Notebook

If you prefer running inside Jupyter:

```bash
jupyter notebook
```

Then open your `.ipynb` file and test modules in cells.

---

## 🧪 Coming Features

- Frame capture every 3s from video
- Detect smoking behavior per image
- Realtime 1–2s clip detection (next phase)
- Server deployment with cronjob
- Git-based collaboration (Git flow and coding rules to be defined)

---

## ✍️ Notes

- Make sure to report progress daily in the team group.
- Push working code to GitHub regularly.
- Follow naming conventions and coding guidelines (to be provided separately).
