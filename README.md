
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

## 📊 Output

- Webcam stream with bounding boxes around smoking actions
- Logged detections in `smoking_detections.log`, e.g.:
  ```
  2025-05-22 09:45:12 - Detected cigarette with confidence 0.91
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
