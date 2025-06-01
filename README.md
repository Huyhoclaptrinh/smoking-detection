
# Smoking Detection Project

This project focuses on detecting whether a person is smoking in an image or webcam feed using a YOLOv5 object detection model. It supports real-time detection with OpenCV, automatic logging when smoking is detected, and is inspired by the research at https://github.com/AarnoStormborn/Smoking-Detection.

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

### 3. Install YOLOv5 (Ultralytics version)
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Training Summary (Latest Run)

**Command:**
```bash
!cd yolov5 && python train.py --img 640 --batch 8 --epochs 50 --data ../dataset.yaml --weights ../yolov5s.pt --cache disk --workers 0
```

**Results:**
- Images: 383
- Instances: 459
- Precision (P): 0.819
- Recall (R): 0.654
- mAP@0.5: 0.704
- mAP@0.5:0.95: 0.447

These metrics indicate strong performance in detecting smoking behavior, especially with mAP@0.5 over 70%.

---

## Output

- Webcam stream with bounding boxes around smoking actions or behaviors.
- I have a demo video(Sorry i don't smoke so you can't see any scenes of me holding a cigarette haha).
- Logged detections in `detection_log.csv`.

---

## Reference

This project uses data and inspiration from:
> AarnoStormborn. *Smoking Detection Using YOLOv5*. GitHub Repository: https://github.com/AarnoStormborn/Smoking-Detection

Please cite this work if you use the project in academic or commercial settings.

---

## Additional Notes

- If you encounter errors related to CUDA or PyTorch versioning, consider reinstalling `torch` with your specific CUDA version.
- Compatible with both CPU and GPU setups (GPU preferred for real-time performance).
