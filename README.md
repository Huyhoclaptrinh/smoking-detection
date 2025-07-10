# Combined Drink & Smoking Detection

This project integrates YOLOv5 and MediaPipe to detect alcohol consumption and smoking behavior in video streams. It first detects containers (bottles, glasses, cups, cans) and classifies their contents as beer or wine. If no alcohol is detected, it uses a pinch gesture heuristic and color clustering to identify smoking actions.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Create Conda Environment](#1-create-conda-environment)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Install YOLOv5](#3-install-yolov5)
  - [4. Download Dataset](#4-download-dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

## Features

- **Alcohol Detection**: Detects beer and wine in containers.
- **Soft Drink Detection**: Identifies soft drink cans and bottles.
- **Smoking Detection**: Uses MediaPipe pinch gesture and K-means color clustering to detect smoking.
- **Fallback Logic**: Prioritizes alcohol detection, then smoking if no alcohol is found.

## Prerequisites

- Python 3.10
- Conda (optional but recommended)
- GPU with CUDA (optional, for faster inference)

## Setup

### 1. Create Conda Environment

```bash
conda create -n smoke-detect-env python=3.10 -y
conda activate smoke-detect-env
```

### 2. Install Dependencies

Ensure you have a `requirements.txt` listing the needed packages:

```
torch>=1.12
opencv-python
numpy
mediapipe
```

Install via pip:

```bash
pip install -r requirements.txt
```

### 3. Install YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### 4. Download Dataset

This project uses two datasets:

- **EyeCon (Drinking)**: exported from Roboflow in YOLOv5 format:
  - Universe Roboflow: https://universe.roboflow.com/project-bfxfc/eyecon
  - Place `data.yaml` and `images/` under `data/`.

- **Smoking Detection**: AarnoStormborn’s smoking dataset and implementation:
  - GitHub Repository: https://github.com/AarnoStormborn/Smoking-Detection
  - Clone or download their repository and place the images and labels under `data/smoking/`.

## Project Structure

```text
├── training/
│   ├── epochs_50/
│   │   └── weights_1.pt   # Soft drink model
│   └── epochs_75/
│       └── best.pt        # Behavior model
├── requirements.txt
├── README.md
└── detect.py               # Main detection script
```

## Usage

Run the detection script against a video file:

```bash
python detect.py --video path/to/Drinking_v5.mp4
```

The script will display a window with bounding boxes and print actions to `action_log.txt`.

## Configuration

- **Confidence thresholds**:
  - `yolo_drink.conf`: 0.3
  - `soft_model.conf`: 0.5
- **Smoking distance threshold**: `SMOKE_DIST_THRESH = 0.25`
- These can be adjusted at the top of `detect.py`.

## License

This project is licensed under the MIT License. Feel free to use and adapt for your own research and applications.

