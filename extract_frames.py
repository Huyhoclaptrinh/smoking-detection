import cv2
import os

# Configuration
VIDEO_PATH = "input_videos/smoking1.mp4"      # Path to the input video
FRAME_OUTPUT_DIR = "frames"                 # Directory to save extracted frames
INTERVAL_SEC = 2                            # Extract 1 frame every 2 seconds

# Create output directory if it doesn't exist
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * INTERVAL_SEC)

frame_id = 0
saved_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        frame_name = f"frame_{saved_id}.jpg"
        frame_path = os.path.join(FRAME_OUTPUT_DIR, frame_name)
        cv2.imwrite(frame_path, frame)
        print(f"Saved {frame_name}")
        saved_id += 1

    frame_id += 1

cap.release()
print("Frame extraction complete.")
