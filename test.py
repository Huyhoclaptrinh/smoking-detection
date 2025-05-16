import cv2
import requests
import base64
from PIL import Image
from io import BytesIO
import time
import threading

# Global state
current_caption = "Start..."
last_sent_time = 0
FRAME_INTERVAL = 5  # seconds

# Function to encode image to base64
def encode_frame_to_base64(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64

# Function to query llama-server in a thread
def query_llama_async(frame):
    global current_caption
    img_b64 = encode_frame_to_base64(frame)
    try:
        res = requests.post("http://localhost:8080/completion", json={
            "prompt": "What do you see?",
            "images": [img_b64]
        }, timeout=20)
        current_caption = res.json().get("content", "")
    except:
        current_caption = "Error or timeout."

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show caption overlay on frame
    display_frame = frame.copy()
    cv2.putText(display_frame, current_caption, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("SmolVLM Smoking Detection (Smooth)", display_frame)

    # Launch AI request in a separate thread every X seconds
    if time.time() - last_sent_time > FRAME_INTERVAL:
        threading.Thread(target=query_llama_async, args=(frame.copy(),)).start()
        last_sent_time = time.time()

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
