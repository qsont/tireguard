# camera/capture.py
from picamera2 import Picamera2
import cv2
import time
from pathlib import Path

class Camera:
    def __init__(self, width=1280, height=720):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)

    def get_frame(self):
        frame = self.picam2.capture_array()  # RGB
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def save_frame(self, frame_bgr, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame_bgr)
