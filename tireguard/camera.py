import cv2

def open_camera(preferred_index=None, width=1280, height=720, fps=30):
    """
    Open UVC camera using OpenCV V4L2 backend (Linux).
    Tries indices 0..5 if preferred_index is None.
    Returns (cap, index).
    """
    indices = [preferred_index] if preferred_index is not None else list(range(0, 6))
    last_error = None

    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, idx

        last_error = f"Camera opened but no frames (index={idx})"
        cap.release()

    raise RuntimeError(last_error or "No camera found. Check /dev/video* and permissions.")
