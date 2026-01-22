import cv2

def open_camera(preferred_index: int | None = None):
    """
    Opens a UVC/V4L2 camera using OpenCV.

    On Linux, CAP_V4L2 improves reliability.
    """
    indices = [preferred_index] if preferred_index is not None else list(range(0, 6))

    last_err = None
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        # Try a reasonable default. You can change later.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, idx

        last_err = f"Camera opened but no frames (index={idx})"
        cap.release()

    raise RuntimeError(last_err or "No camera found. Check /dev/video* and permissions.")
