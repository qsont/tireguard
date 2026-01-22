import cv2
import numpy as np

def preprocess_bgr(frame_bgr):
    """
    Returns a dict of processed images useful for tread analysis.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Reduce noise, keep groove edges
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Contrast normalize for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(den)

    # Edge map (grooves pop out here)
    edges = cv2.Canny(norm, 60, 140)

    # Optional: morphology to connect groove edges a bit
    kernel = np.ones((3, 3), np.uint8)
    edges2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return {
        "gray": gray,
        "norm": norm,
        "edges": edges,
        "edges2": edges2,
    }

# Backward-compatible alias

def crop_roi(frame, roi):
    x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    return frame[y:y+h, x:x+w]
