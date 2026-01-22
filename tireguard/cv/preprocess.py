# cv/preprocess.py
import cv2
import numpy as np

def preprocess(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Reduce noise while preserving edges
    den = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Improve contrast under uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(den)

    # Edge map (for groove visibility / ROI guidance)
    edges = cv2.Canny(norm, 60, 140)

    return {"gray": gray, "norm": norm, "edges": edges}
