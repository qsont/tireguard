import cv2
import numpy as np

def suggest_roi(frame_bgr, roi_w_ratio=0.45, roi_h_ratio=0.45, stride=16):
    """
    Suggest an ROI by finding the window with maximum edge density.
    - roi_w_ratio/roi_h_ratio: size of ROI relative to frame
    - stride: step size for sliding window (bigger = faster, smaller = more accurate)
    Returns: dict {x,y,w,h}
    """
    h, w = frame_bgr.shape[:2]
    rw = max(60, int(w * roi_w_ratio))
    rh = max(60, int(h * roi_h_ratio))

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 140)

    # integral image for fast window sum
    integ = cv2.integral(edges // 255)  # edges are 0/255 -> make it 0/1

    def win_sum(x0, y0, ww, hh):
        x1, y1 = x0 + ww, y0 + hh
        return (
            integ[y1, x1]
            - integ[y0, x1]
            - integ[y1, x0]
            + integ[y0, x0]
        )

    best = (0, 0, rw, rh)
    best_score = -1

    # scan within bounds
    for y0 in range(0, h - rh, stride):
        for x0 in range(0, w - rw, stride):
            s = win_sum(x0, y0, rw, rh)
            if s > best_score:
                best_score = s
                best = (x0, y0, rw, rh)

    x, y, ww, hh = best
    return {"x": int(x), "y": int(y), "w": int(ww), "h": int(hh)}
