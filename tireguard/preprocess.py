import cv2
import numpy as np

def preprocess_bgr(frame_bgr, clahe_clip=2.0, clahe_grid=(8, 8), blur_ksize=3, do_edges=True):
    """
    Preprocess input BGR frame.

    Steps:
      - Convert BGR → Gray
      - CLAHE (contrast-limited adaptive histogram equalization)
      - Optional Gaussian blur
      - Optional edge detection

    Returns:
      dict with keys:
        - "gray": normalized grayscale image
        - "edges": edge image or None
    """
    if frame_bgr is None:
        return None

    # --- normalize clahe_grid to tuple(int,int) ---
    if isinstance(clahe_grid, int):
        clahe_grid = (clahe_grid, clahe_grid)
    elif isinstance(clahe_grid, (list, tuple)) and len(clahe_grid) == 2:
        clahe_grid = (int(clahe_grid[0]), int(clahe_grid[1]))
    else:
        # fallback safe default
        clahe_grid = (8, 8)

    # --- BGR -> Gray ---
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # --- CLAHE ---
    try:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
        gray = clahe.apply(gray)
    except Exception:
        # CLAHE fails? continue with raw gray
        pass

    # --- optional blur ---
    if blur_ksize and int(blur_ksize) > 1:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    out = {"gray": gray}

    # --- edges ---
    if do_edges:
        edges = cv2.Canny(gray, 50, 150)
        out["edges"] = edges

        # --- edges_closed (robust) ---
        try:
            kernel = np.ones((3, 3), np.uint8)
            out["edges_closed"] = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            # if closing fails, just omit edges_closed
            pass
    else:
        # still provide "edges" key for UI fallback consistency
        out["edges"] = np.zeros_like(gray)

    return out

def crop_roi(frame_bgr, roi: dict):
    """
    Crop ROI from a BGR frame.
    ROI dict: {"x": int, "y": int, "w": int, "h": int}
    Always clamps bounds to frame size.
    """
    if frame_bgr is None:
        return None
    if not roi:
        return frame_bgr

    fh, fw = frame_bgr.shape[:2]
    x = int(roi.get("x", 0))
    y = int(roi.get("y", 0))
    w = int(roi.get("w", fw))
    h = int(roi.get("h", fh))

    # clamp
    x = max(0, min(fw - 1, x))
    y = max(0, min(fh - 1, y))
    w = max(1, min(fw - x, w))
    h = max(1, min(fh - y, h))

    return frame_bgr[y:y + h, x:x + w].copy()

__all__ = ["preprocess_bgr", "crop_roi"]
