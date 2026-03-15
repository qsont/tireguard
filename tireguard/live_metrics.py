import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class LiveMetrics:
    brightness: float
    glare_ratio: float
    sharpness: float
    stability: float  # lower = more stable
    ok_hint: bool

def compute_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))

def compute_glare_ratio(gray: np.ndarray, threshold: int = 245) -> float:
    # fraction of pixels near-white
    return float(np.mean(gray >= threshold))

def compute_sharpness(gray: np.ndarray) -> float:
    # variance of Laplacian
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_stability(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    """
    Rough scene stability: mean absolute diff between frames (0..255).
    Lower = more stable.
    """
    if prev_gray is None or prev_gray.shape != gray.shape:
        return 999.0
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))

def compute_live_metrics(
    roi_bgr: np.ndarray,
    prev_gray: np.ndarray | None,
    min_brightness: float,
    max_brightness: float,
    min_sharpness: float,
    max_glare: float = 0.08,
    stable_thresh: float = 3.0
) -> tuple[LiveMetrics, np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    b = compute_brightness(gray)
    g = compute_glare_ratio(gray)
    s = compute_sharpness(gray)
    st = compute_stability(prev_gray, gray)

    ok = (min_brightness <= b <= max_brightness) and (s >= min_sharpness) and (g <= max_glare) and (st <= stable_thresh)
    return LiveMetrics(brightness=b, glare_ratio=g, sharpness=s, stability=st, ok_hint=ok), gray
