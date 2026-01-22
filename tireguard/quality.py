import cv2
import numpy as np

def brightness_mean(gray):
    return float(np.mean(gray))

def glare_ratio(gray, bright_threshold=240):
    bright = np.sum(gray >= bright_threshold)
    return float(bright) / float(gray.size)

def sharpness_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def run_quality_checks(gray, cfg):
    """
    Returns dict:
      ok: bool
      reasons: list[str]
      metrics: dict
    """
    b = brightness_mean(gray)
    g = glare_ratio(gray)
    s = sharpness_laplacian(gray)

    reasons = []
    if b < cfg.min_brightness:
        reasons.append(f"Too dark (brightness={b:.1f})")
    if b > cfg.max_brightness:
        reasons.append(f"Too bright (brightness={b:.1f})")
    if g > cfg.max_glare_ratio:
        reasons.append(f"Too much glare (glare_ratio={g:.3f})")
    if s < cfg.min_sharpness:
        reasons.append(f"Too blurry (sharpness={s:.1f})")

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {"brightness": b, "glare_ratio": g, "sharpness": s},
    }
