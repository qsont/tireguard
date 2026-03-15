import json
from pathlib import Path
import math


def has_score_model(calib: dict | None) -> bool:
    """Return True when calibration includes a usable score->depth model."""
    if not isinstance(calib, dict):
        return False
    model = calib.get("score_model")
    if not isinstance(model, dict):
        return False
    mtype = str(model.get("type", "")).strip().lower()
    if mtype == "linear":
        return "slope" in model and "intercept" in model
    if mtype == "poly":
        coeffs = model.get("coeffs")
        return isinstance(coeffs, list) and len(coeffs) > 0
    return False

def load_calibration(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {
        "px_per_mm": None,
        "mm_per_px": None,
        "method": None
    }

def save_calibration(path: Path, calib: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))

def dist_px(p0, p1):
    return math.hypot(p1[0]-p0[0], p1[1]-p0[1])

def compute_scale_from_two_points(p0, p1, known_mm: float):
    dpx = dist_px(p0, p1)
    if known_mm <= 0 or dpx <= 0:
        raise ValueError("Invalid scale inputs")
    px_per_mm = dpx / known_mm
    mm_per_px = known_mm / dpx
    return px_per_mm, mm_per_px

def score_to_depth_mm(
    score: float,
    calib: dict | None = None,
    fallback_slope: float = -6.0,
    fallback_intercept: float = 6.0,
) -> float:
    """
    Convert TireGuard score -> depth(mm) using model stored in calibration.json.

    Supported model formats in calib:
      {
        "score_model": {
          "type": "linear",
          "slope": -6.0,
          "intercept": 6.0
        }
      }

      {
        "score_model": {
          "type": "poly",
          "coeffs": [a_n, ..., a_1, a_0]  # Horner evaluation
        }
      }
    """
    c = calib or {}
    model = c.get("score_model") or {}
    s = float(score)

    mtype = str(model.get("type", "linear")).lower()

    if mtype == "poly":
        coeffs = model.get("coeffs") or []
        if coeffs:
            y = 0.0
            for a in coeffs:
                y = y * s + float(a)
            return max(0.0, float(y))

    slope = float(model.get("slope", fallback_slope))
    intercept = float(model.get("intercept", fallback_intercept))
    return max(0.0, slope * s + intercept)
