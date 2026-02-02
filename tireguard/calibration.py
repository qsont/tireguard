import json
from pathlib import Path
import math

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
