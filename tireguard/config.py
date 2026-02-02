from dataclasses import dataclass
from pathlib import Path

APP_NAME = "TireGuard"

RES_PRESETS = [
    ("640x480", 640, 480),
    ("1280x720", 1280, 720),
    ("1920x1080", 1920, 1080),
]

@dataclass
class AppConfig:
    data_dir: Path = Path("data")
    captures_dir: Path = Path("data/captures")
    processed_dir: Path = Path("data/processed")
    db_path: Path = Path("data/results.db")
    roi_path: Path = Path("data/roi.json")
    export_csv_path: Path = Path("data/results_export.csv")
    calibration_path: Path = Path("data/calibration.json")

    cam_index: int | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30

    # quality thresholds
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    max_glare_ratio: float = 0.06
    min_sharpness: float = 80.0

    # preprocess
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    def __post_init__(self):
        # normalize clahe_grid
        cg = getattr(self, "clahe_grid", (8, 8))
        if isinstance(cg, int):
            self.clahe_grid = (cg, cg)
        elif isinstance(cg, (list, tuple)) and len(cg) == 2:
            self.clahe_grid = (int(cg[0]), int(cg[1]))
        elif isinstance(cg, str):
            # allow "8,8"
            parts = [p.strip() for p in cg.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                self.clahe_grid = (int(parts[0]), int(parts[1]))
            else:
                self.clahe_grid = (8, 8)
        else:
            self.clahe_grid = (8, 8)

def _normalize_clahe_grid(v, default=(8, 8)):
    try:
        if isinstance(v, int):
            return (v, v)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
    except Exception:
        pass
    return default
