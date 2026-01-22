from dataclasses import dataclass
from pathlib import Path

APP_NAME = "TireGuard"

@dataclass
class AppConfig:
    data_dir: Path = Path("data")
    captures_dir: Path = Path("data/captures")
    processed_dir: Path = Path("data/processed")
    db_path: Path = Path("data/results.db")
    roi_path: Path = Path("data/roi.json")
    export_csv_path: Path = Path("data/results_export.csv")

    cam_index: int | None = None
    width: int = 1280
    height: int = 720
    fps: int = 30

    min_brightness: float = 40.0
    max_brightness: float = 220.0
    max_glare_ratio: float = 0.06
    min_sharpness: float = 80.0

    clahe_clip: float = 2.0
    clahe_grid: int = 8
