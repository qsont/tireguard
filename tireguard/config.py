from dataclasses import dataclass
from pathlib import Path
import json

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
    settings_path: Path = Path("data/app_settings.json")

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

    # tread depth policy (mm)
    car_legal_min_depth_mm: float = 1.6
    motorcycle_legal_min_depth_mm: float = 1.0
    car_warning_band_mm: float = 0.4
    motorcycle_warning_band_mm: float = 0.4
    # Tread defect guard thresholds.
    # Lower groove-channel fraction means flatter/worn tread.
    # REPLACE is stricter; WARNING remains broader to avoid over-replacing borderline wear.
    tread_guard_replace_channel_frac: float = 0.006
    tread_guard_warning_channel_frac: float = 0.018
    # Fallback score thresholds when channel fraction is unavailable.
    tread_guard_replace_score: float = 0.035
    tread_guard_warning_score: float = 0.065
    # Automatic behavior switches.
    auto_detect_tread_on_roi: bool = True
    auto_calibrate_on_roi: bool = True
    # Assumed real-world width represented by ROI span for automatic 2-point calibration.
    auto_calibration_reference_mm: float = 120.0
    auto_calibration_reference_mm_car: float = 120.0
    auto_calibration_reference_mm_motorcycle: float = 85.0
    # Quick-session mode for hands-free scan/capture workflows.
    quick_session_auto_capture: bool = False
    quick_session_capture_cooldown_s: float = 1.8
    # Relax blur-only warnings into advisories when tread evidence is strong.
    quality_relaxed_min_sharpness: float = 24.0
    quality_relaxed_strong_score: float = 0.11
    quality_relaxed_channel_frac: float = 0.020
    quality_relaxed_tread_confidence: float = 0.55
    # Validation tolerances for thesis/device comparison.
    validation_max_percent_diff: float = 10.0
    validation_max_abs_error_mm: float = 0.5
    recycle_retention_days: int = 30

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

        # Load persisted runtime settings (if present).
        self.load_runtime_settings()

    def load_runtime_settings(self):
        """Load persisted runtime settings from settings_path if available."""
        p = Path(self.settings_path)
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        for key in (
            "car_legal_min_depth_mm",
            "motorcycle_legal_min_depth_mm",
            "car_warning_band_mm",
            "motorcycle_warning_band_mm",
            "tread_guard_replace_channel_frac",
            "tread_guard_warning_channel_frac",
            "tread_guard_replace_score",
            "tread_guard_warning_score",
            "auto_detect_tread_on_roi",
            "auto_calibrate_on_roi",
            "auto_calibration_reference_mm",
            "auto_calibration_reference_mm_car",
            "auto_calibration_reference_mm_motorcycle",
            "quick_session_auto_capture",
            "quick_session_capture_cooldown_s",
            "quality_relaxed_min_sharpness",
            "quality_relaxed_strong_score",
            "quality_relaxed_channel_frac",
            "quality_relaxed_tread_confidence",
            "validation_max_percent_diff",
            "validation_max_abs_error_mm",
            "recycle_retention_days",
        ):
            if key in payload:
                try:
                    if key == "recycle_retention_days":
                        setattr(self, key, int(payload[key]))
                    elif key in ("auto_detect_tread_on_roi", "auto_calibrate_on_roi", "quick_session_auto_capture"):
                        setattr(self, key, bool(payload[key]))
                    else:
                        setattr(self, key, float(payload[key]))
                except Exception:
                    continue

    def save_runtime_settings(self):
        """Persist runtime settings used by UI/API depth policy logic."""
        p = Path(self.settings_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "car_legal_min_depth_mm": float(getattr(self, "car_legal_min_depth_mm", 1.6)),
            "motorcycle_legal_min_depth_mm": float(getattr(self, "motorcycle_legal_min_depth_mm", 1.0)),
            "car_warning_band_mm": float(getattr(self, "car_warning_band_mm", 0.4)),
            "motorcycle_warning_band_mm": float(getattr(self, "motorcycle_warning_band_mm", 0.4)),
            "tread_guard_replace_channel_frac": float(getattr(self, "tread_guard_replace_channel_frac", 0.006)),
            "tread_guard_warning_channel_frac": float(getattr(self, "tread_guard_warning_channel_frac", 0.018)),
            "tread_guard_replace_score": float(getattr(self, "tread_guard_replace_score", 0.035)),
            "tread_guard_warning_score": float(getattr(self, "tread_guard_warning_score", 0.065)),
            "auto_detect_tread_on_roi": bool(getattr(self, "auto_detect_tread_on_roi", True)),
            "auto_calibrate_on_roi": bool(getattr(self, "auto_calibrate_on_roi", True)),
            "auto_calibration_reference_mm": float(getattr(self, "auto_calibration_reference_mm", 120.0)),
            "auto_calibration_reference_mm_car": float(getattr(self, "auto_calibration_reference_mm_car", 120.0)),
            "auto_calibration_reference_mm_motorcycle": float(getattr(self, "auto_calibration_reference_mm_motorcycle", 85.0)),
            "quick_session_auto_capture": bool(getattr(self, "quick_session_auto_capture", False)),
            "quick_session_capture_cooldown_s": float(getattr(self, "quick_session_capture_cooldown_s", 1.8)),
            "quality_relaxed_min_sharpness": float(getattr(self, "quality_relaxed_min_sharpness", 24.0)),
            "quality_relaxed_strong_score": float(getattr(self, "quality_relaxed_strong_score", 0.11)),
            "quality_relaxed_channel_frac": float(getattr(self, "quality_relaxed_channel_frac", 0.020)),
            "quality_relaxed_tread_confidence": float(getattr(self, "quality_relaxed_tread_confidence", 0.55)),
            "validation_max_percent_diff": float(getattr(self, "validation_max_percent_diff", 10.0)),
            "validation_max_abs_error_mm": float(getattr(self, "validation_max_abs_error_mm", 0.5)),
            "recycle_retention_days": int(getattr(self, "recycle_retention_days", 30)),
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _normalize_clahe_grid(v, default=(8, 8)):
    try:
        if isinstance(v, int):
            return (v, v)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
    except Exception:
        pass
    return default
