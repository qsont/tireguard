#!/usr/bin/env python3
"""Offline evaluation: compare legacy vs enhanced tread detection on labeled manual-depth data.

Inputs:
- data/manual_readings.csv with columns: ts,manual_depth_mm
- results DB row for each ts (image_path + ROI + vehicle_type)

Outputs:
- terminal summary metrics
- CSV with per-sample predictions for both modes
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tireguard.config import AppConfig
from tireguard.preprocess import crop_roi, preprocess_bgr
from tireguard.measure import apply_defect_guard, pass_fail_from_score, groove_visibility_score
from tireguard.storage import get_result_by_ts


CLASSES = ("GOOD", "WARNING", "REPLACE")


@dataclass
class Sample:
    ts: str
    manual_depth_mm: float


def _normalize_clahe_grid(cfg: AppConfig) -> tuple[int, int]:
    grid = getattr(cfg, "clahe_grid", 8)
    if isinstance(grid, int):
        return (grid, grid)
    if isinstance(grid, (list, tuple)) and len(grid) == 2:
        return (int(grid[0]), int(grid[1]))
    return (8, 8)


def _legal_min_depth_mm(cfg: AppConfig, vehicle_type: str | None) -> float:
    v = (vehicle_type or "car").strip().lower()
    if v == "car":
        return float(getattr(cfg, "car_legal_min_depth_mm", 1.6))
    if v == "motorcycle":
        return float(getattr(cfg, "motorcycle_legal_min_depth_mm", 1.0))
    return float(getattr(cfg, "car_legal_min_depth_mm", 1.6))


def _warning_band_mm(cfg: AppConfig, vehicle_type: str | None) -> float:
    v = (vehicle_type or "car").strip().lower()
    if v == "car":
        return float(getattr(cfg, "car_warning_band_mm", 0.4))
    if v == "motorcycle":
        return float(getattr(cfg, "motorcycle_warning_band_mm", 0.4))
    return float(getattr(cfg, "car_warning_band_mm", 0.4))


def _manual_class(cfg: AppConfig, manual_depth_mm: float, vehicle_type: str | None) -> str:
    min_mm = _legal_min_depth_mm(cfg, vehicle_type)
    warn_band = max(0.0, _warning_band_mm(cfg, vehicle_type))
    if manual_depth_mm < min_mm:
        return "REPLACE"
    if manual_depth_mm < (min_mm + warn_band):
        return "WARNING"
    return "GOOD"


def _defect_guard_kwargs(cfg: AppConfig) -> dict[str, float]:
    return {
        "replace_channel_frac": float(getattr(cfg, "tread_guard_replace_channel_frac", 0.006)),
        "warning_channel_frac": float(getattr(cfg, "tread_guard_warning_channel_frac", 0.018)),
        "replace_score": float(getattr(cfg, "tread_guard_replace_score", 0.035)),
        "warning_score": float(getattr(cfg, "tread_guard_warning_score", 0.065)),
    }


def _load_manual_csv(path: Path) -> list[Sample]:
    out: list[Sample] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = (row.get("ts") or "").strip()
            depth_raw = (row.get("manual_depth_mm") or "").strip()
            if not ts or not depth_raw:
                continue
            try:
                out.append(Sample(ts=ts, manual_depth_mm=float(depth_raw)))
            except ValueError:
                continue
    return out


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else (num / den)


def _binary_replace_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "REPLACE" and p == "REPLACE")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "REPLACE" and p == "REPLACE")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "REPLACE" and p != "REPLACE")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != "REPLACE" and p != "REPLACE")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def _per_class_prf(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        out[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / float(len(y_true))


def _macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    vals = [per_class[c]["f1"] for c in CLASSES]
    return sum(vals) / float(len(vals))


def _iter_predictions(cfg: AppConfig, samples: Iterable[Sample]):
    grid = _normalize_clahe_grid(cfg)
    defect_kwargs = _defect_guard_kwargs(cfg)

    for s in samples:
        db_row = get_result_by_ts(cfg, s.ts)
        if not db_row:
            continue

        image_path = Path(str(db_row.get("image_path") or ""))
        if not image_path.exists():
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        roi = {
            "x": int(db_row["roi_x"]),
            "y": int(db_row["roi_y"]),
            "w": int(db_row["roi_w"]),
            "h": int(db_row["roi_h"]),
        }

        roi_bgr = crop_roi(frame, roi)
        raw_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        processed = preprocess_bgr(
            roi_bgr,
            clahe_clip=float(getattr(cfg, "clahe_clip", 2.0)),
            clahe_grid=grid,
        )
        edges = processed.get("edges_closed", processed.get("edges"))
        if edges is None:
            continue

        mm_per_px = db_row.get("mm_per_px")
        try:
            mm_per_px = float(mm_per_px) if mm_per_px is not None else None
        except Exception:
            mm_per_px = None

        m_legacy = groove_visibility_score(edges, raw_gray=raw_gray, mm_per_px=mm_per_px, mode="legacy")
        m_enhanced = groove_visibility_score(edges, raw_gray=raw_gray, mm_per_px=mm_per_px, mode="enhanced")

        pred_legacy = apply_defect_guard(
            pass_fail_from_score(
                float(m_legacy["score"]),
                groove_channel_frac=m_legacy.get("groove_channel_frac"),
                quality_ok=True,
                **defect_kwargs,
            ),
            groove_channel_frac=m_legacy.get("groove_channel_frac"),
            quality_ok=True,
            score=float(m_legacy["score"]),
            **defect_kwargs,
        )
        pred_enhanced = apply_defect_guard(
            pass_fail_from_score(
                float(m_enhanced["score"]),
                groove_channel_frac=m_enhanced.get("groove_channel_frac"),
                quality_ok=True,
                **defect_kwargs,
            ),
            groove_channel_frac=m_enhanced.get("groove_channel_frac"),
            quality_ok=True,
            score=float(m_enhanced["score"]),
            **defect_kwargs,
        )

        vt = db_row.get("vehicle_type")
        true_class = _manual_class(cfg, s.manual_depth_mm, vt)

        yield {
            "ts": s.ts,
            "vehicle_type": vt or "",
            "manual_depth_mm": s.manual_depth_mm,
            "true_class": true_class,
            "legacy_score": float(m_legacy["score"]),
            "legacy_groove_channel_frac": m_legacy.get("groove_channel_frac"),
            "legacy_pred": pred_legacy,
            "enhanced_score": float(m_enhanced["score"]),
            "enhanced_groove_channel_frac": m_enhanced.get("groove_channel_frac"),
            "enhanced_channel_contrast": m_enhanced.get("channel_contrast"),
            "enhanced_pred": pred_enhanced,
        }


def _print_report(tag: str, y_true: list[str], y_pred: list[str]):
    per_class = _per_class_prf(y_true, y_pred)
    rep = _binary_replace_metrics(y_true, y_pred)
    acc = _accuracy(y_true, y_pred)
    m_f1 = _macro_f1(per_class)

    print(f"\\n== {tag} ==")
    print(f"samples={len(y_true)}")
    print(f"accuracy={acc:.4f} macro_f1={m_f1:.4f}")
    print(
        "replace_precision={:.4f} replace_recall={:.4f} replace_f1={:.4f} replace_specificity={:.4f}".format(
            rep["precision"], rep["recall"], rep["f1"], rep["specificity"]
        )
    )
    print(
        "confusion_replace: TP={} FP={} FN={} TN={}".format(
            int(rep["tp"]), int(rep["fp"]), int(rep["fn"]), int(rep["tn"])
        )
    )
    for cls in CLASSES:
        c = per_class[cls]
        print(f"{cls}: precision={c['precision']:.4f} recall={c['recall']:.4f} f1={c['f1']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate legacy vs enhanced tread detection robustness")
    ap.add_argument("--manual-csv", default="data/manual_readings.csv", help="CSV with ts,manual_depth_mm")
    ap.add_argument("--out-csv", default="data/detection_robustness_eval.csv", help="Per-sample output CSV")
    args = ap.parse_args()

    cfg = AppConfig()
    samples = _load_manual_csv(Path(args.manual_csv))
    if not samples:
        raise SystemExit("No valid manual samples found. Expected columns: ts,manual_depth_mm")

    rows = list(_iter_predictions(cfg, samples))
    if not rows:
        raise SystemExit("No samples were processed. Check DB/image paths and ts values.")

    y_true = [r["true_class"] for r in rows]
    y_legacy = [r["legacy_pred"] for r in rows]
    y_enhanced = [r["enhanced_pred"] for r in rows]

    _print_report("LEGACY", y_true, y_legacy)
    _print_report("ENHANCED", y_true, y_enhanced)

    rep_old = _binary_replace_metrics(y_true, y_legacy)
    rep_new = _binary_replace_metrics(y_true, y_enhanced)
    print("\\n== Delta (ENHANCED - LEGACY) ==")
    print("replace_precision_delta={:+.4f}".format(rep_new["precision"] - rep_old["precision"]))
    print("replace_recall_delta={:+.4f}".format(rep_new["recall"] - rep_old["recall"]))
    print("replace_f1_delta={:+.4f}".format(rep_new["f1"] - rep_old["f1"]))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\\nSaved per-sample report: {out_path}")


if __name__ == "__main__":
    main()
