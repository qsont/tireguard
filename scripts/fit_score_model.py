#!/usr/bin/env python3
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tireguard.calibration import has_score_model, load_calibration, save_calibration
from tireguard.config import AppConfig


def load_manual_rows(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = (row.get("ts") or "").strip()
            depth = (row.get("manual_depth_mm") or "").strip()
            if not ts or not depth:
                continue
            rows.append((ts, float(depth)))
    return rows


def load_score_lookup_from_db(cfg: AppConfig) -> dict[str, float]:
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT ts, score FROM results")
    lookup = {str(row["ts"]): float(row["score"]) for row in cur.fetchall() if row["ts"] is not None and row["score"] is not None}
    con.close()
    return lookup


def load_score_lookup_from_results_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    lookup: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = (row.get("ts") or "").strip()
            score = (row.get("score") or "").strip()
            if not ts or not score:
                continue
            lookup[ts] = float(score)
    return lookup


def fit_linear_model(pairs: list[tuple[str, float, float]]) -> dict:
    scores = np.array([score for _, _, score in pairs], dtype=float)
    depths = np.array([depth for _, depth, _ in pairs], dtype=float)
    slope, intercept = np.polyfit(scores, depths, 1)
    predicted = slope * scores + intercept
    residual = depths - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((depths - depths.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    mae = float(np.mean(np.abs(residual)))
    return {
        "type": "linear",
        "slope": float(slope),
        "intercept": float(intercept),
        "sample_count": len(pairs),
        "r2": float(r2),
        "mae_mm": float(mae),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit TireGuard score-to-depth model from thesis manual readings.")
    parser.add_argument("--manual-csv", default="data/manual_readings.csv")
    parser.add_argument("--results-csv", default="data/test_results.csv")
    parser.add_argument("--save", action="store_true", help="Persist fitted model into calibration.json")
    args = parser.parse_args()

    cfg = AppConfig()
    manual_rows = load_manual_rows(Path(args.manual_csv))
    score_lookup = load_score_lookup_from_db(cfg)

    missing_ts: list[str] = []
    pairs: list[tuple[str, float, float]] = []
    for ts, manual_depth in manual_rows:
        score = score_lookup.get(ts)
        if score is None:
            missing_ts.append(ts)
            continue
        pairs.append((ts, manual_depth, score))

    if len(pairs) < 2:
        csv_lookup = load_score_lookup_from_results_csv(Path(args.results_csv))
        for ts in list(missing_ts):
            score = csv_lookup.get(ts)
            manual_depth = next((depth for sample_ts, depth in manual_rows if sample_ts == ts), None)
            if score is None or manual_depth is None:
                continue
            pairs.append((ts, manual_depth, score))
            missing_ts.remove(ts)

    print(f"manual_rows={len(manual_rows)} matched_pairs={len(pairs)} missing={len(missing_ts)}")
    for ts, depth, score in pairs:
        print(f"pair ts={ts} manual_mm={depth:.3f} score={score:.6f}")
    if missing_ts:
        print("missing_ts=")
        for ts in missing_ts:
            print(f"  - {ts}")

    if len(pairs) < 2:
        print("Insufficient matched manual/score samples to fit a thesis-grade linear score model. Need at least 2 matched points, preferably more.")
        return 2

    model = fit_linear_model(pairs)
    print(json.dumps(model, indent=2))

    if args.save:
        calib = load_calibration(cfg.calibration_path)
        calib["score_model"] = {
            "type": "linear",
            "slope": model["slope"],
            "intercept": model["intercept"],
        }
        save_calibration(cfg.calibration_path, calib)
        print(f"Saved score model to {cfg.calibration_path}")
        print(f"has_score_model={has_score_model(calib)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
