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
from tireguard.measure import apply_defect_guard, estimate_groove_channel_frac


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


def load_pairs_from_validation_results(cfg: AppConfig, include_deleted: bool = False) -> list[tuple[str, float, float]]:
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    deleted_sql = "" if include_deleted else "AND deleted_at IS NULL"
    cur.execute(
        f"""
        SELECT ts, manual_depth, device_score
        FROM validation_results
        WHERE manual_depth IS NOT NULL
          AND device_score IS NOT NULL
          {deleted_sql}
        ORDER BY id ASC
        """
    )
    rows = cur.fetchall()
    con.close()

    out: list[tuple[str, float, float]] = []
    for row in rows:
        try:
            out.append((str(row["ts"]), float(row["manual_depth"]), float(row["device_score"])))
        except Exception:
            continue
    return out


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


def _defect_guard_kwargs(cfg: AppConfig) -> dict[str, float]:
    return {
        "replace_channel_frac": float(getattr(cfg, "tread_guard_replace_channel_frac", 0.006)),
        "warning_channel_frac": float(getattr(cfg, "tread_guard_warning_channel_frac", 0.018)),
        "replace_score": float(getattr(cfg, "tread_guard_replace_score", 0.035)),
        "warning_score": float(getattr(cfg, "tread_guard_warning_score", 0.065)),
    }


def _tread_verdict_from_depth(cfg: AppConfig, depth_mm: float, vehicle_type: str | None) -> str:
    vt = (vehicle_type or "car").strip().lower() or "car"
    if vt == "motorcycle":
        min_mm = float(getattr(cfg, "motorcycle_legal_min_depth_mm", 1.0))
        warn = float(getattr(cfg, "motorcycle_warning_band_mm", 0.4))
    else:
        min_mm = float(getattr(cfg, "car_legal_min_depth_mm", 1.6))
        warn = float(getattr(cfg, "car_warning_band_mm", 0.4))
    if depth_mm < min_mm:
        return "REPLACE"
    if depth_mm < (min_mm + max(0.0, warn)):
        return "WARNING"
    return "GOOD"


def refresh_derived_metrics_from_model(cfg: AppConfig, calib: dict, include_deleted: bool = False) -> tuple[int, int]:
    """Recompute stored depth/raw/tread verdicts after score model updates.

    Also normalizes `verdict` to tread-only policy.
    """
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    deleted_sql = "" if include_deleted else "WHERE deleted_at IS NULL"
    cur.execute(
        f"""
        SELECT id, score, vehicle_type, edge_density, continuity
        FROM results
        {deleted_sql}
        """
    )
    rows = cur.fetchall()

    updated = 0
    skipped = 0
    for row in rows:
        if row["score"] is None:
            skipped += 1
            continue
        try:
            score = float(row["score"])
        except Exception:
            skipped += 1
            continue

        depth_mm = float(score_to_depth_mm(score, calib=calib))
        tread_verdict = _tread_verdict_from_depth(cfg, depth_mm, row["vehicle_type"])
        groove_proxy = estimate_groove_channel_frac(
            score=score,
            edge_density=row["edge_density"],
            continuity=row["continuity"],
        )
        raw_score_verdict = apply_defect_guard(
            tread_verdict,
            groove_channel_frac=groove_proxy,
            quality_ok=True,
            score=score,
            **_defect_guard_kwargs(cfg),
        )

        cur.execute(
            """
            UPDATE results
            SET device_depth_mm=?, raw_score_verdict=?, tread_verdict=?, verdict=?
            WHERE id=?
            """,
            (depth_mm, raw_score_verdict, tread_verdict, tread_verdict, row["id"]),
        )
        updated += 1

    con.commit()
    con.close()
    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit TireGuard score-to-depth model from thesis manual readings.")
    parser.add_argument("--manual-csv", default="data/manual_readings.csv")
    parser.add_argument("--results-csv", default="data/test_results.csv")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "manual-csv", "validation"],
        help="Data source for fitting: auto (manual-csv then validation fallback), manual-csv, or validation",
    )
    parser.add_argument(
        "--include-deleted-validation",
        action="store_true",
        help="Include deleted validation rows when --source validation (or auto fallback).",
    )
    parser.add_argument("--save", action="store_true", help="Persist fitted model into calibration.json")
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Do not auto-refresh stored derived metrics after --save.",
    )
    args = parser.parse_args()

    cfg = AppConfig()
    pairs: list[tuple[str, float, float]] = []
    manual_rows = load_manual_rows(Path(args.manual_csv))
    missing_ts: list[str] = []

    if args.source in ("auto", "manual-csv"):
        score_lookup = load_score_lookup_from_db(cfg)
        for ts, manual_depth in manual_rows:
            score = score_lookup.get(ts)
            if score is None:
                missing_ts.append(ts)
                continue
            pairs.append((ts, manual_depth, score))

        if len(pairs) < 2 and args.source == "auto":
            csv_lookup = load_score_lookup_from_results_csv(Path(args.results_csv))
            for ts in list(missing_ts):
                score = csv_lookup.get(ts)
                manual_depth = next((depth for sample_ts, depth in manual_rows if sample_ts == ts), None)
                if score is None or manual_depth is None:
                    continue
                pairs.append((ts, manual_depth, score))
                missing_ts.remove(ts)

    if (args.source == "validation") or (args.source == "auto" and len(pairs) < 2):
        pairs = load_pairs_from_validation_results(cfg, include_deleted=args.include_deleted_validation)

    print(f"source={args.source} manual_rows={len(manual_rows)} matched_pairs={len(pairs)} missing={len(missing_ts)}")
    for ts, depth, score in pairs:
        print(f"pair ts={ts} manual_mm={depth:.3f} score={score:.6f}")
    if missing_ts and args.source != "validation":
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
        if not args.no_auto_refresh:
            updated, skipped = refresh_derived_metrics_from_model(
                cfg,
                calib,
                include_deleted=args.include_deleted_validation,
            )
            print(f"auto_refresh: updated={updated} skipped={skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
