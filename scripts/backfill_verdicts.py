#!/usr/bin/env python3
import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tireguard.calibration import has_score_model, load_calibration, score_to_depth_mm
from tireguard.config import AppConfig
from tireguard.measure import pass_fail_from_score


def normalize_vehicle_type(value: str | None) -> str:
    return (value or "car").strip().lower() or "car"


def legal_min_depth_mm(cfg: AppConfig, vehicle_type: str | None) -> float:
    vt = normalize_vehicle_type(vehicle_type)
    if vt == "motorcycle":
        return float(getattr(cfg, "motorcycle_legal_min_depth_mm", 1.0))
    return float(getattr(cfg, "car_legal_min_depth_mm", 1.6))


def warning_band_mm(cfg: AppConfig, vehicle_type: str | None) -> float:
    vt = normalize_vehicle_type(vehicle_type)
    if vt == "motorcycle":
        return float(getattr(cfg, "motorcycle_warning_band_mm", 0.4))
    return float(getattr(cfg, "car_warning_band_mm", 0.4))


def tread_verdict_from_depth(cfg: AppConfig, depth_mm: float, vehicle_type: str | None) -> str:
    min_mm = legal_min_depth_mm(cfg, vehicle_type)
    warn_band = max(0.0, warning_band_mm(cfg, vehicle_type))
    if depth_mm < min_mm:
        return "REPLACE"
    if depth_mm < (min_mm + warn_band):
        return "WARNING"
    return "GOOD"


def psi_to_severity(psi_status: str | None) -> int:
    status = (psi_status or "").strip().upper()
    if not status or status == "NO_DATA" or status == "GOOD":
        return 0
    if "WARN" in status or status == "WARNING":
        return 1
    if "REPLACE" in status or "CRITICAL" in status:
        return 2
    return 0


def verdict_to_severity(verdict: str | None) -> int:
    status = (verdict or "").strip().upper()
    if status == "GOOD":
        return 0
    if "WARN" in status or status == "WARNING":
        return 1
    return 2


def combine_verdicts(tread_verdict: str, psi_status: str | None) -> str:
    sev = max(verdict_to_severity(tread_verdict), psi_to_severity(psi_status))
    return {0: "GOOD", 1: "WARNING", 2: "REPLACE"}[sev]


def rewrite_notes(notes: str | None, depth_mm: float | None, raw_score_verdict: str) -> str:
    base = notes or ""
    replacement = (
        f"depth_mm={depth_mm:.3f}; raw_score_verdict={raw_score_verdict}"
        if depth_mm is not None
        else f"depth_mm=UNCALIBRATED; raw_score_verdict={raw_score_verdict}"
    )
    pattern = r"depth_mm=[^;|\n]+; raw_score_verdict=[^;|\n]+"
    if re.search(pattern, base):
        return re.sub(pattern, replacement, base)
    if base:
        return f"{base} | {replacement}"
    return replacement


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill TireGuard results using current calibration/verdict policy.")
    parser.add_argument("--include-deleted", action="store_true", help="Also backfill soft-deleted rows")
    args = parser.parse_args()

    cfg = AppConfig()
    calib = load_calibration(cfg.calibration_path)
    use_depth_model = has_score_model(calib)

    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    query = "SELECT id, score, psi_status, vehicle_type, notes, deleted_at FROM results"
    if not args.include_deleted:
        query += " WHERE deleted_at IS NULL"
    cur.execute(query)
    rows = cur.fetchall()

    updated = 0
    for row in rows:
        score = float(row["score"] or 0.0)
        raw_score_verdict = pass_fail_from_score(score)
        device_depth_mm = score_to_depth_mm(score, calib=calib) if use_depth_model else None
        tread_verdict = tread_verdict_from_depth(cfg, device_depth_mm, row["vehicle_type"]) if device_depth_mm is not None else raw_score_verdict
        verdict = combine_verdicts(tread_verdict, row["psi_status"])
        notes = rewrite_notes(row["notes"], device_depth_mm, raw_score_verdict)

        cur.execute(
            """
            UPDATE results
            SET device_depth_mm=?, raw_score_verdict=?, tread_verdict=?, verdict=?, notes=?
            WHERE id=?
            """,
            (
                float(device_depth_mm) if device_depth_mm is not None else None,
                raw_score_verdict,
                tread_verdict,
                verdict,
                notes,
                row["id"],
            ),
        )
        updated += 1

    con.commit()
    con.close()

    scope = "all rows" if args.include_deleted else "active rows"
    print(f"Backfilled {updated} {scope}; depth_model={'yes' if use_depth_model else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
