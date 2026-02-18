import argparse
import csv
import time
from pathlib import Path

import cv2

from tireguard.config import AppConfig
from tireguard.preprocess import preprocess_bgr, crop_roi
from tireguard.quality import run_quality_checks
from tireguard.measure import groove_visibility_score, pass_fail_from_score
from tireguard.storage import get_result_by_ts


MAX_AVG_PERCENT_DIFF = 5.0
MAX_SINGLE_ERROR_MM = 0.5
MAX_PROCESSING_TIME_S = 5.0


def _normalize_clahe_grid(cfg):
    grid = getattr(cfg, "clahe_grid", 8)
    if isinstance(grid, int):
        return (grid, grid)
    if isinstance(grid, (list, tuple)) and len(grid) == 2:
        return (int(grid[0]), int(grid[1]))
    return (8, 8)


def score_to_depth_mm(score: float, slope: float, intercept: float) -> float:
    # depth_mm = slope * score + intercept (replace slope/intercept from your calibration fit)
    return max(0.0, slope * score + intercept)


def load_manual_csv(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = (row.get("ts") or "").strip()
            md = row.get("manual_depth_mm")
            if not ts or md is None or str(md).strip() == "":
                continue
            rows.append((ts, float(md)))
    return rows


def run(manual_csv: Path, out_csv: Path, slope: float, intercept: float):
    cfg = AppConfig()
    entries = load_manual_csv(manual_csv)
    if not entries:
        raise SystemExit("No valid manual rows. Expected columns: ts,manual_depth_mm")

    out_rows = []
    for ts, manual_mm in entries:
        db_row = get_result_by_ts(cfg, ts)
        if not db_row:
            continue

        image_path = Path(db_row["image_path"])
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

        t0 = time.perf_counter()
        roi_bgr = crop_roi(frame, roi)
        processed = preprocess_bgr(
            roi_bgr,
            clahe_clip=float(getattr(cfg, "clahe_clip", 2.0)),
            clahe_grid=_normalize_clahe_grid(cfg),
        )
        gray = processed.get("gray")
        if gray is None:
            continue

        q = run_quality_checks(gray, cfg)

        edges = processed.get("edges_closed")
        if edges is None:
            edges = processed.get("edges")
        if edges is None:
            continue

        m = groove_visibility_score(edges)
        verdict = pass_fail_from_score(m["score"])
        proc_s = time.perf_counter() - t0

        device_mm = score_to_depth_mm(float(m["score"]), slope=slope, intercept=intercept)
        pct_diff = (abs(device_mm - manual_mm) / manual_mm * 100.0) if manual_mm > 0 else None
        abs_err = abs(device_mm - manual_mm)

        out_rows.append({
            "ts": ts,
            "manual_depth_mm": round(manual_mm, 3),
            "device_depth_mm": round(device_mm, 3),
            "abs_error_mm": round(abs_err, 3),
            "percent_diff": round(pct_diff, 3) if pct_diff is not None else "",
            "score": round(float(m["score"]), 6),
            "verdict": verdict,
            "quality_ok": bool(q["ok"]),
            "processing_time_s": round(proc_s, 4),
        })

    if not out_rows:
        raise SystemExit("No rows processed.")

    valid_pct = [r["percent_diff"] for r in out_rows if r["percent_diff"] != ""]
    avg_pct = (sum(valid_pct) / len(valid_pct)) if valid_pct else float("inf")
    max_err = max(r["abs_error_mm"] for r in out_rows)
    max_t = max(r["processing_time_s"] for r in out_rows)

    pass_avg = avg_pct <= MAX_AVG_PERCENT_DIFF
    pass_err = max_err <= MAX_SINGLE_ERROR_MM
    pass_time = max_t <= MAX_PROCESSING_TIME_S

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Processed: {len(out_rows)}")
    print(f"Average % diff: {avg_pct:.3f} (<= {MAX_AVG_PERCENT_DIFF}) -> {'PASS' if pass_avg else 'FAIL'}")
    print(f"Max abs error (mm): {max_err:.3f} (<= {MAX_SINGLE_ERROR_MM}) -> {'PASS' if pass_err else 'FAIL'}")
    print(f"Max processing time (s): {max_t:.3f} (<= {MAX_PROCESSING_TIME_S}) -> {'PASS' if pass_time else 'FAIL'}")
    print(f"Overall: {'PASS' if (pass_avg and pass_err and pass_time) else 'FAIL'}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-csv", default="data/manual_readings.csv")
    ap.add_argument("--out-csv", default="data/test_results.csv")
    ap.add_argument("--slope", type=float, required=True, help="Calibration slope for depth_mm = slope*score + intercept")
    ap.add_argument("--intercept", type=float, required=True, help="Calibration intercept")
    args = ap.parse_args()

    run(
        manual_csv=Path(args.manual_csv),
        out_csv=Path(args.out_csv),
        slope=args.slope,
        intercept=args.intercept,
    )