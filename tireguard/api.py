# -*- coding: utf-8 -*-
from __future__ import annotations

import mimetypes
import argparse
import time
import sqlite3
from pathlib import Path
from typing import Any
from statistics import mean

from fastapi import FastAPI, HTTPException, Query, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# Reuse your existing modules
try:
    from .config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

from .config import AppConfig
from .calibration import has_score_model, load_calibration, save_calibration, score_to_depth_mm
from .measure import apply_defect_guard, estimate_groove_channel_frac, pass_fail_from_score
from .storage import (
    init_db,  # added
    list_results, get_result_by_ts, export_csv, find_processed_images,
  list_validation_results, export_validation_summary,
  soft_delete_scan_by_ts, soft_delete_scans_by_ts,
  restore_scan_by_ts, restore_scans_by_ts,
  hard_delete_scan_by_ts, hard_delete_scans_by_ts,
  soft_delete_validation_by_id, restore_validation_by_id, hard_delete_validation_by_id,
  soft_delete_all_data, purge_data,
)


LEGAL_MIN_DEPTH_MM = {
  "car": 1.6,
  "motorcycle": 1.0,
}

DEFAULT_WARNING_BAND_MM = {
  "car": 0.4,
  "motorcycle": 0.4,
}


def _legal_min_depth_mm(cfg: AppConfig, vehicle_type: str | None) -> float:
  v = (vehicle_type or "car").strip().lower()
  if v == "car":
    return float(getattr(cfg, "car_legal_min_depth_mm", LEGAL_MIN_DEPTH_MM["car"]))
  if v == "motorcycle":
    return float(getattr(cfg, "motorcycle_legal_min_depth_mm", LEGAL_MIN_DEPTH_MM["motorcycle"]))
  return LEGAL_MIN_DEPTH_MM.get(v, LEGAL_MIN_DEPTH_MM["car"])


def _warning_band_mm(cfg: AppConfig, vehicle_type: str | None) -> float:
  v = (vehicle_type or "car").strip().lower()
  if v == "car":
    return float(getattr(cfg, "car_warning_band_mm", DEFAULT_WARNING_BAND_MM["car"]))
  if v == "motorcycle":
    return float(getattr(cfg, "motorcycle_warning_band_mm", DEFAULT_WARNING_BAND_MM["motorcycle"]))
  return DEFAULT_WARNING_BAND_MM["car"]


def _validation_max_percent_diff(cfg: AppConfig) -> float:
  return max(0.0, float(getattr(cfg, "validation_max_percent_diff", 10.0)))


def _validation_max_abs_error_mm(cfg: AppConfig) -> float:
  return max(0.0, float(getattr(cfg, "validation_max_abs_error_mm", 0.5)))


def _tread_verdict_from_depth(cfg: AppConfig, depth_mm: float, vehicle_type: str | None) -> str:
  min_mm = _legal_min_depth_mm(cfg, vehicle_type)
  warn_band = max(0.0, _warning_band_mm(cfg, vehicle_type))
  if depth_mm < min_mm:
    return "REPLACE"
  if depth_mm < (min_mm + warn_band):
    return "WARNING"
  return "GOOD"


def _defect_guard_kwargs(cfg: AppConfig) -> dict[str, float]:
  return {
    "replace_channel_frac": float(getattr(cfg, "tread_guard_replace_channel_frac", 0.006)),
    "warning_channel_frac": float(getattr(cfg, "tread_guard_warning_channel_frac", 0.018)),
    "replace_score": float(getattr(cfg, "tread_guard_replace_score", 0.035)),
    "warning_score": float(getattr(cfg, "tread_guard_warning_score", 0.065)),
  }


def _enrich_row_depth_and_raw_verdict(cfg: AppConfig, row: dict[str, Any], calib: dict | None = None) -> dict[str, Any]:
  """Populate missing depth/raw verdict fields using the calibrated score model when possible."""
  if not isinstance(row, dict):
    return row

  score = row.get("score")
  if score is None:
    return row

  try:
    score_f = float(score)
  except Exception:
    return row

  c = calib if calib is not None else load_calibration(cfg.calibration_path)
  depth_mm = None
  if has_score_model(c):
    depth_mm = score_to_depth_mm(score_f, calib=c)

  if row.get("device_depth_mm") is None and depth_mm is not None:
    row["device_depth_mm"] = float(depth_mm)

  if not row.get("raw_score_verdict"):
    vt = row.get("vehicle_type")
    groove_proxy = estimate_groove_channel_frac(
      score=score_f,
      edge_density=row.get("edge_density"),
      continuity=row.get("continuity"),
    )
    if depth_mm is not None:
      base = _tread_verdict_from_depth(cfg, float(depth_mm), vt)
    else:
      base = pass_fail_from_score(
        score_f,
        groove_channel_frac=groove_proxy,
        quality_ok=True,
        **_defect_guard_kwargs(cfg),
      )
    row["raw_score_verdict"] = apply_defect_guard(
      base,
      groove_channel_frac=groove_proxy,
      quality_ok=True,
      score=score_f,
      **_defect_guard_kwargs(cfg),
    )
  return row


def _load_validation_pairs(cfg: AppConfig, include_deleted: bool = False) -> list[tuple[float, float]]:
  con = sqlite3.connect(cfg.db_path)
  con.row_factory = sqlite3.Row
  cur = con.cursor()
  deleted_sql = "" if include_deleted else "AND deleted_at IS NULL"
  cur.execute(
    f"""
    SELECT manual_depth, device_score
    FROM validation_results
    WHERE manual_depth IS NOT NULL
      AND device_score IS NOT NULL
      {deleted_sql}
    ORDER BY id ASC
    """
  )
  rows = cur.fetchall()
  con.close()

  out: list[tuple[float, float]] = []
  for row in rows:
    try:
      out.append((float(row["manual_depth"]), float(row["device_score"])))
    except Exception:
      continue
  return out


def _fit_linear_score_model(pairs: list[tuple[float, float]]) -> dict[str, float] | None:
  if len(pairs) < 2:
    return None

  ys = [p[0] for p in pairs]  # manual depth
  xs = [p[1] for p in pairs]  # device score
  mx = mean(xs)
  my = mean(ys)
  den = sum((x - mx) ** 2 for x in xs)
  if den == 0.0:
    return None
  slope = sum((x - mx) * (y - my) for y, x in pairs) / den
  intercept = my - slope * mx

  preds = [slope * x + intercept for x in xs]
  abs_err = [abs(y - p) for y, p in zip(ys, preds)]
  mae = sum(abs_err) / len(abs_err)
  ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
  ss_tot = sum((y - my) ** 2 for y in ys)
  r2 = 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
  return {
    "slope": float(slope),
    "intercept": float(intercept),
    "mae_mm": float(mae),
    "r2": float(r2),
    "sample_count": float(len(pairs)),
  }


def _refresh_derived_metrics_from_calibration(cfg: AppConfig, calib: dict, include_deleted: bool = False) -> tuple[int, int]:
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
    score = row["score"]
    if score is None:
      skipped += 1
      continue
    try:
      score_f = float(score)
    except Exception:
      skipped += 1
      continue

    depth_mm = float(score_to_depth_mm(score_f, calib=calib))
    base_verdict = _tread_verdict_from_depth(cfg, depth_mm, row["vehicle_type"])
    groove_proxy = estimate_groove_channel_frac(
      score=score_f,
      edge_density=row["edge_density"],
      continuity=row["continuity"],
    )
    tread_verdict = apply_defect_guard(
      base_verdict,
      groove_channel_frac=groove_proxy,
      quality_ok=True,
      score=score_f,
      **_defect_guard_kwargs(cfg),
    )

    cur.execute(
      """
      UPDATE results
      SET device_depth_mm = ?, raw_score_verdict = ?, tread_verdict = ?, verdict = ?
      WHERE id = ?
      """,
      (depth_mm, tread_verdict, tread_verdict, tread_verdict, row["id"]),
    )
    updated += 1

  con.commit()
  con.close()
  return updated, skipped


def _auto_fit_score_model_and_refresh(cfg: AppConfig) -> dict[str, Any]:
  pairs = _load_validation_pairs(cfg, include_deleted=False)
  fit = _fit_linear_score_model(pairs)
  if fit is None:
    return {
      "updated_model": False,
      "reason": "insufficient_pairs",
      "pairs": len(pairs),
    }

  calib = load_calibration(cfg.calibration_path)
  calib["score_model"] = {
    "type": "linear",
    "slope": fit["slope"],
    "intercept": fit["intercept"],
  }
  save_calibration(cfg.calibration_path, calib)

  updated, skipped = _refresh_derived_metrics_from_calibration(cfg, calib, include_deleted=False)
  return {
    "updated_model": True,
    "pairs": len(pairs),
    "slope": fit["slope"],
    "intercept": fit["intercept"],
    "mae_mm": fit["mae_mm"],
    "r2": fit["r2"],
    "refreshed_rows": updated,
    "skipped_rows": skipped,
  }


def _cfg():
    """
    Returns your app config. Falls back to AppConfig if load_config() is unavailable.
    """
    if load_config is not None:
        try:
            return load_config()
        except Exception:
            pass
    return AppConfig()


def _safe_path(p: Any) -> Any:
    """Convert Path objects to strings for JSON serialization."""
    if isinstance(p, Path):
        return str(p)
    return p


def _jsonify(obj: Any) -> Any:
    """Recursively convert objects to JSON-safe format."""
    if isinstance(obj, dict):
        return {k: _jsonify(_safe_path(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    return _safe_path(obj)


app = FastAPI(title="TireGuard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "http://0.0.0.0:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Health & Status ----------
@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "service": "tireguard-api"}


@app.get("/api/tread-policy")
def tread_policy():
  cfg = _cfg()
  calib = load_calibration(cfg.calibration_path)
  return _jsonify({
    "legal_min_mm": {
      "car": _legal_min_depth_mm(cfg, "car"),
      "motorcycle": _legal_min_depth_mm(cfg, "motorcycle"),
    },
    "warning_band_mm": {
      "car": _warning_band_mm(cfg, "car"),
      "motorcycle": _warning_band_mm(cfg, "motorcycle"),
    },
    "defect_guard": _defect_guard_kwargs(cfg),
    "quality_policy": {
      "min_brightness": float(getattr(cfg, "min_brightness", 40.0)),
      "max_brightness": float(getattr(cfg, "max_brightness", 220.0)),
      "max_glare_ratio": float(getattr(cfg, "max_glare_ratio", 0.06)),
      "min_sharpness": float(getattr(cfg, "min_sharpness", 80.0)),
      "relaxed_min_sharpness": float(getattr(cfg, "quality_relaxed_min_sharpness", 24.0)),
      "relaxed_strong_score": float(getattr(cfg, "quality_relaxed_strong_score", 0.11)),
      "relaxed_channel_frac": float(getattr(cfg, "quality_relaxed_channel_frac", 0.020)),
      "relaxed_tread_confidence": float(getattr(cfg, "quality_relaxed_tread_confidence", 0.55)),
    },
    "automation": {
      "auto_detect_tread_on_roi": bool(getattr(cfg, "auto_detect_tread_on_roi", True)),
      "auto_calibrate_on_roi": bool(getattr(cfg, "auto_calibrate_on_roi", True)),
      "auto_calibration_reference_mm": float(getattr(cfg, "auto_calibration_reference_mm", 120.0)),
      "auto_calibration_reference_mm_car": float(getattr(cfg, "auto_calibration_reference_mm_car", 120.0)),
      "auto_calibration_reference_mm_motorcycle": float(getattr(cfg, "auto_calibration_reference_mm_motorcycle", 85.0)),
      "quick_session_auto_capture": bool(getattr(cfg, "quick_session_auto_capture", True)),
      "quick_session_capture_cooldown_s": float(getattr(cfg, "quick_session_capture_cooldown_s", 1.8)),
    },
    "validation_policy": {
      "max_percent_diff": _validation_max_percent_diff(cfg),
      "max_abs_error_mm": _validation_max_abs_error_mm(cfg),
      "require_tread_class_match": True,
    },
    "calibration_status": {
      "score_model_available": has_score_model(calib),
      "calibration_method": calib.get("method") if isinstance(calib, dict) else None,
    },
  })


# ---------- Recent Scans ----------
@app.get("/api/scans")
def scans(
    limit: int = Query(50, ge=1, le=500),
    vehicle_id: str | None = None,
    tire_position: str | None = None,
    verdict: str | None = None,
    vehicle_type: str | None = None,
    tire_type: str | None = None,
    tread_design: str | None = None,
    tire_model_code: str | None = None,
    include_deleted: bool = Query(False),
    only_deleted: bool = Query(False),
):
    """List recent scan results with filtering support."""
    cfg = _cfg()
    rows = list_results(
        cfg,
        limit=limit,
        vehicle_id=vehicle_id,
        tire_position=tire_position,
        verdict=verdict,
        vehicle_type=vehicle_type,
        tire_type=tire_type,
        tread_design=tread_design,
        tire_model_code=tire_model_code,
        include_deleted=include_deleted,
        only_deleted=only_deleted,
    )
    calib = load_calibration(cfg.calibration_path)
    rows = [_enrich_row_depth_and_raw_verdict(cfg, r, calib=calib) for r in rows]
    return _jsonify({"items": rows, "limit": limit, "include_deleted": include_deleted, "only_deleted": only_deleted})


@app.get("/api/scans/{ts}")
def scan_detail(ts: str, include_deleted: bool = Query(False)):
    """Get detailed information for a specific scan by timestamp."""
    cfg = _cfg()
    row = get_result_by_ts(cfg, ts, include_deleted=include_deleted)
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    calib = load_calibration(cfg.calibration_path)
    row = _enrich_row_depth_and_raw_verdict(cfg, row, calib=calib)
    return _jsonify(row)


@app.delete("/api/scans/{ts}")
def delete_scan(ts: str):
    """Move one scan row to recycle bin."""
    cfg = _cfg()
    out = soft_delete_scan_by_ts(cfg, ts)
    if out.get("deleted", 0) == 0:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _jsonify({"ok": True, "ts": ts, **out})


@app.post("/api/scans/{ts}/restore")
def restore_scan(ts: str):
    cfg = _cfg()
    out = restore_scan_by_ts(cfg, ts)
    if out.get("restored", 0) == 0:
        raise HTTPException(status_code=404, detail="Scan not found in recycle bin")
    return _jsonify({"ok": True, "ts": ts, **out})


@app.delete("/api/scans/{ts}/hard")
def hard_delete_scan(ts: str, delete_files: bool = Query(True)):
    cfg = _cfg()
    out = hard_delete_scan_by_ts(cfg, ts, delete_files=delete_files)
    if out.get("deleted", 0) == 0:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _jsonify({"ok": True, "ts": ts, **out})


@app.post("/api/scans/delete-batch")
def delete_scan_batch(payload: dict = Body(...)):
    """Batch soft-delete scans by timestamp list."""
    cfg = _cfg()
    ts_list = payload.get("ts_list") if isinstance(payload, dict) else None
    if not isinstance(ts_list, list) or not ts_list:
        raise HTTPException(status_code=400, detail="Body must include non-empty ts_list")
    ts_clean = [str(x).strip() for x in ts_list if str(x).strip()]
    if not ts_clean:
        raise HTTPException(status_code=400, detail="No valid timestamps provided")
    out = soft_delete_scans_by_ts(cfg, ts_clean)
    return _jsonify({"ok": True, "requested": len(ts_clean), **out})


@app.post("/api/scans/restore-batch")
def restore_scan_batch(payload: dict = Body(...)):
    cfg = _cfg()
    ts_list = payload.get("ts_list") if isinstance(payload, dict) else None
    if not isinstance(ts_list, list) or not ts_list:
        raise HTTPException(status_code=400, detail="Body must include non-empty ts_list")
    ts_clean = [str(x).strip() for x in ts_list if str(x).strip()]
    if not ts_clean:
        raise HTTPException(status_code=400, detail="No valid timestamps provided")
    out = restore_scans_by_ts(cfg, ts_clean)
    return _jsonify({"ok": True, "requested": len(ts_clean), **out})


@app.post("/api/scans/hard-delete-batch")
def hard_delete_scan_batch(payload: dict = Body(...)):
    cfg = _cfg()
    ts_list = payload.get("ts_list") if isinstance(payload, dict) else None
    delete_files = bool(payload.get("delete_files", True)) if isinstance(payload, dict) else True
    if not isinstance(ts_list, list) or not ts_list:
        raise HTTPException(status_code=400, detail="Body must include non-empty ts_list")
    ts_clean = [str(x).strip() for x in ts_list if str(x).strip()]
    if not ts_clean:
        raise HTTPException(status_code=400, detail="No valid timestamps provided")
    out = hard_delete_scans_by_ts(cfg, ts_clean, delete_files=delete_files)
    return _jsonify({"ok": True, "requested": len(ts_clean), **out})


@app.get("/api/scans/{ts}/images")
def scan_images(ts: str):
    """Get list of processed images for a scan."""
    cfg = _cfg()
    paths = find_processed_images(cfg, ts)
    if not paths:
        raise HTTPException(status_code=404, detail="No processed images found")
    return _jsonify({"ts": ts, "images": {k: str(v) for k, v in paths.items()}})


@app.get("/api/images/{ts}/{kind}")
def image(ts: str, kind: str):
    """Serve a specific processed image (gray, edges, etc.)."""
    cfg = _cfg()
    paths = find_processed_images(cfg, ts)
    if not paths or kind not in paths:
        raise HTTPException(status_code=404, detail="Image kind not available")

    p = Path(paths[kind])
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image file missing")

    mime, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=mime or "image/png")


# ---------- CSV Export ----------
@app.get("/api/export/csv")
def export_csv_route():
    """Export all scan results to CSV."""
    cfg = _cfg()
    p = export_csv(cfg)
    p = Path(p)
    if not p.exists():
        raise HTTPException(status_code=500, detail="CSV export failed")
    return FileResponse(str(p), media_type="text/csv", filename=p.name)


# ---------- Device Validation ----------
@app.get("/api/validation")
def validation(
    limit: int = Query(50, ge=1, le=500),
    tire_id: str | None = None,
    verdict: str | None = None,
    include_deleted: bool = Query(False),
    only_deleted: bool = Query(False),
):
    """List device validation results (Table 3.2 methodology)."""
    cfg = _cfg()
    rows = list_validation_results(
        cfg,
        limit=limit,
        tire_id=tire_id,
        verdict=verdict,
        include_deleted=include_deleted,
        only_deleted=only_deleted,
    )
    return _jsonify({"items": rows, "limit": limit, "include_deleted": include_deleted, "only_deleted": only_deleted})


@app.delete("/api/validation/{validation_id}")
def delete_validation(validation_id: int):
    """Move one validation entry to recycle bin."""
    cfg = _cfg()
    out = soft_delete_validation_by_id(cfg, validation_id)
    if out.get("deleted", 0) == 0:
        raise HTTPException(status_code=404, detail="Validation entry not found")
    return _jsonify({"ok": True, "id": validation_id, **out})


@app.post("/api/validation/{validation_id}/restore")
def restore_validation(validation_id: int):
    cfg = _cfg()
    out = restore_validation_by_id(cfg, validation_id)
    if out.get("restored", 0) == 0:
        raise HTTPException(status_code=404, detail="Validation entry not found in recycle bin")
    return _jsonify({"ok": True, "id": validation_id, **out})


@app.delete("/api/validation/{validation_id}/hard")
def hard_delete_validation(validation_id: int):
    cfg = _cfg()
    out = hard_delete_validation_by_id(cfg, validation_id)
    if out.get("deleted", 0) == 0:
        raise HTTPException(status_code=404, detail="Validation entry not found")
    return _jsonify({"ok": True, "id": validation_id, **out})


@app.post("/api/data/recycle-all")
def recycle_all_data(include_validation: bool = Query(True)):
    cfg = _cfg()
    out = soft_delete_all_data(cfg, include_validation=include_validation)
    return _jsonify({"ok": True, "include_validation": include_validation, **out})


@app.post("/api/data/purge")
def purge_all_data(
    include_validation: bool = Query(True),
    delete_files: bool = Query(True),
    only_deleted: bool = Query(True),
):
    """Permanently delete data, usually only rows already moved to recycle bin."""
    cfg = _cfg()
    out = purge_data(cfg, include_validation=include_validation, delete_files=delete_files, only_deleted=only_deleted)
    return _jsonify({
        "ok": True,
        "include_validation": include_validation,
        "delete_files": delete_files,
        "only_deleted": only_deleted,
        **out,
    })


@app.get("/api/export/validation")
def export_validation_route():
    """Export validation summary (Table 3.2) to CSV."""
    cfg = _cfg()
    p = export_validation_summary(cfg)
    p = Path(p)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Validation CSV not found")
    return FileResponse(str(p), media_type="text/csv", filename=p.name)


@app.post("/api/calibration/refresh-derived")
def refresh_derived_metrics(include_deleted: bool = Query(False)):
    """Recompute stored derived scan metrics from current score_model calibration."""
    cfg = _cfg()
    calib = load_calibration(cfg.calibration_path)
    if not has_score_model(calib):
        raise HTTPException(status_code=400, detail="No score_model found in calibration. Fit and save model first.")

    updated, skipped = _refresh_derived_metrics_from_calibration(cfg, calib, include_deleted=include_deleted)

    return _jsonify({
      "ok": True,
      "updated": updated,
      "skipped": skipped,
      "include_deleted": include_deleted,
      "score_model_available": True,
    })


# ---------- Web Dashboard ----------
@app.post("/api/validation/submit")
def submit_validation(
    tire_id: str = Form(...),
    manual_depth: float = Form(...),
    scan_ts: str = Form(...)
):
    """Submit a manual validation entry linked to an existing scan."""
    cfg = _cfg()

    # ✅ Fix: always return JSON, never let FastAPI return HTML error pages
    row = get_result_by_ts(cfg, scan_ts)
    if not row:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Scan '{scan_ts}' not found. Make sure you captured a scan first before submitting validation."}
        )

    device_score = float(row.get("score") or 0.0)
    calib = load_calibration(cfg.calibration_path)
    has_model = has_score_model(calib)
    device_depth = score_to_depth_mm(device_score, calib=calib) if has_model else None

    vehicle_type = row.get("vehicle_type")
    groove_proxy = estimate_groove_channel_frac(
      score=device_score,
      edge_density=row.get("edge_density"),
      continuity=row.get("continuity"),
    )
    min_mm = _legal_min_depth_mm(cfg, vehicle_type)
    manual_class = _tread_verdict_from_depth(cfg, float(manual_depth), vehicle_type)
    if has_model and device_depth is not None:
      device_class = _tread_verdict_from_depth(cfg, float(device_depth), vehicle_type)
      device_class = apply_defect_guard(
        device_class,
        groove_channel_frac=groove_proxy,
        quality_ok=True,
        score=device_score,
        **_defect_guard_kwargs(cfg),
      )
    else:
      # Provisional class before depth calibration is available.
      device_class = apply_defect_guard(
        pass_fail_from_score(
          device_score,
          groove_channel_frac=groove_proxy,
          quality_ok=True,
          **_defect_guard_kwargs(cfg),
        ),
        groove_channel_frac=groove_proxy,
        quality_ok=True,
        score=device_score,
        **_defect_guard_kwargs(cfg),
      )
    class_match = (manual_class == device_class)

    abs_error = abs(device_depth - manual_depth) if has_model and device_depth is not None else None
    percent_diff = ((abs_error / manual_depth) * 100.0) if (abs_error is not None and manual_depth > 0) else None

    val_pct_max = _validation_max_percent_diff(cfg)
    val_abs_max = _validation_max_abs_error_mm(cfg)

    if percent_diff is None or abs_error is None:
      verdict = "PENDING"
    else:
      pass_pct = percent_diff <= val_pct_max
      pass_err = abs_error <= val_abs_max
      verdict = "PASS" if (pass_pct and pass_err and class_match) else "FAIL"

    from .storage import insert_validation_result
    insert_validation_result(cfg, {
        "ts": time.strftime("%Y%m%d_%H%M%S"),
        "tire_id": tire_id,
        "manual_depth": manual_depth,
        "device_score": device_score,
        "device_depth": device_depth,
        "percent_diff": percent_diff,
        "abs_error_mm": abs_error,
        "processing_time": 0.0,
        "verdict": verdict,
        "notes": (
          f"Submitted via web dashboard | scan_ts={scan_ts} | vehicle_type={vehicle_type} | "
          f"legal_min={min_mm:.1f}mm | manual_class={manual_class} | device_class={device_class} | "
          f"score_model_available={has_model} | "
          f"class_match={class_match} | max_pct_diff={val_pct_max:.2f} | max_abs_err_mm={val_abs_max:.3f} | "
          f"groove_proxy={groove_proxy if groove_proxy is not None else 'n/a'}"
        ),
    })

    auto_fit = _auto_fit_score_model_and_refresh(cfg)

    return JSONResponse(content=_jsonify({
        "ok": True,
        "tire_id": tire_id,
        "manual_depth": manual_depth,
        "device_depth": device_depth,
        "percent_diff": percent_diff,
        "abs_error_mm": abs_error,
        "vehicle_type": vehicle_type,
        "legal_min_mm": min_mm,
        "manual_class": manual_class,
        "device_class": device_class,
        "groove_channel_proxy": groove_proxy,
        "class_match": class_match,
        "score_model_available": has_model,
        "validation_max_percent_diff": val_pct_max,
        "validation_max_abs_error_mm": val_abs_max,
        "auto_score_model": auto_fit,
        "verdict": verdict,
    }))


@app.get("/")
def index():
    """Serve the main TireGuard web dashboard."""
    cfg = _cfg()
    car_min_mm = _legal_min_depth_mm(cfg, "car")
    car_warn_mm = _warning_band_mm(cfg, "car")
    moto_min_mm = _legal_min_depth_mm(cfg, "motorcycle")
    moto_warn_mm = _warning_band_mm(cfg, "motorcycle")
    val_pct_max = _validation_max_percent_diff(cfg)
    val_abs_max = _validation_max_abs_error_mm(cfg)
    val_pct_warn = max(val_pct_max * 2.5, val_pct_max + 5.0)
    val_abs_warn = max(val_abs_max * 3.0, val_abs_max + 0.5)
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TireGuard Dashboard</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --primary: #0d6efd;
      --primary-dark: #0b5ed7;
      --success: #198754;
      --warning: #ffc107;
      --danger: #dc3545;
      --dark: #212529;
      --light: #f8f9fa;
      --gray-100: #f8f9fa;
      --gray-200: #e9ecef;
      --gray-300: #dee2e6;
      --gray-400: #ced4da;
      --gray-500: #adb5bd;
      --gray-600: #6c757d;
      --gray-700: #495057;
      --gray-800: #343a40;
      --gray-900: #212529;
      --card-bg: #ffffff;
      --card-border: #e0e0e0;
      --shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
      --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      --shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding-bottom: 2rem;
    }

    header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1.25rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .header-content {
      display: flex;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .logo-icon {
      background: white;
      color: var(--primary);
      width: 40px;
      height: 40px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.25rem;
      font-weight: bold;
    }

    .logo h1 {
      font-weight: 700;
      padding: 0.5rem 0;
      margin: 0;
    }

    .logo p {
      font-size: 0.8rem;
      opacity: 0.9;
      margin-top: 0.1rem;
    }

    .controls {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
    }

    .btn {
      padding: 0.65rem 1rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: var(--transition);
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
    }

    .btn-primary {
      background-color: white;
      color: var(--primary);
      box-shadow: var(--shadow);
    }

    .btn-primary:hover {
      background-color: var(--gray-100);
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .btn-outline {
      background-color: transparent;
      color: white;
      border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .btn-outline:hover {
      background-color: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.5);
    }

    .btn-danger {
      background-color: #dc3545;
      color: white;
    }

    .btn-danger:hover {
      background-color: #bb2d3b;
      transform: translateY(-2px);
    }

    .btn-subtle-danger {
      background-color: rgba(220, 53, 69, 0.1);
      color: #dc3545;
      border: 1px solid rgba(220, 53, 69, 0.25);
    }

    .btn-subtle-danger:hover {
      background-color: rgba(220, 53, 69, 0.18);
    }

    .toggle {
      display: inline-block;
      position: relative;
      width: 50px;
      height: 24px;
    }

    .toggle input {
      opacity: 0;
      width: 0;
      height: 0;
    }

    .slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #ccc;
      transition: .4s;
      border-radius: 34px;
    }

    .slider:before {
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 2px;
      bottom: 2px;
      background-color: white;
      transition: .4s;
      border-radius: 50%;
    }

    input:checked + .slider {
      background-color: var(--primary);
    }

    input:checked + .slider:before {
      transform: translateX(18px);
    }

    .status-indicator {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      padding: 0.5rem;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.15);
    }

    .app-launch-btn {
      position: absolute;
      top: 1.25rem;
      right: 1.25rem;
      background: rgba(255, 255, 255, 0.15);
      border: 1px solid rgba(255, 255, 255, 0.3);
      color: white;
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
      font-size: 0.85rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: var(--transition);
    }

    .app-launch-btn:hover {
      background: rgba(255, 255, 255, 0.25);
      transform: translateY(-1px);
    }

    .app-launch-btn i {
      font-size: 1rem;
    }

    /* Main Content */
    .container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      padding: 1rem;
      max-width: 1600px;
      margin: 0 auto;
    }

    /* Card Styles */
    .card {
      background: var(--card-bg);
      border-radius: 12px;
      box-shadow: var(--shadow);
      overflow: hidden;
      border: 1px solid var(--card-border);
    }

    .card-header {
      padding: 1rem;
      border-bottom: 1px solid var(--card-border);
      background: #f8f9fa;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .card-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--gray-800);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .card-title i {
      color: var(--primary);
    }

    .card-body {
      padding: 1.25rem;
    }

    /* Filters */
    .filters {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .filter-group {
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }

    .filter-group label {
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--gray-600);
    }

    .filter-control {
      padding: 0.6rem;
      border: 1px solid var(--gray-300);
      border-radius: 8px;
      font-size: 0.9rem;
      transition: var(--transition);
    }

    .filter-control:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25);
    }

    /* Table Styles */
    table {
      width: 100%;
      border-collapse: collapse;
      display: block;
      overflow-x: auto;
    }

    thead {
      background-color: #f8f9fa;
    }

    th, td {
      padding: 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--gray-200);
      font-size: 0.85rem;
      white-space: nowrap;
    }

    th {
      font-weight: 600;
      color: var(--gray-600);
      text-transform: uppercase;
      letter-spacing: 0.025em;
      font-size: 0.75rem;
    }

    tr:hover {
      background-color: #f8f9fa;
      cursor: pointer;
    }

    .card-body > table {
      max-height: 400px;
      overflow-y: auto;
      display: block;
      border-bottom: 1px solid var(--gray-200);
    }

    @media (max-width: 768px) {
      .card-body > table {
        max-height: 300px;
      }
    }

    .verdict-badge {
      padding: 0.3rem 0.7rem;
      border-radius: 30px;
      font-weight: 600;
      font-size: 0.8rem;
      display: inline-flex;
      align-items: center;
      gap: 0.5;
    }

    .verdict-pass {
      background-color: rgba(25, 135, 84, 0.1);
      color: var(--success);
      border: 1px solid rgba(25, 135, 84, 0.2);
    }

    .verdict-warning {
      background-color: rgba(255, 193, 7, 0.1);
      color: var(--warning);
      border: 1px solid rgba(255, 193, 7, 0.2);
    }

    .verdict-fail {
      background-color: rgba(220, 53, 69, 0.1);
      color: var(--danger);
      border: 1px solid rgba(220, 53, 69, 0.2);
    }

    .verdict-good {
      background-color: rgba(34, 197, 94, 0.1);
      color: #22c55e;
      border: 1px solid rgba(34, 197, 94, 0.2);
    }

    .verdict-replace {
      background-color: rgba(239, 68, 68, 0.1);
      color: #ef4444;
      border: 1px solid rgba(239, 68, 68, 0.2);
    }

    /* Scan Details */
    .scan-info {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .info-item {
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }

    .info-label {
      font-size: 0.75rem;
      color: var(--gray-600);
      font-weight: 600;
    }

    .info-value {
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--gray-800);
    }

    .verdict-container {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .scan-verdict {
      font-size: 1.1rem;
      font-weight: 800;
      padding: 0.5rem 1rem;
      border-radius: 12px;
      display: inline-block;
    }

    .pass-rate {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: #f8f9fa;
      border-radius: 12px;
      font-weight: 600;
      font-size: 1rem;
      border: 1px solid var(--gray-200);
    }

    .pass-rate-value {
      font-size: 1.2rem;
      color: var(--success);
    }

    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .image-card {
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--gray-200);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
      transition: var(--transition);
    }

    .image-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }

    .image-container {
      position: relative;
      padding-top: 66.67%; /* 3:2 aspect ratio */
      background: #f0f2f5;
    }

    .image-container img {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: var(--transition);
    }

    .image-card:hover .image-container img {
      transform: scale(1.05);
    }

    .image-label {
      padding: 0.5rem;
      background: white;
      border-top: 1px solid var(--gray-200);
      font-weight: 600;
      color: var(--gray-700);
      font-size: 0.8rem;
      text-align: center;
    }

    .histogram-container {
      position: relative;
      height: 150px;
      margin: 1rem 0;
      background: white;
      border-radius: 12px;
      border: 1px solid var(--gray-200);
    }

    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }

    .details-panel {
      background: #f8f9fa;
      border-radius: 12px;
      padding: 1rem;
      font-family: 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
      font-size: 0.8rem;
      color: var(--gray-700);
      border: 1px solid var(--gray-200);
      max-height: 250px;
      overflow-y: auto;
    }

    .validation-layout {
      display: grid;
      grid-template-columns: minmax(280px, 0.95fr) minmax(360px, 1.25fr);
      gap: 1rem;
      align-items: start;
      margin-bottom: 1.25rem;
    }

    .guide-card {
      background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
      border: 1px solid #cfe0ff;
      border-radius: 14px;
      padding: 1.25rem;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }

    .guide-card h3 {
      font-size: 1rem;
      font-weight: 800;
      color: #1d4ed8;
      margin-bottom: 0.35rem;
    }

    .guide-card p {
      font-size: 0.84rem;
      color: var(--gray-700);
      line-height: 1.55;
      margin-bottom: 0.8rem;
    }

    .guide-table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 0.85rem;
      font-size: 0.82rem;
      background: rgba(255,255,255,0.9);
      border-radius: 12px;
      overflow: hidden;
    }

    .guide-table th,
    .guide-table td {
      padding: 0.7rem 0.65rem;
      border-bottom: 1px solid #d9e3f7;
      text-align: left;
      vertical-align: top;
    }

    .guide-table th {
      background: #dbeafe;
      color: #1e3a8a;
      font-weight: 800;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .guide-table tr:last-child td {
      border-bottom: none;
    }

    .depth-chip {
      display: inline-flex;
      align-items: center;
      padding: 0.18rem 0.48rem;
      border-radius: 999px;
      font-weight: 800;
      font-size: 0.75rem;
      white-space: nowrap;
    }

    .depth-chip.replace {
      background: rgba(220, 38, 38, 0.12);
      color: #b91c1c;
    }

    .depth-chip.warning {
      background: rgba(245, 158, 11, 0.16);
      color: #b45309;
    }

    .depth-chip.good {
      background: rgba(22, 163, 74, 0.14);
      color: #166534;
    }

    .guide-steps {
      margin: 0;
      padding-left: 1rem;
      color: var(--gray-700);
      font-size: 0.82rem;
      line-height: 1.55;
    }

    .guide-note {
      margin-top: 0.85rem;
      padding: 0.8rem 0.9rem;
      border-radius: 10px;
      background: rgba(29, 78, 216, 0.08);
      color: #1e3a8a;
      font-size: 0.8rem;
      font-weight: 600;
    }

    /* Modal */
    .modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 1rem;
    }

    .modal.active {
      display: flex;
    }

    .modal-content {
      max-width: 95%;
      max-height: 95%;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
      background: white;
    }

    .modal-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .modal-close {
      position: absolute;
      top: 0.75rem;
      right: 0.75rem;
      width: 35px;
      height: 35px;
      border-radius: 50%;
      background: rgba(0, 0, 0, 0.2);
      color: white;
      border: none;
      font-size: 1.5rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10;
    }

    /* Mobile styles */
    @media (max-width: 768px) {
      .container {
        grid-template-columns: 1fr;
        gap: 1rem;
        padding: 0.75rem;
      }

      .header-content {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
      }

      .logo h1 {
        font-size: 1.3rem;
      }

      .controls {
        width: 100%;
        justify-content: space-between;
      }

      .btn {
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
      }

      .card-title {
        font-size: 1rem;
      }

      .scan-info {
        grid-template-columns: 1fr;
      }

      .image-grid {
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 0.5rem;
      }

      .verdict-container {
        flex-direction: column;
        align-items: flex-start;
      }

      .pass-rate {
        width: 100%;
      }

      .validation-layout {
        grid-template-columns: 1fr;
      }

      .filters {
        grid-template-columns: 1fr;
      }

      th, td {
        padding: 0.5rem;
        font-size: 0.75rem;
      }

      .card-body > table {
        display: block;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
      }

      ::-webkit-scrollbar {
        height: 8px;
      }
      ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
      }
      ::-webkit-scrollbar-thumb {
        background: rgba(120, 220, 255, 0.3);
        border-radius: 4px;
      }

      .app-launch-btn {
        display: none;
      }
    }

    /* Tablet styles */
    @media (min-width: 769px) and (max-width: 1024px) {
      .container {
        padding: 1rem;
      }

      .filters {
        grid-template-columns: repeat(2, 1fr);
      }

      .scan-info {
        grid-template-columns: 1fr;
      }
    }

    /* Animations */
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
      animation: fadeIn 0.3s ease-out forwards;
    }
  </style>
</head>
<body>
  <header>
    <div class="header-content">
      <div class="logo">
        <div class="logo-icon">
          <i class="fas fa-tire"></i>
        </div>
        <div>
          <h1>TireGuard Dashboard</h1>
          <p>Real-time tire health monitoring system</p>
        </div>
      </div>
    </div>
    <div class="controls">
      <button id="btnRefresh" class="btn btn-primary">
        <i class="fas fa-sync-alt"></i> Refresh Data
      </button>
      <button id="btnExport" class="btn btn-outline">
        <i class="fas fa-file-csv"></i> Export CSV
      </button>
      <div class="toggle-switch">
        <span>Auto-refresh</span>
        <label class="toggle">
          <input type="checkbox" id="autoRefresh" checked>
          <span class="slider"></span>
        </label>
      </div>
      <div class="status-indicator">
        <i class="fas fa-circle status-dot" style="color: #198754;"></i>
        <span id="status">Ready</span>
      </div>
    </div>
    <div class="app-launch-btn">
      <i class="fas fa-desktop"></i> Desktop App Available
    </div>
  </header>

  <input type="checkbox" id="showRecycleBin" style="display:none;">

  <div class="container">
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-header">
        <h2 class="card-title"><i class="fas fa-recycle"></i> Data Management</h2>
      </div>
      <div class="card-body" style="display:flex; gap:1rem; align-items:center; justify-content:space-between; flex-wrap:wrap;">
        <div style="display:flex; gap:0.75rem; flex-wrap:wrap; align-items:center;">
          <button id="btnViewActiveData" class="btn btn-primary">
            <i class="fas fa-database"></i> Active Data
          </button>
          <button id="btnViewRecycleBin" class="btn btn-outline">
            <i class="fas fa-trash-arrow-up"></i> Recycle Bin
          </button>
          <span id="recyclePolicyNote" style="font-size:0.9rem; color:var(--gray-600);">
            Deleted items are auto-purged after 30 days.
          </span>
        </div>
        <div style="font-size:0.9rem; color:var(--gray-700); font-weight:600;" id="currentDataMode">Viewing: Active Data</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title"><i class="fas fa-list"></i> Recent Scans</h2>
        <div class="controls">
          <button id="btnDeleteSelectedScans" class="btn btn-subtle-danger">
            <i class="fas fa-trash"></i> Delete Selected
          </button>
          <button id="btnPurgeAllData" class="btn btn-danger">
            <i class="fas fa-trash-can"></i> Purge All Data
          </button>
        </div>
      </div>
      <div class="card-body">
        <div class="filters">
          <div class="filter-group">
            <label for="limit">Results per page</label>
            <select id="limit" class="filter-control">
              <option value="10">10</option>
              <option value="25">25</option>
              <option value="50" selected>50</option>
              <option value="100">100</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="fVehicleType">Vehicle Type</label>
            <select id="fVehicleType" class="filter-control">
              <option value="">All types</option>
              <option value="Car">Car</option>
              <option value="Motorcycle">Motorcycle</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="fVehicle">Vehicle ID</label>
            <input type="text" id="fVehicle" placeholder="Enter vehicle ID" class="filter-control">
          </div>
          <div class="filter-group">
            <label for="fModel">Tire Model Code</label>
            <input type="text" id="fModel" placeholder="Enter tire model" class="filter-control">
          </div>
          <div class="filter-group">
            <label for="fTire">Tire Position</label>
            <select id="fTire" class="filter-control">
              <option value="">All positions</option>
              <optgroup label="Car">
                <option>FL</option>
                <option>FR</option>
                <option>RL</option>
                <option>RR</option>
                <option>SPARE</option>
              </optgroup>
              <optgroup label="Motorcycle">
                <option>F (Front)</option>
                <option>R (Rear)</option>
              </optgroup>
            </select>
          </div>
          <div class="filter-group">
            <label for="fTireType">Tire Type</label>
            <select id="fTireType" class="filter-control">
              <option value="">All tire types</option>
              <option>All-Season Tires</option>
              <option>Summer Tires</option>
              <option>Winter/Snow Tires</option>
              <option>All-Terrain Tires</option>
              <option>Performance Tires</option>
              <option>Touring Tires</option>
              <option>Mud-Terrain Tires</option>
              <option>Run-Flat Tires</option>
              <option>Competition Tires</option>
              <option>Eco-Friendly Tires</option>
              <option>Spare Tires</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="fTreadDesign">Tread Design</label>
            <select id="fTreadDesign" class="filter-control">
              <option value="">All designs</option>
              <option>Symmetrical</option>
              <option>Asymmetrical</option>
              <option>Directional</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="fVerdict">Verdict</label>
            <select id="fVerdict" class="filter-control">
              <option value="">All verdicts</option>
              <option>GOOD</option>
              <option>WARNING</option>
              <option>REPLACE</option>
            </select>
          </div>
        </div>

        <table id="scanTable">
          <thead>
            <tr>
              <th><input type="checkbox" id="chkAllScans" title="Select all"></th>
              <th>Timestamp</th>
              <th>Verdict</th>
              <th>Score</th>
              <th>Veh. Type</th>
              <th>Vehicle</th>
              <th>Tire Model</th>
              <th>Tire</th>
              <th>Tire Type</th>
              <th>Tread Design</th>
              <th>Operator</th>
              <th>Brightness</th>
              <th>Sharpness</th>
              <th>Edge Density</th>
              <th>Continuity</th>
              <th>PSI Measured</th>
              <th>PSI Recommended</th>
              <th>PSI Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2 class="card-title"><i class="fas fa-tire"></i> Scan Details</h2>
      </div>
      <div class="card-body">
        <div class="verdict-container">
          <div>
            <div class="info-label">Selected Scan</div>
            <div class="info-value" id="selTs">Select a scan from the list</div>
          </div>
          <div class="scan-verdict" id="selVerdict">—</div>
        </div>

        <div class="scan-info">
          <div class="info-item">
            <div class="info-label">Vehicle Type</div>
            <div class="info-value" id="selVehicleType">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Vehicle ID</div>
            <div class="info-value" id="selVehicle">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Tire Model Code</div>
            <div class="info-value" id="selTireModel">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Tire Position</div>
            <div class="info-value" id="selTire">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Tire Type</div>
            <div class="info-value" id="selTireType">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Tread Design</div>
            <div class="info-value" id="selTreadDesign">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Operator</div>
            <div class="info-value" id="selOperator">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Timestamp</div>
            <div class="info-value" id="selTimestamp">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Brightness</div>
            <div class="info-value" id="selBrightness">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Sharpness</div>
            <div class="info-value" id="selSharpness">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Edge Density</div>
            <div class="info-value" id="selEdgeDensity">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Continuity</div>
            <div class="info-value" id="selContinuity">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Tread Verdict</div>
            <div class="info-value" id="selTreadVerdict">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Capture Quality</div>
            <div class="info-value" id="selQualityVerdict">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">PSI Measured</div>
            <div class="info-value" id="selPsiMeasured">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">PSI Recommended</div>
            <div class="info-value" id="selPsiRecommended">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">PSI Status</div>
            <div class="info-value" id="selPsiStatus">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">PSI Delta</div>
            <div class="info-value" id="selPsiGap">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Groove Score</div>
            <div class="info-value" id="selScore">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Depth Est. (mm)</div>
            <div class="info-value" id="selDepthEst">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Raw Score Verdict</div>
            <div class="info-value" id="selRawVerdict">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Notes</div>
            <div class="info-value" id="selNotes">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">MM per Pixel</div>
            <div class="info-value" id="selMmPerPx">—</div>
          </div>
        </div>

        <div class="image-grid" id="imgs">
          <div class="image-card">
            <div class="image-container">
              <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%23e0e0e0'/%3E%3Ctext x='50' y='50' font-family='Arial' font-size='12' fill='%23757575' text-anchor='middle' dominant-baseline='middle'%3ENo image%3C/text%3E%3C/svg%3E" alt="No image">
            </div>
            <div class="image-label">Select a scan to view images</div>
          </div>
        </div>

        <div class="pass-rate">
          <div>
            <div class="info-label">Overall Pass Rate</div>
            <div class="pass-rate-value" id="passRate">0%</div>
          </div>
          <div class="info-value" id="passCount">0/0</div>
        </div>

        <div class="histogram-container">
          <canvas id="hist"></canvas>
        </div>

        <div class="details-panel">
          <pre id="details">Select a scan to view detailed information</pre>
        </div>
      </div>
    </div>

    <!-- ✅ Device Validation Card -->
    <div class="card" style="grid-column: 1 / -1;">
      <div class="card-header">
        <h2 class="card-title"><i class="fas fa-balance-scale"></i> Device Validation (Table 3.2)</h2>
        <div class="controls">
          <button class="btn btn-subtle-danger" id="btnDeleteSelectedVal">
            <i class="fas fa-trash"></i> Delete Selected
          </button>
          <button class="btn btn-outline" id="btnExportValidation">
            <i class="fas fa-file-csv"></i> Export Table 3.2
          </button>
          <button class="btn btn-outline" id="btnRefreshVal">
            <i class="fas fa-sync-alt"></i> Refresh
          </button>
        </div>
      </div>
      <div class="card-body">

        <div class="validation-layout">
          <div class="guide-card">
            <h3><i class="fas fa-ruler-combined"></i> Industry Tread Measurement Guide</h3>
            <p>
              Use this reference when filling out Table 3.2 so the manual gauge reading follows the same tread thresholds used by the app.
              Measure in millimeters from the main groove and compare that lowest stable reading against the device result.
            </p>

            <table class="guide-table">
              <thead>
                <tr>
                  <th>Vehicle</th>
                  <th>Replace</th>
                  <th>Warning</th>
                  <th>Good</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Car</strong><br><span style="color:var(--gray-600);">Code-aligned threshold</span></td>
                  <td><span class="depth-chip replace">&lt; __CAR_MIN_MM__ mm</span></td>
                  <td><span class="depth-chip warning">__CAR_MIN_MM__ to &lt; __CAR_WARN_UPPER_MM__ mm</span></td>
                  <td><span class="depth-chip good">>= __CAR_WARN_UPPER_MM__ mm</span></td>
                </tr>
                <tr>
                  <td><strong>Motorcycle</strong><br><span style="color:var(--gray-600);">Code-aligned threshold</span></td>
                  <td><span class="depth-chip replace">&lt; __MOTO_MIN_MM__ mm</span></td>
                  <td><span class="depth-chip warning">__MOTO_MIN_MM__ to &lt; __MOTO_WARN_UPPER_MM__ mm</span></td>
                  <td><span class="depth-chip good">>= __MOTO_WARN_UPPER_MM__ mm</span></td>
                </tr>
              </tbody>
            </table>

            <ol class="guide-steps">
              <li>Clean the groove and remove trapped stones or debris before inserting the tread-depth gauge.</li>
              <li>Place the gauge base flat on the tire and read from the main circumferential groove, not the tread block edge.</li>
              <li>Check inner, center, and outer groove positions, then record the lowest stable millimeter value for Table 3.2.</li>
              <li>Match the manual reading to the exact scan timestamp selected above so the validation entry compares like-for-like data.</li>
            </ol>

            <div class="guide-note">
              Current warning bands in code: Car +__CAR_WARN_MM__ mm and Motorcycle +__MOTO_WARN_MM__ mm above the legal minimum before the verdict changes from WARNING to GOOD.
              Validation pass gate: % diff ≤ __VAL_PCT_MAX__% and abs error ≤ __VAL_ABS_MAX__ mm with class_match=True.
              Millimeter validation requires a calibrated score_model.
            </div>
          </div>

          <!-- ✅ Submit Validation Form -->
          <div id="valForm" style="background:#f0f4ff; border:1px solid #c7d7ff; border-radius:12px; padding:1.25rem; margin-bottom:0;">
            <h3 style="font-size:1rem; font-weight:700; color:#1d4ed8; margin-bottom:1rem;">
              <i class="fas fa-plus-circle"></i> Submit New Validation Entry
            </h3>
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; align-items:end;">
              <div class="filter-group">
                <label>Scan Timestamp</label>
                <input type="text" id="valScanTs" placeholder="e.g. 20260207_152411" class="filter-control" style="background:white;">
              </div>
              <div class="filter-group">
                <label>Tire ID</label>
                <input type="text" id="valTireIdInput" placeholder="e.g. TEST-01" class="filter-control" style="background:white;">
              </div>
              <div class="filter-group">
                <label>Manual Depth (mm)</label>
                <input type="number" id="valManualDepth" placeholder="e.g. 3.5" step="0.1" min="0" max="15" class="filter-control" style="background:white;">
              </div>
              <div>
                <button id="btnSubmitVal" class="btn" style="background:#1d4ed8; color:white; width:100%; justify-content:center; padding:0.7rem 1rem;">
                  <i class="fas fa-check-circle"></i> Submit Validation
                </button>
              </div>
            </div>
            <div id="valFormResult" style="margin-top:0.75rem; display:none; padding:0.75rem; border-radius:8px; font-weight:600; font-size:0.9rem;"></div>
            <p style="margin-top:0.75rem; font-size:0.78rem; color:#6c757d;">
              💡 Tip: Click a scan row above to auto-fill the Scan Timestamp field.
            </p>
          </div>
        </div>

        <!-- Filters -->
        <div class="filters">
          <div class="filter-group">
            <label for="fValTire">Filter by Tire ID</label>
            <input type="text" id="fValTire" placeholder="Filter by tire ID" class="filter-control">
          </div>
          <div class="filter-group">
            <label for="fValVerdict">Filter by Verdict</label>
            <select id="fValVerdict" class="filter-control">
              <option value="">All</option>
              <option value="PASS">PASS</option>
              <option value="FAIL">FAIL</option>
            </select>
          </div>
        </div>

        <table id="valTable">
          <thead>
            <tr>
              <th><input type="checkbox" id="chkAllVal" title="Select all"></th>
              <th>Timestamp</th>
              <th>Tire ID</th>
              <th>Manual (mm)</th>
              <th>Device (mm)</th>
              <th>% Diff</th>
              <th>Error (mm)</th>
              <th>Time (s)</th>
              <th>Tread Class</th>
              <th>Verdict</th>
              <th>Notes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="valTbody"></tbody>
        </table>

        <div id="valSummary" style="margin-top:1rem; padding:1rem; background:#f8f9fa; border-radius:12px; display:none;">
          <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1rem; text-align:center;">
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">TRIALS</div>
              <div id="valCount" style="font-size:1.4rem; font-weight:800; color:var(--gray-800);">0</div>
            </div>
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">AVG % DIFF</div>
              <div id="valAvgPct" style="font-size:1.4rem; font-weight:800; color:#1d4ed8;">—</div>
            </div>
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">MAX ERROR (mm)</div>
              <div id="valMaxErr" style="font-size:1.4rem; font-weight:800; color:#dc2626;">—</div>
            </div>
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">PASS RATE</div>
              <div id="valPassRate" style="font-size:1.4rem; font-weight:800; color:#16a34a;">—</div>
            </div>
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">OVERALL</div>
              <div id="valOverall" style="font-size:1.4rem; font-weight:800;">—</div>
            </div>
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">CLASS MATCH</div>
              <div id="valClassMatch" style="font-size:1.4rem; font-weight:800; color:#7c3aed;">—</div>
            </div>
            <div>
              <div style="font-size:0.75rem; color:var(--gray-600); font-weight:600;">AVG ERROR (mm)</div>
              <div id="valAvgErr" style="font-size:1.4rem; font-weight:800; color:#d97706;">—</div>
            </div>
          </div>
          <div style="font-size:0.7rem; color:#9ca3af; text-align:center; margin-top:0.6rem; border-top:1px solid #e9ecef; padding-top:0.5rem;">
            ✓ <strong>PASS criteria:</strong> % diff ≤ __VAL_PCT_MAX__% &amp;&amp; error ≤ __VAL_ABS_MAX__ mm &amp;&amp; tread class matches manual gauge
          </div>

          <div style="margin-top:1rem; border-top:1px solid #e5e7eb; padding-top:0.8rem;">
            <div style="font-size:0.9rem; font-weight:700; color:#334155; margin-bottom:0.5rem;">
              Thesis Summary Table
            </div>
            <table id="thesisSummaryTable" style="width:100%; border-collapse:collapse; background:white; border:1px solid #e5e7eb; border-radius:10px; overflow:hidden;">
              <thead>
                <tr style="background:#f1f5f9;">
                  <th style="text-align:left; padding:0.55rem; font-size:0.78rem; color:#475569;">Metric</th>
                  <th style="text-align:left; padding:0.55rem; font-size:0.78rem; color:#475569;">Value</th>
                  <th style="text-align:left; padding:0.55rem; font-size:0.78rem; color:#475569;">Target</th>
                  <th style="text-align:left; padding:0.55rem; font-size:0.78rem; color:#475569;">Status</th>
                </tr>
              </thead>
              <tbody id="thesisSummaryBody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="modal" id="imgModal">
    <div class="modal-content">
      <img src="" alt="Scan preview" class="modal-img" id="modalImg">
      <div class="modal-close" id="modalClose">
        <i class="fas fa-times"></i>
      </div>
    </div>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const VAL_PCT_PASS_MAX = __VAL_PCT_MAX__;
    const VAL_ABS_PASS_MAX = __VAL_ABS_MAX__;
    const VAL_PCT_WARN_MAX = __VAL_PCT_WARN__;
    const VAL_ABS_WARN_MAX = __VAL_ABS_WARN__;
    const status = (t) => {
      $("status").textContent = t;
      $("status").parentElement.classList.remove('fade-in');
      void $("status").offsetWidth; // Trigger reflow
      $("status").parentElement.classList.add('fade-in');
    };

    function verdictClass(v) {
      v = (v || "").toUpperCase();
      if (v === "GOOD" || v === "OK") return "verdict-good";
      if (v === "ADVISORY") return "verdict-warning";
      if (v === "WARNING" || v.includes("WARN")) return "verdict-warning";
      if (v === "REPLACE") return "verdict-replace";
      if (v === "PASS") return "verdict-pass";
      if (v === "PENDING") return "verdict-warning";
      if (v === "FAIL") return "verdict-fail";
      return "verdict-fail";
    }

    // Safe numeric formatter for nullable/string values
    function nfmt(v, digits = 2, fallback = "—") {
      const n = Number(v);
      return Number.isFinite(n) ? n.toFixed(digits) : fallback;
    }

    function recycleMode() {
      return !!$("showRecycleBin")?.checked;
    }

    function updateRecycleUi() {
      const inRecycle = recycleMode();
      if ($("btnViewActiveData")) {
        $("btnViewActiveData").className = `btn ${inRecycle ? 'btn-outline' : 'btn-primary'}`;
      }
      if ($("btnViewRecycleBin")) {
        $("btnViewRecycleBin").className = `btn ${inRecycle ? 'btn-primary' : 'btn-outline'}`;
      }
      if ($("currentDataMode")) {
        $("currentDataMode").textContent = `Viewing: ${inRecycle ? 'Recycle Bin' : 'Active Data'}`;
      }
      $("btnDeleteSelectedScans").innerHTML = inRecycle
        ? '<i class="fas fa-rotate-left"></i> Restore Selected'
        : '<i class="fas fa-trash"></i> Delete Selected';
      $("btnDeleteSelectedVal").innerHTML = inRecycle
        ? '<i class="fas fa-rotate-left"></i> Restore Selected'
        : '<i class="fas fa-trash"></i> Delete Selected';
      $("btnPurgeAllData").innerHTML = inRecycle
        ? '<i class="fas fa-trash-can"></i> Empty Recycle Bin'
        : '<i class="fas fa-box-archive"></i> Move All To Recycle Bin';
    }

    let currentSelectedScanTs = null;
    let lastFilteredScans = [];

    async function api(path) {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      return res;
    }

    function applyFilters(items) {
      const v = $("fVehicle").value.trim().toLowerCase();
      const vt = $("fVehicleType").value.trim().toLowerCase();
      const model = $("fModel").value.trim().toLowerCase();
      const t = $("fTire").value.trim().toUpperCase();
      const tt = $("fTireType").value.trim().toLowerCase();
      const td = $("fTreadDesign").value.trim().toLowerCase();
      const vd = $("fVerdict").value.trim().toUpperCase();
      return items.filter(r => {
        const rv = (r.vehicle_id ?? r["vehicle_id"] ?? "").toString().toLowerCase();
        const rvt = (r.vehicle_type ?? r["vehicle_type"] ?? "").toString().toLowerCase();
        const rmodel = (r.tire_model_code ?? r["tire_model_code"] ?? "").toString().toLowerCase();
        const rt = (r.tire_position ?? r["tire_position"] ?? "").toString().toUpperCase();
        const rtt = (r.tire_type ?? r["tire_type"] ?? "").toString().toLowerCase();
        const rtd = (r.tread_design ?? r["tread_design"] ?? "").toString().toLowerCase();
        const rvd = (r.verdict ?? r["verdict"] ?? "").toString().toUpperCase();
        if (v && !rv.includes(v)) return false;
        if (vt && rvt !== vt) return false;
        if (model && !rmodel.includes(model)) return false;
        if (t && rt !== t) return false;
        if (tt && rtt !== tt) return false;
        if (td && rtd !== td) return false;
        if (vd && rvd !== vd) return false;
        return true;
      });
    }

    function drawHistogram(scores) {
      const c = $("hist");
      const ctx = c.getContext("2d");
      const w = c.width = c.clientWidth * 2;
      const h = c.height = c.clientHeight * 2;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      const bins = 10;
      const counts = new Array(bins).fill(0);
      scores.forEach(s => {
        const v = Math.max(0, Math.min(0.999, Number(s) || 0));
        const i = Math.floor(v * bins);
        counts[i]++;
      });
      const max = Math.max(1, ...counts);
      const pad = 20;
      const barW = (w - pad * 2) / bins;
      counts.forEach((c, i) => {
        const bh = (c / max) * (h - pad * 2);
        const x = pad + i * barW;
        const y = h - pad - bh;
        ctx.fillStyle = "rgba(13, 110, 253, 0.7)";
        ctx.fillRect(x + 4, y, barW - 8, bh);
      });
    }

    async function loadScans() {
      status("Loading scans...");
      const limit = Number($("limit").value || 50);
      try {
        const query = new URLSearchParams();
        query.set("limit", String(limit));

        const qVehicleType = $("fVehicleType").value.trim();
        const qTire = $("fTire").value.trim();
        const qVerdict = $("fVerdict").value.trim();
        const qTireType = $("fTireType").value.trim();
        const qTreadDesign = $("fTreadDesign").value.trim();

        // Send exact-match filters to API; keep text contains filters client-side.
        if (qVehicleType) query.set("vehicle_type", qVehicleType);
        if (qTire) query.set("tire_position", qTire);
        if (qVerdict) query.set("verdict", qVerdict);
        if (qTireType) query.set("tire_type", qTireType);
        if (qTreadDesign) query.set("tread_design", qTreadDesign);
        if (recycleMode()) query.set("only_deleted", "true");

        const res = await api(`/api/scans?${query.toString()}`);
        const data = await res.json();
        const items = Array.isArray(data.items) ? data.items : [];
        const filtered = applyFilters(items);
        lastFilteredScans = filtered;
        const tbody = $("tbody");
        tbody.innerHTML = "";

        if (filtered.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="19" style="text-align: center; padding: 2rem; color: #6c757d;">
                <i class="fas fa-search" style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.3;"></i>
                <p>${recycleMode() ? 'Recycle bin is empty' : 'No scans match your criteria'}</p>
                <button id="resetFilters" class="btn btn-outline" style="margin-top: 1rem;">
                  <i class="fas fa-undo"></i> Reset Filters
                </button>
              </td>
            </tr>
          `;
          $("resetFilters").onclick = () => {
            $("fVehicleType").value = "";
            $("fVehicle").value = "";
            $("fModel").value = "";
            $("fTire").value = "";
            $("fTireType").value = "";
            $("fTreadDesign").value = "";
            $("fVerdict").value = "";
            loadScans();
          };
        } else {
          filtered.forEach((row, i) => {
            const tr = document.createElement("tr");
            tr.onclick = () => selectScan(row.ts || row["ts"] || `scan-${i}`);

            const get = (key, fallback = "-") => {
              return (row[key] !== undefined && row[key] !== null) 
                ? String(row[key]) 
                : (row[key.toLowerCase()] !== undefined && row[key.toLowerCase()] !== null)
                  ? String(row[key.toLowerCase()]) 
                  : fallback;
            };

            const ts = get("ts", "-");
            const verdict = get("verdict", "—");
            const score = get("score", "-");
            const vehicleType = get("vehicle_type", "-");
            const vehicle = get("vehicle_id", "-");
            const tireModel = get("tire_model_code", "-");
            const tire = get("tire_position", "-");
            const tireType = get("tire_type", "-");
            const treadDesign = get("tread_design", "-");
            const operator = get("operator", "-");
            const brightness = get("brightness", "-");
            const sharpness = get("sharpness", "-");
            const edgeDensity = get("edge_density", "-");
            const continuity = get("continuity", "-");
            const psiMeasured = row.psi_measured == null ? "—" : Number(row.psi_measured).toFixed(1);
            const psiRec = row.psi_recommended == null ? "—" : Number(row.psi_recommended).toFixed(1);
            const psiStatus = get("psi_status", "—");
            const rowTs = ts;
            const actionCell = recycleMode()
              ? `<td>
                  <button class="btn btn-outline btn-restore-scan" data-ts="${rowTs}" style="padding:0.35rem 0.55rem;"><i class="fas fa-rotate-left"></i></button>
                  <button class="btn btn-subtle-danger btn-hard-del-scan" data-ts="${rowTs}" style="padding:0.35rem 0.55rem;"><i class="fas fa-trash-can"></i></button>
                </td>`
              : `<td><button class="btn btn-subtle-danger btn-del-scan" data-ts="${rowTs}" style="padding:0.35rem 0.55rem;"><i class="fas fa-trash"></i></button></td>`;

            tr.innerHTML = `
              <td><input type="checkbox" class="scan-chk" data-ts="${rowTs}"></td>
              <td>${ts}</td>
              <td><span class="verdict-badge ${verdictClass(verdict)}">${verdict}</span></td>
              <td>${score}</td>
              <td>${vehicleType}</td>
              <td>${vehicle}</td>
              <td>${tireModel}</td>
              <td>${tire}</td>
              <td>${tireType}</td>
              <td>${treadDesign}</td>
              <td>${operator}</td>
              <td>${brightness}</td>
              <td>${sharpness}</td>
              <td>${edgeDensity}</td>
              <td>${continuity}</td>
              <td>${psiMeasured}</td>
              <td>${psiRec}</td>
              <td><span class="verdict-badge ${verdictClass(psiStatus)}">${psiStatus}</span></td>
              ${actionCell}
            `;
            tbody.appendChild(tr);
          });

          tbody.querySelectorAll(".scan-chk, .btn-del-scan").forEach((el) => {
            el.addEventListener("click", (e) => e.stopPropagation());
          });
          tbody.querySelectorAll(".btn-del-scan").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const ts = e.currentTarget.getAttribute("data-ts");
              await deleteSingleScan(ts);
            });
          });
          tbody.querySelectorAll(".btn-restore-scan").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const ts = e.currentTarget.getAttribute("data-ts");
              await restoreSingleScan(ts);
            });
          });
          tbody.querySelectorAll(".btn-hard-del-scan").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const ts = e.currentTarget.getAttribute("data-ts");
              await hardDeleteSingleScan(ts);
            });
          });
        }

        const scores = filtered.map(r => parseFloat(r.score) || 0);
        drawHistogram(scores);

        const passCount = filtered.filter(r => (r.verdict || "").toUpperCase().startsWith("GOOD")).length;
        const rate = filtered.length ? Math.round((passCount / filtered.length) * 100) : 0;
        $("passRate").textContent = `${rate}%`;
        $("passCount").textContent = `${passCount}/${filtered.length}`;

        status(`Loaded ${filtered.length} scans`);
      } catch (error) {
        status(`Error: ${error.message}`);
        console.error("Scan list error:", error);
      }
    }

    async function deleteSingleScan(ts) {
      if (!ts) return;
      if (!confirm(`Move scan ${ts} to recycle bin?`)) return;
      try {
        const res = await fetch(`/api/scans/${encodeURIComponent(ts)}`, { method: "DELETE" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Delete failed (${res.status})`);
        if (currentSelectedScanTs === ts) {
          currentSelectedScanTs = null;
          $("selTs").textContent = "Select a scan from the list";
          $("selVerdict").textContent = "—";
          $("selQualityVerdict").textContent = "—";
          $("selPsiGap").textContent = "—";
          $("selScore").textContent = "—";
          $("selDepthEst").textContent = "—";
          $("selRawVerdict").textContent = "—";
        }
        await loadScans();
        await loadValidation();
        status(`Moved scan ${ts} to recycle bin`);
      } catch (err) {
        status(`Delete error: ${err.message}`);
      }
    }

    async function restoreSingleScan(ts) {
      if (!ts) return;
      try {
        const res = await fetch(`/api/scans/${encodeURIComponent(ts)}/restore`, { method: "POST" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Restore failed (${res.status})`);
        await loadScans();
        await loadValidation();
        status(`Restored scan ${ts}`);
      } catch (err) {
        status(`Restore error: ${err.message}`);
      }
    }

    async function hardDeleteSingleScan(ts) {
      if (!ts) return;
      if (!confirm(`Permanently delete scan ${ts} and its files? This cannot be undone.`)) return;
      try {
        const res = await fetch(`/api/scans/${encodeURIComponent(ts)}/hard?delete_files=true`, { method: "DELETE" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Hard delete failed (${res.status})`);
        await loadScans();
        await loadValidation();
        status(`Permanently deleted scan ${ts}`);
      } catch (err) {
        status(`Hard delete error: ${err.message}`);
      }
    }

    async function deleteSelectedScans() {
      const tsList = Array.from(document.querySelectorAll(".scan-chk:checked")).map((x) => x.getAttribute("data-ts"));
      if (!tsList.length) {
        alert("No scans selected.");
        return;
      }
      const inRecycle = recycleMode();
      if (!confirm(inRecycle
        ? `Restore ${tsList.length} selected scan(s)?`
        : `Move ${tsList.length} selected scan(s) to recycle bin?`)) return;
      try {
        const res = await fetch(inRecycle ? "/api/scans/restore-batch" : "/api/scans/delete-batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ts_list: tsList }),
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `${inRecycle ? 'Batch restore' : 'Batch delete'} failed (${res.status})`);
        if (currentSelectedScanTs && tsList.includes(currentSelectedScanTs)) {
          currentSelectedScanTs = null;
          $("selTs").textContent = "Select a scan from the list";
          $("selVerdict").textContent = "—";
          $("selQualityVerdict").textContent = "—";
          $("selPsiGap").textContent = "—";
          $("selScore").textContent = "—";
          $("selDepthEst").textContent = "—";
          $("selRawVerdict").textContent = "—";
        }
        await loadScans();
        await loadValidation();
        status(inRecycle
          ? `Restored ${payload.restored || tsList.length} scan(s)`
          : `Moved ${payload.deleted || tsList.length} scan(s) to recycle bin`);
      } catch (err) {
        status(`${inRecycle ? 'Batch restore' : 'Batch delete'} error: ${err.message}`);
      }
    }

    async function purgeAllData() {
      const inRecycle = recycleMode();
      if (!confirm(inRecycle
        ? "Permanently delete all items currently in the recycle bin? This cannot be undone."
        : "Move all active scans and validation rows to the recycle bin?")) return;
      try {
        const res = await fetch(
          inRecycle
            ? "/api/data/purge?include_validation=true&delete_files=true&only_deleted=true"
            : "/api/data/recycle-all?include_validation=true",
          { method: "POST" }
        );
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Purge failed (${res.status})`);
        currentSelectedScanTs = null;

        // After moving active data, switch to recycle bin so users can
        // immediately verify where the records went.
        if (!inRecycle) {
          $("showRecycleBin").checked = true;
          updateRecycleUi();
        }

        await loadScans();
        await loadValidation();
        status(inRecycle
          ? `Recycle bin emptied: ${payload.results_deleted || 0} scans, ${payload.validation_deleted || 0} validation rows`
          : `Moved to recycle bin: ${payload.results_deleted || 0} scans, ${payload.validation_deleted || 0} validation rows`);
      } catch (err) {
        status(`Purge error: ${err.message}`);
      }
    }

    async function selectScan(ts) {
      currentSelectedScanTs = ts;
      $("selTs").textContent = ts;
      status("Loading scan details...");
      
      try {
        const suffix = recycleMode() ? "?include_deleted=true" : "";
        const res = await api(`/api/scans/${encodeURIComponent(ts)}${suffix}`);
        const row = await res.json();

        const verdict = row.verdict || "";
        $("selVerdict").className = `scan-verdict ${verdictClass(verdict)}`;
        $("selVerdict").textContent = verdict || "—";
        $("selVerdict").style.display = verdict ? 'inline-block' : 'none';

        $("selVehicleType").textContent = row.vehicle_type || "—";
        $("selVehicle").textContent = row.vehicle_id || "—";
        $("selTireModel").textContent = row.tire_model_code || "—";
        $("selTire").textContent = row.tire_position || "—";
        $("selTireType").textContent = row.tire_type || "—";
        $("selTreadDesign").textContent = row.tread_design || "—";
        $("selOperator").textContent = row.operator || "—";
        $("selTimestamp").textContent = row.ts || "—";
        $("selBrightness").textContent = nfmt(row.brightness, 2);
        $("selSharpness").textContent = nfmt(row.sharpness, 2);
        $("selEdgeDensity").textContent = nfmt(row.edge_density, 2);
        $("selContinuity").textContent = nfmt(row.continuity, 2);

        const treadVerdict = row.tread_verdict || "—";
        const qualityVerdict = row.quality_verdict || "—";
        const psiMeasured = (typeof row.psi_measured === "number")
          ? row.psi_measured.toFixed(1)
          : (row.psi_measured ?? "—");
        const psiRecommended = (typeof row.psi_recommended === "number")
          ? row.psi_recommended.toFixed(1)
          : (row.psi_recommended ?? "—");
        const psiStatus = row.psi_status || "—";

        $("selTreadVerdict").innerHTML = `<span class="verdict-badge ${verdictClass(treadVerdict)}">${treadVerdict}</span>`;
        $("selQualityVerdict").innerHTML = `<span class="verdict-badge ${verdictClass(qualityVerdict)}">${qualityVerdict}</span>`;
        $("selPsiMeasured").textContent = psiMeasured;
        $("selPsiRecommended").textContent = psiRecommended;
        $("selPsiStatus").innerHTML = `<span class="verdict-badge ${verdictClass(psiStatus)}">${psiStatus}</span>`;

        // PSI Delta — red when critically low, amber when slightly low, green OK
        if (typeof row.psi_measured === "number" && typeof row.psi_recommended === "number") {
          const gap = row.psi_measured - row.psi_recommended;
          const gapClr = gap < -4 ? "#dc2626" : gap < 0 ? "#d97706" : "#16a34a";
          $("selPsiGap").innerHTML = `<span style="color:${gapClr};font-weight:700;">${gap >= 0 ? "+" : ""}${gap.toFixed(1)} PSI</span>`;
        } else {
          $("selPsiGap").textContent = "—";
        }
        // Groove score, estimated depth, and raw verdict before PSI override
        $("selScore").textContent = typeof row.score === "number" ? row.score.toFixed(4) : "—";
        $("selDepthEst").textContent = typeof row.device_depth_mm === "number"
          ? row.device_depth_mm.toFixed(2) + " mm"
          : "—";
        const rawVerdict = row.raw_score_verdict || "";
        $("selRawVerdict").innerHTML = rawVerdict ? `<span class="verdict-badge ${verdictClass(rawVerdict)}">${rawVerdict}</span>` : "—";

        $("selNotes").textContent = row.notes || row["notes"] || "—";
        $("selMmPerPx").textContent = nfmt(row.mm_per_px, 6);

        $("details").textContent = JSON.stringify(row, null, 2);

        const imgs = $("imgs");
        imgs.innerHTML = "";
        
        try {
          const res2 = await api(`/api/scans/${encodeURIComponent(ts)}/images`);
          const d2 = await res2.json();
          const keys = Object.keys(d2.images || {});
          if (keys.length === 0) throw new Error("no images");
          
          const prefer = ["gray", "edges_closed", "edges", "original"];
          const ordered = [...new Set([...prefer, ...keys])].filter(k => keys.includes(k));

          ordered.slice(0, 4).forEach(kind => {
            const box = document.createElement("div");
            box.className = "image-card";
            const src = `/api/images/${encodeURIComponent(ts)}/${encodeURIComponent(kind)}`;
            box.innerHTML = `
              <div class="image-container">
                <img data-src="${src}" src="${src}" alt="${kind} preview" />
              </div>
              <div class="image-label">${kind}</div>
            `;
            box.querySelector("img").onclick = (e) => {
              $("modalImg").src = e.target.getAttribute("data-src");
              $("imgModal").classList.add("active");
            };
            imgs.appendChild(box);
          });
        } catch (e) {
          imgs.innerHTML = `
            <div class="image-card" style="border: 1px dashed #adb5bd; background: #f8f9fa;">
              <div class="image-container" style="display: flex; align-items: center; justify-content: center; background: #f0f2f5;">
                <i class="fas fa-image" style="font-size: 2.5rem; color: #adb5bd;"></i>
              </div>
              <div class="image-label" style="text-align: center; color: #6c757d;">No images available</div>
            </div>
          `;
        }

        status("Scan details loaded");
      } catch (error) {
        status(`Error: ${error.message}`);
        console.error("Scan detail error:", error);
      }
    }

    async function loadValidation() {
      status("Loading validation data...");
      const tireId = $("fValTire").value.trim() || undefined;
      const verdict = $("fValVerdict").value.trim() || undefined;

      const parseValidationClasses = (notes) => {
        const text = String(notes || "");
        let m = text.match(/manual_class=(\\w+).*?device_class=(\\w+)/i);
        if (m) return { manual: String(m[1]).toUpperCase(), device: String(m[2]).toUpperCase() };
        m = text.match(/Manual\\s*Class:\\s*(\\w+).*?Device\\s*Class:\\s*(\\w+)/i);
        if (m) return { manual: String(m[1]).toUpperCase(), device: String(m[2]).toUpperCase() };
        return null;
      };

      const statusChip = (ok, na = false) => {
        if (na) return '<span style="font-size:0.72rem; font-weight:700; color:#64748b;">N/A</span>';
        const color = ok ? "#16a34a" : "#dc2626";
        const label = ok ? "PASS" : "CHECK";
        return `<span style="font-size:0.72rem; font-weight:700; color:${color};">${label}</span>`;
      };

      const thesisRow = (metric, value, target, statusHtml) => `
        <tr>
          <td style="padding:0.5rem 0.55rem; border-top:1px solid #f1f5f9; font-size:0.8rem; color:#0f172a;">${metric}</td>
          <td style="padding:0.5rem 0.55rem; border-top:1px solid #f1f5f9; font-size:0.8rem; color:#1f2937;">${value}</td>
          <td style="padding:0.5rem 0.55rem; border-top:1px solid #f1f5f9; font-size:0.8rem; color:#475569;">${target}</td>
          <td style="padding:0.5rem 0.55rem; border-top:1px solid #f1f5f9;">${statusHtml}</td>
        </tr>
      `;
      
      try {
        const query = new URLSearchParams();
        query.append("limit", 100);
        if (tireId) query.append("tire_id", tireId);
        if (verdict) query.append("verdict", verdict);
        if (recycleMode()) query.append("only_deleted", "true");
        
        const res = await api(`/api/validation?${query}`);
        const data = await res.json();
        const items = Array.isArray(data.items) ? data.items : [];
        const tbody = $("valTbody");
        tbody.innerHTML = "";

        if (items.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="12" style="text-align: center; padding: 2rem; color: #6c757d;">
                ${recycleMode() ? 'Validation recycle bin is empty' : 'No validation results found'}
              </td>
            </tr>
          `;
          $("valSummary").style.display = "none";
          $("thesisSummaryBody").innerHTML = "";
        } else {
          items.forEach((row) => {
            const tr = document.createElement("tr");
            const rowId = row.id;
            const ts         = row.ts || row.created_at || "—";
            const tireId     = row.tire_id || "—";
            const manualDepth  = typeof row.manual_depth  === "number" ? row.manual_depth.toFixed(2)  : "—";
            const deviceDepth  = typeof row.device_depth  === "number" ? row.device_depth.toFixed(2)  : "—";
            const percentDiff  = typeof row.percent_diff  === "number" ? row.percent_diff.toFixed(2)  : "—";
            const absErr       = typeof row.abs_error_mm  === "number" ? row.abs_error_mm.toFixed(3)  : "—";
            const procTime     = typeof row.processing_time === "number" ? row.processing_time.toFixed(2) : "—";
            const verd         = row.verdict || "—";
            const notes        = row.notes || "—";

            // Tread class from notes: manual_class=X device_class=Y class_match=Z
            const classM = notes.match(/manual_class=(\\w+).*?device_class=(\\w+).*?class_match=(\\w+)/);
            const treadClassHtml = classM ? (() => {
              const mc = classM[1], dc = classM[2], matched = classM[3].toLowerCase() === "true";
              const clr = matched ? "#16a34a" : "#dc2626";
              return `<span style="font-size:0.75rem;font-weight:700;color:${clr};" title="manual=${mc} device=${dc}">${mc}&thinsp;/&thinsp;${dc}</span>`;
            })() : "—";

            // Color-code % diff using configured validation thresholds
            const pctNum = percentDiff === "—" ? null : parseFloat(percentDiff);
            const pctClr = pctNum === null ? "" : pctNum <= VAL_PCT_PASS_MAX ? "color:#16a34a;font-weight:700;" : pctNum <= VAL_PCT_WARN_MAX ? "color:#d97706;font-weight:700;" : "color:#dc2626;font-weight:700;";

            // Color-code abs error using configured validation thresholds
            const errNum = absErr === "—" ? null : parseFloat(absErr);
            const errClr = errNum === null ? "" : errNum <= VAL_ABS_PASS_MAX ? "color:#16a34a;font-weight:700;" : errNum <= VAL_ABS_WARN_MAX ? "color:#d97706;font-weight:700;" : "color:#dc2626;font-weight:700;";

            // Fail-reason tooltip
            const failReasons = [];
            if (pctNum !== null && pctNum > VAL_PCT_PASS_MAX)   failReasons.push(`%diff ${percentDiff}% > ${VAL_PCT_PASS_MAX.toFixed(1)}%`);
            if (errNum !== null && errNum > VAL_ABS_PASS_MAX)  failReasons.push(`error ${absErr} mm > ${VAL_ABS_PASS_MAX.toFixed(3)} mm`);
            if (notes.includes("class_match=False")) failReasons.push("tread class mismatch");
            const failTip = failReasons.length ? ` title="FAIL: ${failReasons.join(" | ")}"` : "";

            const actionCell = recycleMode()
              ? `<td>
                  <button class="btn btn-outline btn-restore-val" data-id="${rowId}" style="padding:0.35rem 0.55rem;"><i class="fas fa-rotate-left"></i></button>
                  <button class="btn btn-subtle-danger btn-hard-del-val" data-id="${rowId}" style="padding:0.35rem 0.55rem;"><i class="fas fa-trash-can"></i></button>
                </td>`
              : `<td><button class="btn btn-subtle-danger btn-del-val" data-id="${rowId}" style="padding:0.35rem 0.55rem;"><i class="fas fa-trash"></i></button></td>`;

            tr.innerHTML = `
              <td><input type="checkbox" class="val-chk" data-id="${rowId}"></td>
              <td>${ts}</td>
              <td><strong>${tireId}</strong></td>
              <td>${manualDepth}</td>
              <td>${deviceDepth}</td>
              <td><span style="${pctClr}">${percentDiff === "—" ? "—" : percentDiff + "%"}</span></td>
              <td><span style="${errClr}">${absErr}</span></td>
              <td>${procTime}</td>
              <td>${treadClassHtml}</td>
              <td${failTip}><span class="verdict-badge ${verdictClass(verd)}">${verd}</span>${failReasons.length ? ' <span style="color:#dc2626;font-size:0.7rem;cursor:help;">⚠</span>' : ""}</td>
              <td style="font-size:0.75rem; color:#6c757d; max-width:180px; white-space:normal;">${notes}</td>
              ${actionCell}
            `;
            tbody.appendChild(tr);
          });

          tbody.querySelectorAll(".btn-del-val").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const id = e.currentTarget.getAttribute("data-id");
              await deleteSingleValidation(id);
            });
          });
          tbody.querySelectorAll(".btn-restore-val").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const id = e.currentTarget.getAttribute("data-id");
              await restoreSingleValidation(id);
            });
          });
          tbody.querySelectorAll(".btn-hard-del-val").forEach((btn) => {
            btn.addEventListener("click", async (e) => {
              const id = e.currentTarget.getAttribute("data-id");
              await hardDeleteSingleValidation(id);
            });
          });

          // Summary stats (numeric metrics only from calibrated/evaluable rows)
          const evalRows = items.filter(r => typeof r.percent_diff === "number" && typeof r.abs_error_mm === "number");
          const avgPct = evalRows.length
            ? (evalRows.reduce((s, r) => s + r.percent_diff, 0) / evalRows.length).toFixed(2)
            : "—";
          const maxErr = evalRows.length
            ? Math.max(...evalRows.map(r => r.abs_error_mm)).toFixed(3)
            : "—";
          const passCount  = evalRows.filter(r => (r.verdict || "").toUpperCase() === "PASS").length;
          const passRate   = evalRows.length ? Math.round((passCount / evalRows.length) * 100) : 0;
          const overall    = evalRows.length ? (passCount === evalRows.length ? "✅ ALL PASS" : `${passCount}/${evalRows.length} PASS`) : "PENDING CALIBRATION";

          $("valCount").textContent    = items.length;
          $("valAvgPct").textContent   = avgPct === "—" ? "—" : `${avgPct}%`;
          $("valMaxErr").textContent   = maxErr;
          $("valPassRate").textContent = `${passRate}%`;
          $("valOverall").textContent  = overall;

          // Class match count
          const classMatchCount = items.filter(r => (r.notes || "").includes("class_match=True")).length;
          $("valClassMatch").textContent = `${classMatchCount}/${items.length}`;
          $("valClassMatch").style.color = classMatchCount === items.length ? "#16a34a"
            : classMatchCount >= items.length * 0.75 ? "#d97706" : "#dc2626";

          // Avg abs error
          const avgErr = evalRows.length
            ? (evalRows.reduce((s, r) => s + r.abs_error_mm, 0) / evalRows.length).toFixed(3)
            : "—";
          $("valAvgErr").textContent = avgErr === "—" ? "—" : (avgErr + " mm");

          // Color avg % diff based on configured threshold
          const avgPctNum = parseFloat(avgPct);
          $("valAvgPct").style.color = Number.isFinite(avgPctNum)
            ? (avgPctNum <= VAL_PCT_PASS_MAX ? "#16a34a" : avgPctNum <= VAL_PCT_WARN_MAX ? "#d97706" : "#dc2626")
            : "#6c757d";

          // Thesis summary table for panel-ready reporting.
          const classPairs = items
            .map(r => parseValidationClasses(r.notes))
            .filter(Boolean);

          const replaceTP = classPairs.filter(p => p.manual === "REPLACE" && p.device === "REPLACE").length;
          const replaceFP = classPairs.filter(p => p.manual !== "REPLACE" && p.device === "REPLACE").length;
          const replaceFN = classPairs.filter(p => p.manual === "REPLACE" && p.device !== "REPLACE").length;
          const replacePrecision = (replaceTP + replaceFP) > 0 ? (replaceTP / (replaceTP + replaceFP)) : null;
          const replaceRecall = (replaceTP + replaceFN) > 0 ? (replaceTP / (replaceTP + replaceFN)) : null;

          const classMatchRate = items.length > 0 ? (classMatchCount / items.length) : 0;
          const passRate01 = evalRows.length > 0 ? (passCount / evalRows.length) : 0;

          const thesisRows = [
            thesisRow("Total Validation Trials", `${items.length}`, ">= 1", statusChip(items.length >= 1)),
            thesisRow("Evaluable Trials (with depth/error)", `${evalRows.length}`, ">= 1", statusChip(evalRows.length >= 1)),
            thesisRow("Average % Difference", avgPct === "—" ? "—" : `${avgPct}%`, `<= ${VAL_PCT_PASS_MAX.toFixed(1)}%`, statusChip(Number.isFinite(avgPctNum) ? (avgPctNum <= VAL_PCT_PASS_MAX) : false, !Number.isFinite(avgPctNum))),
            thesisRow("Maximum Absolute Error", maxErr === "—" ? "—" : `${maxErr} mm`, `<= ${VAL_ABS_PASS_MAX.toFixed(3)} mm`, statusChip(maxErr !== "—" ? (parseFloat(maxErr) <= VAL_ABS_PASS_MAX) : false, maxErr === "—")),
            thesisRow("Tread Class Match Rate", `${(classMatchRate * 100).toFixed(1)}% (${classMatchCount}/${items.length})`, "100% ideal", statusChip(classMatchRate >= 0.80)),
            thesisRow("Validation PASS Rate", `${(passRate01 * 100).toFixed(1)}% (${passCount}/${evalRows.length})`, "As high as possible", statusChip(passRate01 >= 0.80, evalRows.length === 0)),
            thesisRow("REPLACE Precision", replacePrecision === null ? "—" : `${(replacePrecision * 100).toFixed(1)}%`, "High precision", statusChip(replacePrecision !== null ? replacePrecision >= 0.80 : false, replacePrecision === null)),
            thesisRow("REPLACE Recall", replaceRecall === null ? "—" : `${(replaceRecall * 100).toFixed(1)}%`, "High recall", statusChip(replaceRecall !== null ? replaceRecall >= 0.80 : false, replaceRecall === null)),
          ];
          $("thesisSummaryBody").innerHTML = thesisRows.join("");

          $("valSummary").style.display = "block";
        }

        status(`Validation: ${items.length} trials`);
      } catch (error) {
        status(`Validation error: ${error.message}`);
        console.error("Validation load error:", error);
      }
    }

    async function deleteSingleValidation(id, askConfirm = true) {
      if (!id) return;
      if (askConfirm && !confirm(`Move validation row #${id} to recycle bin?`)) return;
      try {
        const res = await fetch(`/api/validation/${encodeURIComponent(id)}`, { method: "DELETE" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Delete failed (${res.status})`);
        await loadValidation();
        status(`Moved validation #${id} to recycle bin`);
      } catch (err) {
        status(`Validation delete error: ${err.message}`);
      }
    }

    async function restoreSingleValidation(id) {
      if (!id) return;
      try {
        const res = await fetch(`/api/validation/${encodeURIComponent(id)}/restore`, { method: "POST" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Restore failed (${res.status})`);
        await loadValidation();
        status(`Restored validation #${id}`);
      } catch (err) {
        status(`Validation restore error: ${err.message}`);
      }
    }

    async function hardDeleteSingleValidation(id) {
      if (!id) return;
      if (!confirm(`Permanently delete validation row #${id}? This cannot be undone.`)) return;
      try {
        const res = await fetch(`/api/validation/${encodeURIComponent(id)}/hard`, { method: "DELETE" });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(payload.detail || `Hard delete failed (${res.status})`);
        await loadValidation();
        status(`Permanently deleted validation #${id}`);
      } catch (err) {
        status(`Validation hard delete error: ${err.message}`);
      }
    }

    async function deleteSelectedValidationRows() {
      const ids = Array.from(document.querySelectorAll(".val-chk:checked")).map((x) => x.getAttribute("data-id"));
      if (!ids.length) {
        alert("No validation rows selected.");
        return;
      }
      const inRecycle = recycleMode();
      if (!confirm(inRecycle
        ? `Restore ${ids.length} selected validation row(s)?`
        : `Move ${ids.length} selected validation row(s) to recycle bin?`)) return;
      for (const id of ids) {
        // Best-effort sequential delete to keep API surface simple and robust.
        if (inRecycle) {
          await restoreSingleValidation(id);
        } else {
          await deleteSingleValidation(id, false);
        }
      }
      await loadValidation();
    }

    let pollTimer = null;
    function startPolling() {
      if (pollTimer) clearInterval(pollTimer);
      if ($("autoRefresh").checked) {
        pollTimer = setInterval(loadScans, 5000);
      }
    }

    $("btnRefresh").onclick = loadScans;
    $("btnExport").onclick = () => { window.location.href = "/api/export/csv"; };
    $("btnDeleteSelectedScans").onclick = deleteSelectedScans;
    $("btnPurgeAllData").onclick = purgeAllData;
    $("btnExportValidation").onclick = () => { window.location.href = "/api/export/validation"; };
    $("btnRefreshVal").onclick = loadValidation;
    $("btnDeleteSelectedVal").onclick = deleteSelectedValidationRows;
    $("showRecycleBin").onchange = () => {
      updateRecycleUi();
      loadScans();
      loadValidation();
    };
    $("btnViewActiveData").onclick = () => {
      $("showRecycleBin").checked = false;
      $("showRecycleBin").dispatchEvent(new Event("change"));
    };
    $("btnViewRecycleBin").onclick = () => {
      $("showRecycleBin").checked = true;
      $("showRecycleBin").dispatchEvent(new Event("change"));
    };
    $("chkAllScans").onchange = (e) => {
      document.querySelectorAll(".scan-chk").forEach((x) => { x.checked = !!e.target.checked; });
    };
    $("chkAllVal").onchange = (e) => {
      document.querySelectorAll(".val-chk").forEach((x) => { x.checked = !!e.target.checked; });
    };
    $("autoRefresh").onchange = startPolling;

    // ✅ Auto-fill scan timestamp when clicking a scan row
    const origLoadScans = loadScans;
    async function loadScansWithAutoFill() {
      await origLoadScans();
      // Attach click-to-fill on all scan rows
      document.querySelectorAll("#tbody tr").forEach(tr => {
        tr.addEventListener("click", () => {
          const tsCell = tr.querySelector("td:first-child");
          if (tsCell) {
            $("valScanTs").value = tsCell.textContent.trim();
            $("valScanTs").style.borderColor = "#1d4ed8";
            $("valForm").scrollIntoView({ behavior: "smooth", block: "nearest" });
          }
        });
      });
    }

    // Ensure every caller gets auto-fill behavior
    loadScans = loadScansWithAutoFill;

    // ✅ Submit validation form
    $("btnSubmitVal").onclick = async () => {
      const scanTs     = $("valScanTs").value.trim();
      const tireId     = $("valTireIdInput").value.trim();
      const manualMm   = parseFloat($("valManualDepth").value);
      const resultDiv  = $("valFormResult");

      if (!scanTs || !tireId || isNaN(manualMm) || manualMm <= 0) {
        resultDiv.style.display = "block";
        resultDiv.style.background = "#fff3cd";
        resultDiv.style.color = "#856404";
        resultDiv.textContent = "⚠️ Please fill in all fields: Scan Timestamp, Tire ID, and Manual Depth.";
        return;
      }

      $("btnSubmitVal").disabled = true;
      $("btnSubmitVal").textContent = "Submitting...";

      try {
        const form = new FormData();
        form.append("tire_id", tireId);
        form.append("manual_depth", manualMm);
        form.append("scan_ts", scanTs);

        const res = await fetch("/api/validation/submit", { method: "POST", body: form });
        
        // ✅ Fix: check content-type before parsing JSON
        const contentType = res.headers.get("content-type") || "";
        let data;
        if (contentType.includes("application/json")) {
          data = await res.json();
        } else {
          const text = await res.text();
          throw new Error(`Server error (${res.status}): ${text.substring(0, 200)}`);
        }

        if (!res.ok) throw new Error(data.detail || "Submission failed");

        const isPass = data.verdict === "PASS";
        const isPending = data.verdict === "PENDING";
        const depthTxt = (typeof data.device_depth === "number") ? `${data.device_depth.toFixed(2)} mm` : "—";
        const errTxt = (typeof data.abs_error_mm === "number") ? `${data.abs_error_mm.toFixed(3)} mm` : "—";
        const pctTxt = (typeof data.percent_diff === "number") ? `${data.percent_diff.toFixed(2)}%` : "—";
        resultDiv.style.display = "block";
        resultDiv.style.background = isPass ? "#d1fae5" : (isPending ? "#fff3cd" : "#fee2e2");
        resultDiv.style.color = isPass ? "#065f46" : (isPending ? "#856404" : "#991b1b");
        resultDiv.innerHTML = `
          ${isPass ? "✅" : (isPending ? "⏳" : "❌")} <strong>${data.verdict}</strong> &nbsp;|&nbsp;
          Tire: <strong>${data.tire_id}</strong> &nbsp;|&nbsp;
          Manual: <strong>${data.manual_depth.toFixed(2)} mm</strong> &nbsp;|&nbsp;
          Device: <strong>${depthTxt}</strong> &nbsp;|&nbsp;
          Error: <strong>${errTxt}</strong> &nbsp;|&nbsp;
          % Diff: <strong>${pctTxt}</strong>
        `;

        $("valTireIdInput").value = "";
        $("valManualDepth").value = "";
        await loadValidation();

      } catch (err) {
        resultDiv.style.display = "block";
        resultDiv.style.background = "#fee2e2";
        resultDiv.style.color = "#991b1b";
        resultDiv.textContent = `❌ Error: ${err.message}`;
      } finally {
        $("btnSubmitVal").disabled = false;
        $("btnSubmitVal").innerHTML = '<i class="fas fa-check-circle"></i> Submit Validation';
      }
    };

    // Also update valTbody to include notes column
    ["fVehicleType", "fVehicle", "fModel", "fTire", "fTireType", "fTreadDesign", "fVerdict", "limit"].forEach(id => {
      const el = $(id);
      if (el) el.addEventListener("change", loadScans);
      if (el) el.addEventListener("input", () => {
        clearTimeout(el._timeout);
        el._timeout = setTimeout(loadScans, 300);
      });
    });

    ["fValTire", "fValVerdict"].forEach(id => {
      const el = $(id);
      if (el) el.addEventListener("change", loadValidation);
      if (el) el.addEventListener("input", () => {
        clearTimeout(el._timeout);
        el._timeout = setTimeout(loadValidation, 300);
      });
    });

    $("imgModal").onclick = (e) => {
      if (e.target === $("imgModal")) {
        $("imgModal").classList.remove("active");
      }
    };
    $("modalClose").onclick = () => {
      $("imgModal").classList.remove("active");
    };

    document.addEventListener('DOMContentLoaded', () => {
      updateRecycleUi();
      loadScans();
      loadValidation();
      startPolling();
    });
  </script>
</body>
</html>"""
    html = (
        html
        .replace("__CAR_MIN_MM__", f"{car_min_mm:.1f}")
        .replace("__CAR_WARN_MM__", f"{car_warn_mm:.1f}")
        .replace("__CAR_WARN_UPPER_MM__", f"{(car_min_mm + car_warn_mm):.1f}")
        .replace("__MOTO_MIN_MM__", f"{moto_min_mm:.1f}")
        .replace("__MOTO_WARN_MM__", f"{moto_warn_mm:.1f}")
        .replace("__MOTO_WARN_UPPER_MM__", f"{(moto_min_mm + moto_warn_mm):.1f}")
        .replace("__VAL_PCT_MAX__", f"{val_pct_max:.1f}")
        .replace("__VAL_ABS_MAX__", f"{val_abs_max:.3f}")
        .replace("__VAL_PCT_WARN__", f"{val_pct_warn:.1f}")
        .replace("__VAL_ABS_WARN__", f"{val_abs_warn:.3f}")
    )
    return HTMLResponse(html)


def main():
    """Run the FastAPI server with uvicorn."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("tireguard.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()