# -*- coding: utf-8 -*-
from __future__ import annotations

import mimetypes
import argparse
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# Reuse your existing modules
try:
    from .config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

from .config import AppConfig
from .storage import (
    init_db,  # added
    list_results, get_result_by_ts, export_csv, find_processed_images,
    list_validation_results, export_validation_summary
)


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


# ---------- Recent Scans ----------
@app.get("/api/scans")
def scans(limit: int = Query(50, ge=1, le=500)):
    """List recent scan results with filtering support."""
    cfg = _cfg()
    rows = list_results(cfg, limit=limit)
    return _jsonify({"items": rows, "limit": limit})


@app.get("/api/scans/{ts}")
def scan_detail(ts: str):
    """Get detailed information for a specific scan by timestamp."""
    cfg = _cfg()
    row = get_result_by_ts(cfg, ts)
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _jsonify(row)


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
    verdict: str | None = None
):
    """List device validation results (Table 3.2 methodology)."""
    cfg = _cfg()
    rows = list_validation_results(cfg, limit=limit, tire_id=tire_id, verdict=verdict)
    return _jsonify({"items": rows, "limit": limit})


@app.get("/api/export/validation")
def export_validation_route():
    """Export validation summary (Table 3.2) to CSV."""
    cfg = _cfg()
    p = export_validation_summary(cfg)
    p = Path(p)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Validation CSV not found")
    return FileResponse(str(p), media_type="text/csv", filename=p.name)


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

    # Compute device depth from linear model (slope=-6, intercept=6 default)
    device_depth = max(0.0, -6.0 * device_score + 6.0)

    abs_error = abs(device_depth - manual_depth)
    percent_diff = (abs_error / manual_depth * 100.0) if manual_depth > 0 else 0.0

    pass_pct = percent_diff <= 5.0
    pass_err = abs_error    <= 0.5
    verdict  = "PASS" if (pass_pct and pass_err) else "FAIL"

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
        "notes": f"Submitted via web dashboard | scan_ts={scan_ts}",
    })

    return JSONResponse(content=_jsonify({
        "ok": True,
        "tire_id": tire_id,
        "manual_depth": manual_depth,
        "device_depth": device_depth,
        "percent_diff": percent_diff,
        "abs_error_mm": abs_error,
        "verdict": verdict,
    }))


@app.get("/")
def index():
    """Serve the main TireGuard web dashboard."""
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

  <div class="container">
    <div class="card">
      <div class="card-header">
        <h2 class="card-title"><i class="fas fa-list"></i> Recent Scans</h2>
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
            <label for="fVehicle">Vehicle ID</label>
            <input type="text" id="fVehicle" placeholder="Enter vehicle ID" class="filter-control">
          </div>
          <div class="filter-group">
            <label for="fTire">Tire Position</label>
            <select id="fTire" class="filter-control">
              <option value="">All positions</option>
              <option>FL</option>
              <option>FR</option>
              <option>RL</option>
              <option>RR</option>
              <option>SPARE</option>
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
              <th>Timestamp</th>
              <th>Verdict</th>
              <th>Score</th>
              <th>Tire</th>
              <th>Vehicle</th>
              <th>Operator</th>
              <th>Brightness</th>
              <th>Sharpness</th>
              <th>Edge Density</th>
              <th>Continuity</th>
              <th>PSI Measured</th>
              <th>PSI Recommended</th>
              <th>PSI Status</th>
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
            <div class="info-label">Vehicle ID</div>
            <div class="info-value" id="selVehicle">—</div>
          </div>
          <div class="info-item">
            <div class="info-label">Tire Position</div>
            <div class="info-value" id="selTire">—</div>
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
          <button class="btn btn-outline" id="btnExportValidation">
            <i class="fas fa-file-csv"></i> Export Table 3.2
          </button>
          <button class="btn btn-outline" id="btnRefreshVal">
            <i class="fas fa-sync-alt"></i> Refresh
          </button>
        </div>
      </div>
      <div class="card-body">

        <!-- ✅ Submit Validation Form -->
        <div id="valForm" style="background:#f0f4ff; border:1px solid #c7d7ff; border-radius:12px; padding:1.25rem; margin-bottom:1.25rem;">
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
              <th>Timestamp</th>
              <th>Tire ID</th>
              <th>Manual (mm)</th>
              <th>Device (mm)</th>
              <th>% Diff</th>
              <th>Error (mm)</th>
              <th>Time (s)</th>
              <th>Verdict</th>
              <th>Notes</th>
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
    const status = (t) => {
      $("status").textContent = t;
      $("status").parentElement.classList.remove('fade-in');
      void $("status").offsetWidth; // Trigger reflow
      $("status").parentElement.classList.add('fade-in');
    };

    function verdictClass(v) {
      v = (v || "").toUpperCase();
      if (v === "GOOD" || v === "OK") return "verdict-good";
      if (v === "WARNING" || v.includes("WARN")) return "verdict-warning";
      if (v === "REPLACE") return "verdict-replace";
      if (v === "PASS") return "verdict-pass";
      if (v === "FAIL") return "verdict-fail";
      return "verdict-fail";
    }

    // Safe numeric formatter for nullable/string values
    function nfmt(v, digits = 2, fallback = "—") {
      const n = Number(v);
      return Number.isFinite(n) ? n.toFixed(digits) : fallback;
    }

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res;
    }

    function applyFilters(items) {
      const v = $("fVehicle").value.trim().toLowerCase();
      const t = $("fTire").value.trim().toUpperCase();
      const vd = $("fVerdict").value.trim().toUpperCase();
      return items.filter(r => {
        const rv = (r.vehicle_id ?? r["vehicle_id"] ?? "").toString().toLowerCase();
        const rt = (r.tire_position ?? r["tire_position"] ?? "").toString().toUpperCase();
        const rvd = (r.verdict ?? r["verdict"] ?? "").toString().toUpperCase();
        if (v && !rv.includes(v)) return false;
        if (t && rt !== t) return false;
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
        const res = await api(`/api/scans?limit=${limit}`);
        const data = await res.json();
        const items = Array.isArray(data.items) ? data.items : [];
        const filtered = applyFilters(items);
        const tbody = $("tbody");
        tbody.innerHTML = "";

        if (filtered.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="13" style="text-align: center; padding: 2rem; color: #6c757d;">
                <i class="fas fa-search" style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.3;"></i>
                <p>No scans match your criteria</p>
                <button id="resetFilters" class="btn btn-outline" style="margin-top: 1rem;">
                  <i class="fas fa-undo"></i> Reset Filters
                </button>
              </td>
            </tr>
          `;
          $("resetFilters").onclick = () => {
            $("fVehicle").value = "";
            $("fTire").value = "";
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
            const tire = get("tire_position", "-");
            const vehicle = get("vehicle_id", "-");
            const operator = get("operator", "-");
            const brightness = get("brightness", "-");
            const sharpness = get("sharpness", "-");
            const edgeDensity = get("edge_density", "-");
            const continuity = get("continuity", "-");
            const psiMeasured = row.psi_measured == null ? "—" : Number(row.psi_measured).toFixed(1);
            const psiRec = row.psi_recommended == null ? "—" : Number(row.psi_recommended).toFixed(1);
            const psiStatus = get("psi_status", "—");

            tr.innerHTML = `
              <td>${ts}</td>
              <td><span class="verdict-badge ${verdictClass(verdict)}">${verdict}</span></td>
              <td>${score}</td>
              <td>${tire}</td>
              <td>${vehicle}</td>
              <td>${operator}</td>
              <td>${brightness}</td>
              <td>${sharpness}</td>
              <td>${edgeDensity}</td>
              <td>${continuity}</td>
              <td>${psiMeasured}</td>
              <td>${psiRec}</td>
              <td><span class="verdict-badge ${verdictClass(psiStatus)}">${psiStatus}</span></td>
            `;
            tbody.appendChild(tr);
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

    async function selectScan(ts) {
      $("selTs").textContent = ts;
      status("Loading scan details...");
      
      try {
        const res = await api(`/api/scans/${encodeURIComponent(ts)}`);
        const row = await res.json();

        const verdict = row.verdict || "";
        $("selVerdict").className = `scan-verdict ${verdictClass(verdict)}`;
        $("selVerdict").textContent = verdict || "—";
        $("selVerdict").style.display = verdict ? 'inline-block' : 'none';

        $("selVehicle").textContent = row.vehicle_id || "—";
        $("selTire").textContent = row.tire_position || "—";
        $("selOperator").textContent = row.operator || "—";
        $("selTimestamp").textContent = row.ts || "—";
        $("selBrightness").textContent = nfmt(row.brightness, 2);
        $("selSharpness").textContent = nfmt(row.sharpness, 2);
        $("selEdgeDensity").textContent = nfmt(row.edge_density, 2);
        $("selContinuity").textContent = nfmt(row.continuity, 2);

        const treadVerdict = row.tread_verdict || "—";
        const psiMeasured = (typeof row.psi_measured === "number")
          ? row.psi_measured.toFixed(1)
          : (row.psi_measured ?? "—");
        const psiRecommended = (typeof row.psi_recommended === "number")
          ? row.psi_recommended.toFixed(1)
          : (row.psi_recommended ?? "—");
        const psiStatus = row.psi_status || "—";

        $("selTreadVerdict").innerHTML = `<span class="verdict-badge ${verdictClass(treadVerdict)}">${treadVerdict}</span>`;
        $("selPsiMeasured").textContent = psiMeasured;
        $("selPsiRecommended").textContent = psiRecommended;
        $("selPsiStatus").innerHTML = `<span class="verdict-badge ${verdictClass(psiStatus)}">${psiStatus}</span>`;

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
      
      try {
        const query = new URLSearchParams();
        query.append("limit", 100);
        if (tireId) query.append("tire_id", tireId);
        if (verdict) query.append("verdict", verdict);
        
        const res = await api(`/api/validation?${query}`);
        const data = await res.json();
        const items = Array.isArray(data.items) ? data.items : [];
        const tbody = $("valTbody");
        tbody.innerHTML = "";

        if (items.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="9" style="text-align: center; padding: 2rem; color: #6c757d;">
                No validation results found
              </td>
            </tr>
          `;
          $("valSummary").style.display = "none";
        } else {
          items.forEach((row) => {
            const tr = document.createElement("tr");
            const ts         = row.ts || row.created_at || "—";
            const tireId     = row.tire_id || "—";
            const manualDepth  = typeof row.manual_depth  === "number" ? row.manual_depth.toFixed(2)  : "—";
            const deviceDepth  = typeof row.device_depth  === "number" ? row.device_depth.toFixed(2)  : "—";
            const percentDiff  = typeof row.percent_diff  === "number" ? row.percent_diff.toFixed(2)  : "—";
            const absErr       = typeof row.abs_error_mm  === "number" ? row.abs_error_mm.toFixed(3)  : "—";
            const procTime     = typeof row.processing_time === "number" ? row.processing_time.toFixed(2) : "—";
            const verd         = row.verdict || "—";
            const notes        = row.notes || "—";

            tr.innerHTML = `
              <td>${ts}</td>
              <td><strong>${tireId}</strong></td>
              <td>${manualDepth}</td>
              <td>${deviceDepth}</td>
              <td>${percentDiff === "—" ? "—" : percentDiff + "%"}</td>
              <td>${absErr}</td>
              <td>${procTime}</td>
              <td><span class="verdict-badge ${verdictClass(verd)}">${verd}</span></td>
              <td style="font-size:0.75rem; color:#6c757d; max-width:180px; white-space:normal;">${notes}</td>
            `;
            tbody.appendChild(tr);
          });

          // Summary stats
          const avgPct     = (items.reduce((s, r) => s + (r.percent_diff || 0), 0) / items.length).toFixed(2);
          const maxErr     = Math.max(...items.map(r => r.abs_error_mm || 0)).toFixed(3);
          const passCount  = items.filter(r => (r.verdict || "").toUpperCase() === "PASS").length;
          const passRate   = Math.round((passCount / items.length) * 100);
          const overall    = passCount === items.length ? "✅ ALL PASS" : `${passCount}/${items.length} PASS`;

          $("valCount").textContent    = items.length;
          $("valAvgPct").textContent   = `${avgPct}%`;
          $("valMaxErr").textContent   = maxErr;
          $("valPassRate").textContent = `${passRate}%`;
          $("valOverall").textContent  = overall;
          $("valSummary").style.display = "block";
        }

        status(`Validation: ${items.length} trials`);
      } catch (error) {
        status(`Validation error: ${error.message}`);
        console.error("Validation load error:", error);
      }
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
    $("btnExportValidation").onclick = () => { window.location.href = "/api/export/validation"; };
    $("btnRefreshVal").onclick = loadValidation;
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
        resultDiv.style.display = "block";
        resultDiv.style.background = isPass ? "#d1fae5" : "#fee2e2";
        resultDiv.style.color = isPass ? "#065f46" : "#991b1b";
        resultDiv.innerHTML = `
          ${isPass ? "✅" : "❌"} <strong>${data.verdict}</strong> &nbsp;|&nbsp;
          Tire: <strong>${data.tire_id}</strong> &nbsp;|&nbsp;
          Manual: <strong>${data.manual_depth.toFixed(2)} mm</strong> &nbsp;|&nbsp;
          Device: <strong>${data.device_depth.toFixed(2)} mm</strong> &nbsp;|&nbsp;
          Error: <strong>${data.abs_error_mm.toFixed(3)} mm</strong> &nbsp;|&nbsp;
          % Diff: <strong>${data.percent_diff.toFixed(2)}%</strong>
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
    ["fVehicle", "fTire", "fVerdict", "limit"].forEach(id => {
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
      loadScans();
      loadValidation();
      startPolling();
    });
  </script>
</body>
</html>"""
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