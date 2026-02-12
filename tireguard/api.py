# -*- coding: utf-8 -*-
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

# Reuse your existing modules
try:
    from .config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

from .config import AppConfig
from .storage import list_results, get_result_by_ts, export_csv, find_processed_images


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
    if isinstance(p, Path):
        return str(p)
    return p


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(_safe_path(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    return _safe_path(obj)


app = FastAPI(title="TireGuard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "tireguard-api"}


@app.get("/api/scans")
def scans(limit: int = Query(50, ge=1, le=500)):
    cfg = _cfg()
    rows = list_results(cfg, limit=limit)
    return _jsonify({"items": rows, "limit": limit})


@app.get("/api/scans/{ts}")
def scan_detail(ts: str):
    cfg = _cfg()
    row = get_result_by_ts(cfg, ts)
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _jsonify(row)


@app.get("/api/scans/{ts}/images")
def scan_images(ts: str):
    cfg = _cfg()
    paths = find_processed_images(cfg, ts)
    if not paths:
        raise HTTPException(status_code=404, detail="No processed images found")
    return _jsonify({"ts": ts, "images": {k: str(v) for k, v in paths.items()}})


@app.get("/api/images/{ts}/{kind}")
def image(ts: str, kind: str):
    cfg = _cfg()
    paths = find_processed_images(cfg, ts)
    if not paths or kind not in paths:
        raise HTTPException(status_code=404, detail="Image kind not available")

    p = Path(paths[kind])
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image file missing")

    mime, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=mime or "image/png")


@app.get("/api/export/csv")
def export_csv_route():
    cfg = _cfg()
    p = export_csv(cfg)
    p = Path(p)
    if not p.exists():
        raise HTTPException(status_code=500, detail="CSV export failed")
    return FileResponse(str(p), media_type="text/csv", filename=p.name)


@app.get("/")
def index():
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
      --transition: all 0.2s ease;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    body {
      background-color: #f5f7fb;
      color: var(--gray-800);
      line-height: 1.6;
      min-height: 100vh;
    }

    /* Header Styles */
    header {
      background: linear-gradient(135deg, var(--primary), #0a58ca);
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
      font-size: 1.5rem;
      letter-spacing: -0.025em;
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
      box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }

    .btn-outline {
      background: transparent;
      border: 2px solid white;
      color: white;
    }

    .btn-outline:hover {
      background: white;
      color: var(--primary);
    }

    .toggle-switch {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      color: white;
    }

    .toggle-switch input {
      position: absolute;
      opacity: 0;
      width: 0;
      height: 0;
    }

    .toggle {
      position: relative;
      display: inline-block;
      width: 40px;
      height: 22px;
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

    /* Desktop App Button - top right */
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

    /* ✅ Fix: Add vertical scroll to Recent Scans card */
    .card-body > table {
      max-height: 400px;
      overflow-y: auto;
      display: block;
      border-bottom: 1px solid var(--gray-200);
    }

    /* For mobile: make table scroll horizontally */
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

    .verdict-pass .scan-verdict {
      background-color: rgba(25, 135, 84, 0.1);
      color: var(--success);
      border: 1px solid rgba(25, 135, 84, 0.2);
    }

    .verdict-warning .scan-verdict {
      background-color: rgba(255, 193, 7, 0.1);
      color: var(--warning);
      border: 1px solid rgba(255, 193, 7, 0.2);
    }

    .verdict-fail .scan-verdict {
      background-color: rgba(220, 53, 69, 0.1);
      color: var(--danger);
      border: 1px solid rgba(220, 53, 69, 0.2);
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
      overflow: hidden;
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
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      cursor: pointer;
      z-index: 10;
    }

    /* ✅ Mobile-specific styles - improved */
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

      /* ✅ Critical: Make table scrollable on mobile */
      .card-body > table {
        display: block;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
      }

      /* Add horizontal scrollbar for small screens */
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

      /* Hide desktop app button on mobile */
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
              <option>PASS</option>
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
      if (v.startsWith("PASS") || v === "OK") return "verdict-pass";
      if (v.includes("WARN")) return "verdict-warning";
      return "verdict-fail";
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
              <td colspan="10" style="text-align: center; padding: 2rem; color: #6c757d;">
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

            // ✅ Safe field extraction — prevents "undefined" → blank cells
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
            `;
            tbody.appendChild(tr);
          });
        }

        const scores = filtered.map(r => parseFloat(r.score) || 0);
        drawHistogram(scores);

        const passCount = filtered.filter(r => (r.verdict || "").toUpperCase().startsWith("PASS")).length;
        const rate = filtered.length ? Math.round((passCount / filtered.length) * 100) : 0;
        $("passRate").textContent = `${rate}%`;
        $("passCount").textContent = `${passCount}/${filtered.length}`;

        status(`Loaded ${filtered.length} scans`);
      } catch (error) {
        status(`Error: ${error.message}`);
        console.error("API Error:", error);
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
        $("selBrightness").textContent = (row.brightness || "").toFixed(2);
        $("selSharpness").textContent = (row.sharpness || "").toFixed(2);
        $("selEdgeDensity").textContent = (row.edge_density || "").toFixed(2);
        $("selContinuity").textContent = (row.continuity || "").toFixed(2);
        $("selNotes").textContent = row.notes || row["notes"] || "—";
        $("selMmPerPx").textContent = (row.mm_per_px || "").toFixed(6);

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

    let pollTimer = null;
    function startPolling() {
      if (pollTimer) clearInterval(pollTimer);
      if ($("autoRefresh").checked) {
        pollTimer = setInterval(loadScans, 5000);
      }
    }

    $("btnRefresh").onclick = loadScans;
    $("btnExport").onclick = () => { window.location.href = "/api/export/csv"; };
    $("autoRefresh").onchange = startPolling;

    ["fVehicle", "fTire", "fVerdict", "limit"].forEach(id => {
      $(id).addEventListener("input", loadScans);
    });

    $("imgModal").onclick = (e) => {
      if (e.target === $("imgModal")) {
        $("imgModal").classList.remove("active");
      }
    };

    $("modalClose").onclick = () => {
      $("imgModal").classList.remove("active");
    };

    // Initialize
    document.addEventListener('DOMContentLoaded', () => {
      loadScans();
      startPolling();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("tireguard.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()