from pathlib import Path
from datetime import datetime
import json
import sqlite3
import cv2

def ensure_dirs(cfg):
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.captures_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
def _json_safe(o):
    """Recursively convert objects (Path, numpy types, dataclasses) to JSON-safe primitives."""
    from pathlib import Path
    try:
        import numpy as np
        numpy_types = (np.integer, np.floating, np.bool_)
    except Exception:
        numpy_types = ()

    if o is None:
        return None
    if isinstance(o, Path):
        return str(o)
    if numpy_types and isinstance(o, numpy_types):
        return o.item()
    if isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    # fallback: string representation (last resort)
    return str(o)

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())


def _capture_meta_path_for_ts(cfg, ts: str) -> Path:
    return cfg.captures_dir / f"tire_{ts}.json"


def _backfill_results_metadata(cfg):
    """Backfill new metadata columns from capture JSON for older rows.

    This is idempotent and only updates rows where one or more new fields are missing.
    """
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute(
        """
        SELECT id, ts, vehicle_type, tire_model_code, tire_type, tread_design
        FROM results
        WHERE vehicle_type IS NULL
           OR tire_model_code IS NULL
           OR tire_type IS NULL
           OR tread_design IS NULL
        """
    )
    rows = cur.fetchall()

    for row in rows:
        ts = row["ts"]
        if not ts:
            continue
        meta_path = _capture_meta_path_for_ts(cfg, ts)
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        session = payload.get("session") if isinstance(payload, dict) else None
        if not isinstance(session, dict):
            continue

        vehicle_type = session.get("vehicle_type")
        tire_model_code = session.get("tire_model_code")
        tire_type = session.get("tire_type")
        tread_design = session.get("tread_design")

        if all(v in (None, "", "-", "—") for v in [vehicle_type, tire_model_code, tire_type, tread_design]):
            continue

        cur.execute(
            """
            UPDATE results
            SET vehicle_type = COALESCE(vehicle_type, ?),
                tire_model_code = COALESCE(tire_model_code, ?),
                tire_type = COALESCE(tire_type, ?),
                tread_design = COALESCE(tread_design, ?)
            WHERE id = ?
            """,
            (
                vehicle_type if vehicle_type not in ("", "-", "—") else None,
                tire_model_code if tire_model_code not in ("", "-", "—") else None,
                tire_type if tire_type not in ("", "-", "—") else None,
                tread_design if tread_design not in ("", "-", "—") else None,
                row["id"],
            ),
        )

    con.commit()
    con.close()

def init_db(cfg):
    ensure_dirs(cfg)
    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        image_path TEXT,
        roi_x INTEGER,
        roi_y INTEGER,
        roi_w INTEGER,
        roi_h INTEGER,
        brightness REAL,
        glare_ratio REAL,
        sharpness REAL,
        edge_density REAL,
        continuity REAL,
        score REAL,
        verdict TEXT,
        notes TEXT
    )
    """)

    # --- lightweight migrations for new thesis workflow fields ---
    migrations = [
        ("vehicle_id", "TEXT"),
        ("tire_position", "TEXT"),
        ("operator", "TEXT"),
        ("session_notes", "TEXT"),
        ("mm_per_px", "REAL"),
        # PSI + combined logic fields
        ("tread_verdict", "TEXT"),
        ("psi_measured", "REAL"),
        ("psi_recommended", "REAL"),
        ("psi_status", "TEXT"),
        # Extended metadata fields
        ("vehicle_type", "TEXT"),
        ("tire_model_code", "TEXT"),
        ("tire_type", "TEXT"),
        ("tread_design", "TEXT"),
    ]
    for col, typ in migrations:
        if not _col_exists(cur, "results", col):
            cur.execute(f"ALTER TABLE results ADD COLUMN {col} {typ}")

    # NEW: validation table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS validation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        tire_id TEXT,
        manual_depth REAL NOT NULL,
        device_score REAL NOT NULL,
        device_depth REAL,
        percent_diff REAL,
        abs_error_mm REAL,
        processing_time REAL,
        verdict TEXT,
        notes TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)

    con.commit()
    con.close()

    # Backfill new metadata fields for existing records where possible.
    _backfill_results_metadata(cfg)

def save_capture(cfg, frame_bgr, meta: dict):
    ensure_dirs(cfg)
    ts = now_ts()
    img_path = cfg.captures_dir / f"tire_{ts}.jpg"
    meta_path = cfg.captures_dir / f"tire_{ts}.json"
    cv2.imwrite(str(img_path), frame_bgr)
    meta_path.write_text(json.dumps({"ts": ts, **meta}, indent=2, default=str), encoding='utf-8')
    return ts, img_path, meta_path

def save_processed(cfg, ts: str, processed: dict):
    ensure_dirs(cfg)
    out_paths = {}
    for k, img in processed.items():
        p = cfg.processed_dir / f"{ts}_{k}.png"
        cv2.imwrite(str(p), img)
        out_paths[k] = str(p)
    return out_paths

def insert_result(cfg, row: dict):
    """
    Insert a row into results table.
    Robust: builds column list + placeholders from the provided dict keys.
    This prevents 'N values for M columns' mismatches when schema evolves.
    """
    import sqlite3
    from pathlib import Path

    db_path = getattr(cfg, "db_path", None) or getattr(cfg, "data_dir", None)
    if db_path is None:
        # fallback: look for cfg.db_path attribute that your init_db() uses
        raise RuntimeError("Config has no db_path/data_dir; cannot locate database.")
    if not str(db_path).endswith(".db"):
        # if cfg.data_dir is provided, assume db file is inside it as tireguard.db
        db_path = Path(db_path) / "tireguard.db"
    else:
        db_path = Path(db_path)

    # Keep only known columns that exist in the table (protect against stray keys)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("PRAGMA table_info(results)")
    cols_in_db = {r["name"] for r in cur.fetchall()}

    clean = {k: row.get(k) for k in row.keys() if k in cols_in_db}
    if "ts" not in clean:
        clean["ts"] = row.get("ts")

    cols = list(clean.keys())
    vals = [clean[c] for c in cols]
    placeholders = ", ".join(["?"] * len(cols))

    sql = f"INSERT INTO results ({', '.join(cols)}) VALUES ({placeholders})"
    cur.execute(sql, vals)
    con.commit()
    con.close()

def insert_validation_result(cfg, row: dict):
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("PRAGMA table_info(validation_results)")
    cols_in_db = {r["name"] for r in cur.fetchall()}

    clean = {k: row.get(k) for k in row.keys() if k in cols_in_db}
    if "ts" not in clean:
        clean["ts"] = now_ts()

    cols = list(clean.keys())
    vals = [clean[c] for c in cols]
    placeholders = ", ".join(["?"] * len(cols))

    sql = f"INSERT INTO validation_results ({', '.join(cols)}) VALUES ({placeholders})"
    cur.execute(sql, vals)
    con.commit()
    con.close()


def list_results(
    cfg,
    limit=30,
    vehicle_id=None,
    tire_position=None,
    verdict=None,
    vehicle_type=None,
    tire_type=None,
    tread_design=None,
    tire_model_code=None,
):
    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()

    where = []
    params = []

    if vehicle_id:
        where.append("vehicle_id = ?")
        params.append(vehicle_id)
    if tire_position:
        where.append("tire_position = ?")
        params.append(tire_position)
    if verdict:
        where.append("verdict = ?")
        params.append(verdict)
    if vehicle_type:
        where.append("vehicle_type = ?")
        params.append(vehicle_type)
    if tire_type:
        where.append("tire_type = ?")
        params.append(tire_type)
    if tread_design:
        where.append("tread_design = ?")
        params.append(tread_design)
    if tire_model_code:
        where.append("tire_model_code = ?")
        params.append(tire_model_code)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    cur.execute(f"""
        SELECT 
            ts, verdict, score,
            vehicle_type, vehicle_id, tire_model_code,
            tire_position, tire_type, tread_design,
            operator,
            brightness, sharpness, edge_density, continuity,
            psi_measured, psi_recommended, psi_status, tread_verdict
        FROM results
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """, (*params, int(limit)))
    rows = cur.fetchall()
    con.close()
    return [
        {
            "ts": r[0],
            "verdict": r[1],
            "score": r[2],
            "vehicle_type": r[3],
            "vehicle_id": r[4],
            "tire_model_code": r[5],
            "tire_position": r[6],
            "tire_type": r[7],
            "tread_design": r[8],
            "operator": r[9],
            "brightness": r[10],
            "sharpness": r[11],
            "edge_density": r[12],
            "continuity": r[13],
            "psi_measured": r[14],
            "psi_recommended": r[15],
            "psi_status": r[16],
            "tread_verdict": r[17],
        }
        for r in rows
    ]

def get_result_by_ts(cfg, ts: str):
    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT ts,image_path,roi_x,roi_y,roi_w,roi_h,
               brightness,glare_ratio,sharpness,
               edge_density,continuity,score,verdict,notes,
               vehicle_type,vehicle_id,tire_model_code,tire_position,tire_type,tread_design,
               operator,session_notes,mm_per_px,
               tread_verdict,psi_measured,psi_recommended,psi_status
        FROM results
        WHERE ts=?
        LIMIT 1
    """, (ts,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    keys = ["ts","image_path","roi_x","roi_y","roi_w","roi_h",
            "brightness","glare_ratio","sharpness",
            "edge_density","continuity","score","verdict","notes",
            "vehicle_type","vehicle_id","tire_model_code","tire_position","tire_type","tread_design",
            "operator","session_notes","mm_per_px",
            "tread_verdict","psi_measured","psi_recommended","psi_status"]
    return dict(zip(keys, row))

def find_processed_images(cfg, ts: str):
    candidates = ["norm", "edges", "edges_closed", "gray"]
    out = {}
    for k in candidates:
        p = cfg.processed_dir / f"{ts}_{k}.png"
        if p.exists():
            out[k] = str(p)
    return out

def export_csv(cfg):
    import csv
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    COLUMNS = [
        "ts","image_path","vehicle_type","vehicle_id","tire_model_code",
        "tire_position","tire_type","tread_design",
        "operator","session_notes",
        "brightness","glare_ratio","sharpness",
        "edge_density","continuity","score","verdict","tread_verdict",
        "psi_measured","psi_recommended","psi_status",
        "mm_per_px","notes"
    ]

    cur.execute(f"""
        SELECT {", ".join(COLUMNS)}
        FROM results
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    con.close()

    cfg.export_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.export_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            rd = dict(r)  # sqlite3.Row -> dict
            w.writerow([rd.get(c, "") if rd.get(c, None) is not None else "" for c in COLUMNS])

    return cfg.export_csv_path

def list_validation_results(cfg, limit=50, tire_id=None, verdict=None):
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    where = []
    params = []

    if tire_id:
        where.append("tire_id = ?")
        params.append(tire_id)
    if verdict:
        where.append("verdict = ?")
        params.append(verdict)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    cur.execute(f"""
        SELECT
            id, ts, tire_id, manual_depth, device_score, device_depth,
            percent_diff, abs_error_mm, processing_time, verdict, notes, created_at
        FROM validation_results
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """, (*params, int(limit)))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def export_validation_summary(cfg):
    import csv

    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT
            tire_id, created_at, device_depth, manual_depth,
            percent_diff, processing_time, verdict
        FROM validation_results
        ORDER BY id ASC
    """)
    rows = cur.fetchall()
    con.close()

    out_path = cfg.data_dir / "validation_summary_table3_2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Trial Number", "Date and Time",
            "Device Prototype Reading (mm)", "Traditional Tread Depth Gauge Reading (mm)",
            "Percentage Difference (%)", "Processing Time (s)", "Verdict"
        ])
        for tire_id, created_at, device_depth, manual_depth, percent_diff, processing_time, v in rows:
            w.writerow([
                tire_id or "",
                created_at or "",
                round(float(device_depth), 2) if device_depth is not None else "",
                round(float(manual_depth), 2) if manual_depth is not None else "",
                round(float(percent_diff), 2) if percent_diff is not None else "",
                round(float(processing_time), 2) if processing_time is not None else "",
                v or "",
            ])

    return out_path
