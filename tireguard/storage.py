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
    ]
    for col, typ in migrations:
        if not _col_exists(cur, "results", col):
            cur.execute(f"ALTER TABLE results ADD COLUMN {col} {typ}")

    con.commit()
    con.close()

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

def list_results(cfg, limit=30, vehicle_id=None, tire_position=None, verdict=None):
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

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    cur.execute(f"""
        SELECT ts, verdict, score, image_path, vehicle_id, tire_position
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
            "image_path": r[3],
            "vehicle_id": r[4],
            "tire_position": r[5],
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
               vehicle_id,tire_position,operator,session_notes,mm_per_px
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
            "vehicle_id","tire_position","operator","session_notes","mm_per_px"]
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
        "ts","image_path","vehicle_id","tire_position","operator","session_notes",
        "brightness","glare_ratio","sharpness",
        "edge_density","continuity","score","verdict",
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
