from pathlib import Path
from datetime import datetime
import json
import sqlite3
import cv2

def ensure_dirs(cfg):
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.captures_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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
    con.commit()
    con.close()

def save_capture(cfg, frame_bgr, meta: dict):
    ensure_dirs(cfg)
    ts = now_ts()
    img_path = cfg.captures_dir / f"tire_{ts}.jpg"
    meta_path = cfg.captures_dir / f"tire_{ts}.json"
    cv2.imwrite(str(img_path), frame_bgr)
    meta_path.write_text(json.dumps({"ts": ts, **meta}, indent=2))
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
    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()
    cur.execute("""
    INSERT INTO results (
        ts, image_path, roi_x, roi_y, roi_w, roi_h,
        brightness, glare_ratio, sharpness,
        edge_density, continuity, score, verdict, notes
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        row["ts"], row["image_path"],
        row["roi_x"], row["roi_y"], row["roi_w"], row["roi_h"],
        row["brightness"], row["glare_ratio"], row["sharpness"],
        row["edge_density"], row["continuity"], row["score"],
        row["verdict"], row.get("notes","")
    ))
    con.commit()
    con.close()

def list_results(cfg, limit=30):
    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT ts, verdict, score, image_path
        FROM results
        ORDER BY id DESC
        LIMIT ?
    """, (int(limit),))
    rows = cur.fetchall()
    con.close()
    return [{"ts": r[0], "verdict": r[1], "score": r[2], "image_path": r[3]} for r in rows]

def get_result_by_ts(cfg, ts: str):
    con = sqlite3.connect(cfg.db_path)
    cur = con.cursor()
    cur.execute("""
        SELECT ts,image_path,roi_x,roi_y,roi_w,roi_h,
               brightness,glare_ratio,sharpness,
               edge_density,continuity,score,verdict,notes
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
            "edge_density","continuity","score","verdict","notes"]
    return dict(zip(keys, row))

def find_processed_images(cfg, ts: str):
    """
    Returns paths if present: norm, edges, edges_closed, gray
    """
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
    cur = con.cursor()
    cur.execute("""
        SELECT ts,image_path,roi_x,roi_y,roi_w,roi_h,
               brightness,glare_ratio,sharpness,
               edge_density,continuity,score,verdict,notes
        FROM results
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    con.close()

    cfg.export_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.export_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts","image_path","roi_x","roi_y","roi_w","roi_h",
            "brightness","glare_ratio","sharpness",
            "edge_density","continuity","score","verdict","notes"
        ])
        w.writerows(rows)

    return cfg.export_csv_path
