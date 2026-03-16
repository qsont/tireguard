#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
TARGET_DB="${ROOT_DIR}/data/results.db"
MODE="${1:---apply}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: ${VENV_PYTHON} not found. Run ./scripts/rpi_setup.sh first."
  exit 1
fi

if [[ "${MODE}" != "--apply" && "${MODE}" != "--dry-run" ]]; then
  echo "Usage: $0 [--apply|--dry-run]"
  exit 1
fi

mkdir -p "${ROOT_DIR}/data"

"${VENV_PYTHON}" - <<'PY' "${ROOT_DIR}" "${TARGET_DB}" "${MODE}"
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
target = Path(sys.argv[2])
mode = sys.argv[3]
home = Path.home()


def get_stats(db_path: Path) -> tuple[int, int, str]:
    if not db_path.exists():
        return (0, 0, "")
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM results")
        results = int(cur.fetchone()[0])
        try:
            cur.execute("SELECT COUNT(*) FROM validation_results")
            vals = int(cur.fetchone()[0])
        except Exception:
            vals = 0
        try:
            cur.execute("SELECT COALESCE(MAX(ts), '') FROM results")
            max_ts = str(cur.fetchone()[0] or "")
        except Exception:
            max_ts = ""
        con.close()
        return (results, vals, max_ts)
    except Exception:
        return (0, 0, "")


# Find candidate DBs commonly created by cwd mistakes.
candidates: set[Path] = set()
for pattern in ("results.db", "tireguard.db"):
    for p in home.rglob(pattern):
        # Skip caches and virtual environments for speed/noise.
        s = str(p)
        if "/.cache/" in s or "/.venv/" in s or "/venv/" in s:
            continue
        candidates.add(p)

# Always include expected target.
candidates.add(target)

rows = []
for p in sorted(candidates):
    r, v, mx = get_stats(p)
    if r > 0 or v > 0 or p == target:
        rows.append((p, r, v, mx))

if not rows:
    print("No candidate database files found.")
    raise SystemExit(0)

rows.sort(key=lambda x: (x[1], x[2], x[3], str(x[0])))
best_path, best_results, best_vals, best_max_ts = rows[-1]

target_results, target_vals, target_max_ts = get_stats(target)

print("Detected database candidates:")
for p, r, v, mx in rows:
    marker = " <- BEST" if p == best_path else ""
    marker += " <- TARGET" if p == target else ""
    print(f"- {p} | results={r} validation={v} max_ts={mx or '-'}{marker}")

if best_path == target:
    print("Target DB is already the best candidate. No migration needed.")
    raise SystemExit(0)

if (best_results, best_vals, best_max_ts) <= (target_results, target_vals, target_max_ts):
    print("Target DB has equal or more data than alternatives. No migration performed.")
    raise SystemExit(0)

print(f"\nBest data source: {best_path}")
print(f"Target path:      {target}")

if mode == "--dry-run":
    print("Dry-run mode: no files were copied.")
    raise SystemExit(0)

# Backup target if present.
if target.exists():
    backup = target.with_name(f"results.db.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(target, backup)
    print(f"Backed up existing target DB to: {backup}")

shutil.copy2(best_path, target)
new_results, new_vals, new_max_ts = get_stats(target)
print(f"Migrated DB to target. New stats: results={new_results} validation={new_vals} max_ts={new_max_ts or '-'}")
PY

echo "Done. Restart app: pkill -f 'python app.py' || true && cd '${ROOT_DIR}' && python app.py"
