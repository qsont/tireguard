#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Error: virtual environment not found at ${VENV_DIR}."
  echo "Run setup first: ./scripts/rpi_setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${ROOT_DIR}"

echo "Resetting TireGuard dataset while keeping DB file/schema..."

python - <<'PY'
from pathlib import Path
import sqlite3

root = Path('.')
db_path = root / 'data' / 'results.db'

if not db_path.exists():
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.close()

con = sqlite3.connect(db_path)
cur = con.cursor()

for table in ('results', 'validation_results'):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if cur.fetchone():
        cur.execute(f'DELETE FROM {table}')

# Reset autoincrement counters when present
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
if cur.fetchone():
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('results', 'validation_results')")

con.commit()
con.close()

for pattern in [
    'data/captures/*',
    'data/processed/*',
    'data/results_export.csv',
    'data/test_results.csv',
]:
    for p in root.glob(pattern):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass

print('Done. DB file kept at data/results.db, records cleared.')
PY

echo "Fresh dataset ready."
