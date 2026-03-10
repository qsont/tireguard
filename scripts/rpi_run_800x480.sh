#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

HOST="${TIREGUARD_HOST:-0.0.0.0}"
PORT="${TIREGUARD_PORT:-8000}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Error: virtual environment not found at ${VENV_DIR}."
  echo "Run setup first: ./scripts/rpi_setup.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${ROOT_DIR}"

# Dedicated 800x480 kiosk run mode: simple touchscreen UI + optional local web API.
exec python app.py --simple-ui --no-browser --host "${HOST}" --port "${PORT}" "$@"
