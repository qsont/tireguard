#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

HOST="${TIREGUARD_HOST:-0.0.0.0}"
PORT="${TIREGUARD_PORT:-8000}"
AUTO_SETUP="${TIREGUARD_AUTO_SETUP:-0}"
LOG_FILE="${TIREGUARD_LOG_FILE:-}"

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  # Route launcher output to a persistent file for kiosk troubleshooting.
  exec >> "${LOG_FILE}" 2>&1
  echo "[$(date +"%F %T")] TireGuard launcher starting"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  if [[ "${AUTO_SETUP}" == "1" ]]; then
    echo "Virtual environment missing. Running setup..."
    "${ROOT_DIR}/scripts/rpi_setup.sh"
  else
    echo "Error: virtual environment not found at ${VENV_DIR}."
    echo "Run setup first: ./scripts/rpi_setup.sh"
    echo "Or set TIREGUARD_AUTO_SETUP=1 to bootstrap automatically."
    exit 1
  fi
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: Python not found in virtual environment: ${VENV_PYTHON}"
  echo "Re-run setup: ./scripts/rpi_setup.sh"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/app.py" ]]; then
  echo "Error: app.py not found in ${ROOT_DIR}."
  exit 1
fi

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Error: TIREGUARD_PORT must be a valid port (1-65535). Got: ${PORT}"
  exit 1
fi

cd "${ROOT_DIR}"

# Dedicated 800x480 kiosk run mode: simple touchscreen UI + optional local web API.
exec "${VENV_PYTHON}" app.py --simple-ui --no-browser --host "${HOST}" --port "${PORT}" "$@"
