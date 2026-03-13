#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} is not installed."
  exit 1
fi

APT_INSTALL_CMD=""
if command -v apt-get >/dev/null 2>&1; then
  if [[ "${EUID}" -eq 0 ]]; then
    APT_INSTALL_CMD="apt-get"
  elif command -v sudo >/dev/null 2>&1; then
    APT_INSTALL_CMD="sudo apt-get"
  else
    echo "Error: apt-get requires root or sudo privileges."
    exit 1
  fi
fi

if [[ -n "${APT_INSTALL_CMD}" ]]; then
  echo "[1/4] Installing Raspberry Pi system dependencies..."
  ${APT_INSTALL_CMD} update
  ${APT_INSTALL_CMD} install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0
else
  echo "[1/4] Skipping apt packages (apt-get not available on this system)."
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[2/4] Creating virtual environment in ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[2/4] Reusing existing virtual environment in ${VENV_DIR} ..."
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Error: Python executable missing in ${VENV_DIR}."
  exit 1
fi

echo "[3/4] Upgrading pip tooling..."
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "[4/4] Installing Python requirements..."
if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo "Error: requirements.txt not found at ${ROOT_DIR}/requirements.txt"
  exit 1
fi
"${VENV_PYTHON}" -m pip install -r "${ROOT_DIR}/requirements.txt"

echo ""
echo "Setup complete."
echo "Run the 800x480 app with: ./scripts/rpi_run_800x480.sh"
echo "Reset dataset (keep DB file): ./scripts/rpi_reset_data.sh"
