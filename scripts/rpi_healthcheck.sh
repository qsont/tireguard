#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

HOST="${TIREGUARD_HOST:-0.0.0.0}"
PORT="${TIREGUARD_PORT:-8000}"
TARGET_USER="${SUDO_USER:-${USER:-}}"

PASS=0
WARN=0
FAIL=0

ok() {
  PASS=$((PASS + 1))
  echo "[OK]  $*"
}

warn() {
  WARN=$((WARN + 1))
  echo "[WARN] $*"
}

fail() {
  FAIL=$((FAIL + 1))
  echo "[FAIL] $*"
}

section() {
  echo ""
  echo "== $* =="
}

check_file_exec() {
  local p="$1"
  if [[ -f "$p" && -x "$p" ]]; then
    ok "Executable script present: ${p#"${ROOT_DIR}/"}"
  elif [[ -f "$p" ]]; then
    fail "Script exists but not executable: ${p#"${ROOT_DIR}/"}"
  else
    fail "Missing script: ${p#"${ROOT_DIR}/"}"
  fi
}

section "Scripts"
REQUIRED_SCRIPTS=(
  "${SCRIPTS_DIR}/rpi_setup.sh"
  "${SCRIPTS_DIR}/rpi_run_800x480.sh"
  "${SCRIPTS_DIR}/rpi_run.sh"
  "${SCRIPTS_DIR}/rpi_kiosk_autostart.sh"
  "${SCRIPTS_DIR}/rpi_desktop_launcher.sh"
  "${SCRIPTS_DIR}/rpi_reset_data.sh"
  "${SCRIPTS_DIR}/rpi_fix_db_path.sh"
)

for s in "${REQUIRED_SCRIPTS[@]}"; do
  check_file_exec "$s"
done

_syntax_ok=1
for _script in "${REQUIRED_SCRIPTS[@]}"; do
  if ! bash -n "$_script" 2>/dev/null; then
    fail "Bash syntax error in: ${_script#"${ROOT_DIR}/"}"
    _syntax_ok=0
  fi
done
(( _syntax_ok )) && ok "All required scripts pass bash syntax check"

section "App Layout"
if [[ -f "${ROOT_DIR}/app.py" ]]; then
  ok "app.py found"
else
  fail "app.py missing"
fi

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  ok "requirements.txt found"
else
  fail "requirements.txt missing"
fi

if [[ -d "${ROOT_DIR}/data" ]]; then
  ok "data directory exists"
else
  warn "data directory missing (it can be created by app/setup)"
fi

section "Environment"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  fail "TIREGUARD_PORT invalid: ${PORT}"
else
  ok "TIREGUARD_PORT valid: ${PORT}"
fi

if [[ -n "$HOST" ]]; then
  ok "TIREGUARD_HOST set: ${HOST}"
else
  fail "TIREGUARD_HOST is empty"
fi

if [[ -n "$TARGET_USER" ]]; then
  ok "Target user resolved: ${TARGET_USER}"
else
  fail "Could not resolve target user"
fi

TARGET_HOME=""
if [[ -n "$TARGET_USER" ]]; then
  TARGET_HOME="$(eval echo "~${TARGET_USER}")"
  if [[ -d "$TARGET_HOME" ]]; then
    ok "Target home exists: ${TARGET_HOME}"
  else
    fail "Target home missing: ${TARGET_HOME}"
  fi
fi

section "Python Runtime"
if [[ -x "$VENV_PYTHON" ]]; then
  ok "Virtualenv Python present: ${VENV_PYTHON}"

  if "$VENV_PYTHON" -c "import sys; print(sys.version.split()[0])" >/dev/null 2>&1; then
    ok "Virtualenv Python executes"
  else
    fail "Virtualenv Python failed to execute"
  fi

  if "$VENV_PYTHON" -c "import cv2" >/dev/null 2>&1; then
    ok "OpenCV import ok"
  else
    fail "OpenCV import failed in .venv"
  fi

  if "$VENV_PYTHON" -c "import PySide6" >/dev/null 2>&1; then
    ok "PySide6 import ok"
  else
    fail "PySide6 import failed in .venv"
  fi
else
  fail "Missing virtualenv Python: ${VENV_PYTHON}"
  warn "Run setup: ./scripts/rpi_setup.sh"
fi

section "Launcher Paths"
if [[ -n "$TARGET_HOME" && -d "$TARGET_HOME" ]]; then
  if [[ -d "$TARGET_HOME/.config" ]]; then
    ok "Desktop config directory exists"
  else
    warn "~/.config missing; autostart install will create it"
  fi

  if [[ -d "$TARGET_HOME/Desktop" ]]; then
    ok "Desktop directory exists"
  else
    warn "~/Desktop missing; desktop launcher install will create it"
  fi

  LOG_DIR="$TARGET_HOME/.local/state/tireguard"
  if mkdir -p "$LOG_DIR" >/dev/null 2>&1; then
    ok "Launcher log directory writable: ${LOG_DIR}"
  else
    fail "Cannot create launcher log directory: ${LOG_DIR}"
  fi
fi

section "Camera Hint"
if compgen -G "/dev/video*" > /dev/null; then
  ok "Camera device nodes detected (/dev/video*)"
else
  warn "No /dev/video* found (camera may be disconnected or permission-limited)"
fi

if [[ -n "$TARGET_USER" ]]; then
  if id -nG "$TARGET_USER" 2>/dev/null | grep -qw "video"; then
    ok "User '${TARGET_USER}' is in the 'video' group (camera access OK)"
  else
    warn "User '${TARGET_USER}' is NOT in 'video' group — camera may be denied. Run: sudo usermod -aG video ${TARGET_USER}"
  fi
fi

echo ""
echo "Summary: PASS=${PASS} WARN=${WARN} FAIL=${FAIL}"

if (( FAIL > 0 )); then
  exit 1
fi

exit 0
