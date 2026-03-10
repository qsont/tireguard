#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-install}"

TARGET_USER="${SUDO_USER:-${USER:-}}"
if [[ -z "${TARGET_USER}" ]]; then
  echo "Error: Could not determine target user."
  exit 1
fi

TARGET_HOME="$(eval echo "~${TARGET_USER}")"
if [[ ! -d "${TARGET_HOME}" ]]; then
  echo "Error: Home directory not found for user ${TARGET_USER}."
  exit 1
fi

DESKTOP_DIR="${TARGET_HOME}/Desktop"
LAUNCHER_PATH="${DESKTOP_DIR}/TireGuard.desktop"

install_launcher() {
  mkdir -p "${DESKTOP_DIR}"

  cat > "${LAUNCHER_PATH}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=TireGuard
Comment=Launch TireGuard scanner app (800x480 mode)
Exec=/bin/bash -lc 'cd "${ROOT_DIR}" && ./scripts/rpi_run_800x480.sh'
Path=${ROOT_DIR}
Terminal=false
Categories=Utility;
EOF

  chmod +x "${LAUNCHER_PATH}"

  if command -v chown >/dev/null 2>&1; then
    chown "${TARGET_USER}:${TARGET_USER}" "${LAUNCHER_PATH}" || true
  fi

  echo "Desktop launcher created: ${LAUNCHER_PATH}"
  echo "If Raspberry Pi asks to trust/allow launching, right-click the icon and choose 'Allow Launching'."
}

remove_launcher() {
  if [[ -f "${LAUNCHER_PATH}" ]]; then
    rm -f "${LAUNCHER_PATH}"
    echo "Desktop launcher removed: ${LAUNCHER_PATH}"
  else
    echo "No launcher found at: ${LAUNCHER_PATH}"
  fi
}

case "${MODE}" in
  install)
    install_launcher
    ;;
  remove|uninstall)
    remove_launcher
    ;;
  *)
    echo "Usage: $0 [install|remove]"
    exit 1
    ;;
esac
