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

AUTOSTART_DIR="${TARGET_HOME}/.config/autostart"
DESKTOP_FILE="${AUTOSTART_DIR}/tireguard.desktop"
LOG_DIR="${TARGET_HOME}/.local/state/tireguard"
LOG_FILE="${LOG_DIR}/kiosk.log"

install_autostart() {
  mkdir -p "${AUTOSTART_DIR}"
  mkdir -p "${LOG_DIR}"

  cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=TireGuard
Comment=Start TireGuard in dedicated 800x480 kiosk mode
Exec=/bin/bash -lc 'cd "${ROOT_DIR}" && TIREGUARD_AUTO_SETUP=1 TIREGUARD_LOG_FILE="${LOG_FILE}" ./scripts/rpi_run_800x480.sh'
Path=${ROOT_DIR}
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
X-LXDE-Autostart=true
EOF

  # Ensure autostart file ownership matches desktop user.
  if command -v chown >/dev/null 2>&1; then
    chown "${TARGET_USER}:${TARGET_USER}" "${DESKTOP_FILE}" || true
    chown "${TARGET_USER}:${TARGET_USER}" "${LOG_DIR}" "${LOG_FILE}" 2>/dev/null || true
  fi

  echo "Installed kiosk autostart entry: ${DESKTOP_FILE}"
  echo "Kiosk logs: ${LOG_FILE}"
  echo ""
  echo "Next steps:"
  echo "1) Ensure Raspberry Pi boots to Desktop with auto-login for user '${TARGET_USER}'."
  echo "2) Reboot: sudo reboot"
}

remove_autostart() {
  if [[ -f "${DESKTOP_FILE}" ]]; then
    rm -f "${DESKTOP_FILE}"
    echo "Removed kiosk autostart entry: ${DESKTOP_FILE}"
  else
    echo "No kiosk autostart entry found at: ${DESKTOP_FILE}"
  fi
}

case "${MODE}" in
  install)
    install_autostart
    ;;
  remove|uninstall)
    remove_autostart
    ;;
  *)
    echo "Usage: $0 [install|remove]"
    exit 1
    ;;
esac
