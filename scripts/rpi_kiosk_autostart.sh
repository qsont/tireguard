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

install_autostart() {
  mkdir -p "${AUTOSTART_DIR}"

  cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=TireGuard
Comment=Start TireGuard in dedicated 800x480 kiosk mode
Exec=/bin/bash -lc 'cd "${ROOT_DIR}" && ./scripts/rpi_run_800x480.sh'
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

  # Ensure autostart file ownership matches desktop user.
  if command -v chown >/dev/null 2>&1; then
    chown "${TARGET_USER}:${TARGET_USER}" "${DESKTOP_FILE}" || true
  fi

  echo "Installed kiosk autostart entry: ${DESKTOP_FILE}"
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
