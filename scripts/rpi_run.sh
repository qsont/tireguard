#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Compatibility wrapper: keep old command working while standardizing on the
# dedicated 800x480 launcher.
exec ./scripts/rpi_run_800x480.sh "$@"
