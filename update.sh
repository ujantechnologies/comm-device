#!/usr/bin/env bash
# update.sh — pull the latest code and sync Python packages.
# Does NOT re-download models or re-install system packages.
# Usage (from ~/comm-device):  ./update.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# -- Stop any running instance -------------------------------------------------
echo "==> Stopping any running comm_device instance..."
pkill -f comm_device 2>/dev/null && sleep 1 || true

# -- Pull latest code ----------------------------------------------------------
echo "==> Pulling latest code..."
git -C "${REPO_ROOT}" fetch --quiet origin
BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
git -C "${REPO_ROOT}" reset --hard "origin/${BRANCH}"

# -- Sync Python packages (skip mediapipe — handled separately) ----------------
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: virtual environment not found at ${VENV_DIR}"
    echo "Run ./install.sh first to create the environment."
    exit 1
fi

echo "==> Syncing Python packages..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
grep -v '^mediapipe' "${REPO_ROOT}/requirements.txt" | pip install -r /dev/stdin --quiet

# Ensure all scripts are executable after a reset
chmod +x "${REPO_ROOT}/run.sh" "${REPO_ROOT}/update.sh" "${REPO_ROOT}/install.sh"

# -- Regenerate the desktop launcher with correct paths and LF line endings ----
DESKTOP_FILE="${REPO_ROOT}/comm-device.desktop"
cat > "${DESKTOP_FILE}" << EOF
[Desktop Entry]
Type=Application
Name=Comm Device
Comment=AAC communication device
Exec=${REPO_ROOT}/run.sh
Path=${REPO_ROOT}
Icon=utilities-terminal
Terminal=false
Categories=Utility;
X-GNOME-Autostart-enabled=true
EOF
chmod +x "${DESKTOP_FILE}"
echo "Desktop launcher written to ${DESKTOP_FILE}"

DESKTOP_DIR="${HOME}/Desktop"
if [ -d "${DESKTOP_DIR}" ]; then
    cp "${DESKTOP_FILE}" "${DESKTOP_DIR}/comm-device.desktop"
    chmod +x "${DESKTOP_DIR}/comm-device.desktop"
    echo "Desktop launcher copied to ${DESKTOP_DIR}/comm-device.desktop"
else
    echo "Warning: ${DESKTOP_DIR} does not exist; skipping desktop copy"
fi

echo ""
echo "Update complete."
echo "Run app : ./run.sh"
