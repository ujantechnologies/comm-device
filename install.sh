#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
DIST_REPO="https://raw.githubusercontent.com/ujantechnologies/comm-device-dist/main"
MANIFEST_URL="${DIST_REPO}/releases/latest.json"
ARTIFACT_DIR="${REPO_ROOT}/artifacts"

sudo apt-get update

# Detect OS codename — package names differ between Bullseye and Bookworm
OS_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-bookworm}")"

# Core packages (available on all supported Pi OS releases)
sudo apt-get install -y \
    python3 python3-venv python3-pip \
    ffmpeg alsa-utils \
    curl

# jq is used to parse the dist manifest; non-fatal if unavailable
sudo apt-get install -y jq 2>/dev/null \
    || echo "Warning: jq not found — dist manifest step will be skipped"

# libatlas-base-dev was removed in Bookworm; numpy >= 1.26 bundles its own BLAS
if [ "${OS_CODENAME}" != "bookworm" ]; then
    sudo apt-get install -y libatlas-base-dev || true
fi

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"

mkdir -p "${ARTIFACT_DIR}"
if curl -fsSL "${MANIFEST_URL}" -o "${ARTIFACT_DIR}/latest.json"; then
  mapfile -t urls < <(jq -r '.assets[].url' "${ARTIFACT_DIR}/latest.json")
  for url in "${urls[@]}"; do
    file_name="$(basename "${url}")"
    curl -fsSL "${url}" -o "${ARTIFACT_DIR}/${file_name}"
  done
else
  echo "No distribution manifest found yet at ${MANIFEST_URL}; continuing scaffold install."
fi

echo "Install complete. Activate venv with: source ${VENV_DIR}/bin/activate"
echo "Run: python -m comm_device"
