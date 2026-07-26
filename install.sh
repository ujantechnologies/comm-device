#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
DIST_REPO="https://raw.githubusercontent.com/ujantechnologies/comm-device-dist/main"
MANIFEST_URL="${DIST_REPO}/releases/latest.json"
ARTIFACT_DIR="${REPO_ROOT}/artifacts"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl jq ffmpeg libatlas-base-dev

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
