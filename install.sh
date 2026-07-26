#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
MODEL_DIR="${REPO_ROOT}/models"
ARTIFACT_DIR="${REPO_ROOT}/artifacts"
DIST_MANIFEST="https://raw.githubusercontent.com/ujantechnologies/comm-device-dist/main/releases/latest.json"

mkdir -p "${MODEL_DIR}" "${ARTIFACT_DIR}"

# -- System packages ------------------------------------------------------------
# Detect OS codename — package names differ between Bullseye and Bookworm
OS_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-bookworm}")"

sudo apt-get update -qq

# Core packages available on all supported Pi OS releases
sudo apt-get install -y \
    python3 python3-venv python3-pip \
    python3-picamera2 \
    ffmpeg alsa-utils \
    curl

# python3-rpi.gpio was renamed to python3-rpi-lgpio in Bookworm
if apt-cache show python3-rpi-lgpio &>/dev/null; then
    sudo apt-get install -y python3-rpi-lgpio
else
    sudo apt-get install -y python3-rpi.gpio
fi

# jq: used to parse the dist manifest; non-fatal if unavailable
sudo apt-get install -y jq 2>/dev/null \
    || echo "Warning: jq not found — dist manifest step will be skipped"

# Audio: Bookworm uses PipeWire (provides pactl/paplay); older releases use pulseaudio-utils
if [ "${OS_CODENAME}" = "bookworm" ]; then
    sudo apt-get install -y pipewire-pulse
else
    sudo apt-get install -y pulseaudio-utils
fi

# libatlas-base-dev and portaudio19-dev were removed in Bookworm;
# numpy >= 1.26 bundles its own BLAS so they are not required
if [ "${OS_CODENAME}" != "bookworm" ]; then
    sudo apt-get install -y libatlas-base-dev libopenblas-dev portaudio19-dev || true
fi

# -- Python virtual environment -------------------------------------------------
python3 -m venv "${VENV_DIR}" --system-site-packages
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
pip install -r "${REPO_ROOT}/requirements.txt" --quiet

# -- Pi-specific Python packages ------------------------------------------------
# llama-cpp-python: build for ARM with NEON + KleidiAI
CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON" \
    pip install llama-cpp-python --no-binary llama-cpp-python --quiet
pip install piper-tts --quiet

# -- MediaPipe face landmarker model -------------------------------------------
FACE_MODEL="${MODEL_DIR}/face_landmarker.task"
if [ ! -f "${FACE_MODEL}" ]; then
    echo "Downloading MediaPipe face landmarker..."
    curl -fsSL \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
        -o "${FACE_MODEL}"
fi

# -- Piper voice model ---------------------------------------------------------
VOICE_MODEL="${MODEL_DIR}/en_US-lessac-medium.onnx"
VOICE_JSON="${MODEL_DIR}/en_US-lessac-medium.onnx.json"
if [ ! -f "${VOICE_MODEL}" ]; then
    echo "Downloading Piper voice model..."
    curl -fsSL \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        -o "${VOICE_MODEL}"
    curl -fsSL \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        -o "${VOICE_JSON}"
fi

# -- Gemma 3 1B GGUF (lmstudio-community — public, no login required) ----------
# Source: https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF
GGUF_MODEL="${MODEL_DIR}/gemma3-1b-Q4_K_M.gguf"
if [ ! -f "${GGUF_MODEL}" ]; then
    echo "Downloading Gemma 3 1B Q4_K_M GGUF (~806 MB)..."
    curl -L --progress-bar \
        "https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf" \
        -o "${GGUF_MODEL}"
fi

# -- Pre-built ARM64 binaries from comm-device-dist ----------------------------
if curl -fsSL "${DIST_MANIFEST}" -o "${ARTIFACT_DIR}/latest.json" 2>/dev/null; then
    VERSION=$(jq -r '.version' "${ARTIFACT_DIR}/latest.json")
    echo "Dist manifest: ${VERSION}"
    mapfile -t URLS < <(jq -r '.assets[].url' "${ARTIFACT_DIR}/latest.json")
    for URL in "${URLS[@]}"; do
        FNAME=$(basename "${URL}")
        if [ ! -f "${ARTIFACT_DIR}/${FNAME}" ]; then
            echo "Downloading ${FNAME}..."
            curl -fsSL "${URL}" -o "${ARTIFACT_DIR}/${FNAME}"
            case "${FNAME}" in
                *.tar.gz) tar -xzf "${ARTIFACT_DIR}/${FNAME}" -C "${ARTIFACT_DIR}/" ;;
            esac
        fi
    done
else
    echo "No dist manifest available yet; skipping binary download."
fi

echo ""
echo "Install complete."
echo "Activate environment : source ${VENV_DIR}/bin/activate"
echo "Run app              : python -m comm_device"
echo "Retrain classifier   : python scripts/train_classifier.py"
echo "Export training data : python scripts/export_training_data.py"
