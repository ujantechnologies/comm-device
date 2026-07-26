#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
MODEL_DIR="${REPO_ROOT}/models"
ARTIFACT_DIR="${REPO_ROOT}/artifacts"
DIST_MANIFEST="https://raw.githubusercontent.com/ujantechnologies/comm-device-dist/main/releases/latest.json"

mkdir -p "${MODEL_DIR}" "${ARTIFACT_DIR}"

# ── System packages ────────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-venv python3-pip \
    python3-picamera2 python3-rpi.gpio \
    libatlas-base-dev libopenblas-dev \
    ffmpeg portaudio19-dev \
    pulseaudio-utils curl jq

# ── Python virtual environment ─────────────────────────────────────────────────
python3 -m venv "${VENV_DIR}" --system-site-packages
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
pip install -r "${REPO_ROOT}/requirements.txt" --quiet

# ── Pi-specific Python packages ────────────────────────────────────────────────
# llama-cpp-python: build for ARM with NEON + KleidiAI
CMAKE_ARGS="-DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON" \
    pip install llama-cpp-python --no-binary llama-cpp-python --quiet
pip install piper-tts --quiet

# ── MediaPipe face landmarker model ───────────────────────────────────────────
FACE_MODEL="${MODEL_DIR}/face_landmarker.task"
if [ ! -f "${FACE_MODEL}" ]; then
    echo "Downloading MediaPipe face landmarker..."
    curl -fsSL \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
        -o "${FACE_MODEL}"
fi

# ── Piper voice model ─────────────────────────────────────────────────────────
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

# ── Gemma 3 1B GGUF (from HuggingFace, requires HF token for gated model) ─────
GGUF_MODEL="${MODEL_DIR}/gemma3-1b-Q4_K_M.gguf"
if [ ! -f "${GGUF_MODEL}" ]; then
    echo "Downloading Gemma 3 1B GGUF (Q4_K_M)..."
    echo "Note: you may be prompted to log in with your HuggingFace token."
    pip install huggingface_hub --quiet
    python3 - <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(
        repo_id="google/gemma-3-1b-it-GGUF",
        filename="gemma3-1b-it-Q4_K_M.gguf",
        local_dir=os.path.join(os.environ["REPO_ROOT"], "models"),
    )
    # Rename to match config default
    import shutil
    shutil.move(path, os.path.join(os.environ["REPO_ROOT"], "models", "gemma3-1b-Q4_K_M.gguf"))
except Exception as e:
    print(f"GGUF download failed: {e}")
    print("Place the GGUF file manually at models/gemma3-1b-Q4_K_M.gguf")
    sys.exit(0)
PY
fi

# ── Pre-built ARM64 binaries from comm-device-dist ────────────────────────────
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
