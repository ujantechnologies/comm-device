# Comm Device — Raspberry Pi 5 Setup & User Instructions

## What This Device Does

This is an **AAC (Augmentative and Alternative Communication) device** that:
- Watches your face through a camera
- Detects **head nods** (YES) and **head shakes** (NO) via MediaPipe
- Generates a warm, empathetic spoken response using the **Gemma 3 1B** language model
- Speaks aloud via **Piper TTS** through Bluetooth or ALSA audio
- Shows live status on an **MHS-3.5" SPI display** (480×320 on `/dev/fb1`)
- Accepts **button feedback** (GPIO 17 = CORRECT, GPIO 27 = WRONG) to improve accuracy over time

---

## Hardware You Need

| Item | Notes |
|---|---|
| Raspberry Pi 5 | 4 GB or 8 GB RAM recommended |
| Pi Camera Module | Any CSI camera; optional IMX500 module for hardware face detection |
| MHS-3.5" SPI display | Wired to `/dev/fb1`. Skip if using HDMI monitor instead |
| 2× momentary push buttons | For GPIO feedback (see wiring below) |
| Bluetooth speaker **or** USB/3.5 mm audio output | For TTS playback |
| MicroSD card | 32 GB+ recommended (model files alone are ~1 GB) |

---

## Step 1 — Prepare the Raspberry Pi OS

1. Flash **Raspberry Pi OS Lite (64-bit)** or **Raspberry Pi OS Desktop (64-bit)** using the Raspberry Pi Imager.
2. Enable SSH, set your username/password, and configure Wi-Fi in the Imager's settings before flashing.
3. Boot, SSH in (or open a terminal), and update the system:

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

---

## Step 2 — Clone the Repository

```bash
cd ~
git clone <your-repo-url> comm-device
cd comm-device
```

> Replace `<your-repo-url>` with your actual Git remote URL.

---

## Step 3 — Run the Installer

The `install.sh` script handles everything: system packages, Python virtual environment, all Python dependencies, and model downloads (~1.2 GB total).

```bash
chmod +x install.sh
./install.sh
```

What gets downloaded automatically:
- `models/face_landmarker.task` — MediaPipe face landmarker (Google Storage)
- `models/en_US-lessac-medium.onnx` + `.json` — Piper TTS voice (HuggingFace)
- `models/gemma3-1b-Q4_K_M.gguf` — Gemma 3 1B quantized LLM (~806 MB, HuggingFace)

> The install can take **10–20 minutes** on a Pi 5 due to `llama-cpp-python` compiling from source with NEON + KleidiAI optimisations. This is normal.

When it finishes you will see:
```
Install complete.
Activate environment : source .venv/bin/activate
Run app              : python -m comm_device
```

---

## Step 3b — Hugging Face Model Downloads (Manual / Alternative)

`install.sh` downloads the HuggingFace models anonymously via `curl`. Both repositories are **public — no account or token is required**. If the automatic download fails (slow connection, rate-limit, or a partial download), follow these steps.

### Model sources

| File | HuggingFace Repository | Size |
|---|---|---|
| `gemma3-1b-Q4_K_M.gguf` | [`lmstudio-community/gemma-3-1b-it-GGUF`](https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF) | ~806 MB |
| `en_US-lessac-medium.onnx` | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) | ~63 MB |
| `en_US-lessac-medium.onnx.json` | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) | <1 MB |

### Option A — Re-run only the curl commands manually

```bash
mkdir -p models

# Gemma 3 1B GGUF
curl -L --progress-bar \
    "https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf" \
    -o models/gemma3-1b-Q4_K_M.gguf

# Piper voice model
curl -fsSL \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
    -o models/en_US-lessac-medium.onnx

curl -fsSL \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
    -o models/en_US-lessac-medium.onnx.json
```

### Option B — Use the Hugging Face CLI (more reliable on slow connections)

The HuggingFace CLI supports **resumable downloads** and is better for large files on an unreliable connection.

```bash
source .venv/bin/activate
pip install huggingface-hub

# Download Gemma 3 1B GGUF
huggingface-cli download \
    lmstudio-community/gemma-3-1b-it-GGUF \
    gemma-3-1b-it-Q4_K_M.gguf \
    --local-dir models \
    --local-dir-use-symlinks False

# Rename to match the expected filename
mv models/gemma-3-1b-it-Q4_K_M.gguf models/gemma3-1b-Q4_K_M.gguf

# Download Piper voice model
huggingface-cli download \
    rhasspy/piper-voices \
    en/en_US/lessac/medium/en_US-lessac-medium.onnx \
    en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
    --local-dir /tmp/piper-voices \
    --local-dir-use-symlinks False

mv /tmp/piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx models/
mv /tmp/piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json models/
```

### Option C — Download on another machine and copy to the Pi

If your Pi has no internet access or a very slow connection, download the files on a faster machine and copy them over:

```bash
# On your laptop / desktop (needs huggingface-hub installed)
pip install huggingface-hub
huggingface-cli download lmstudio-community/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf --local-dir .
huggingface-cli download rhasspy/piper-voices \
    en/en_US/lessac/medium/en_US-lessac-medium.onnx \
    en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
    --local-dir .

# Copy to the Pi over SSH (replace pi@raspberrypi.local with your Pi's address)
scp gemma-3-1b-it-Q4_K_M.gguf pi@raspberrypi.local:~/comm-device/models/gemma3-1b-Q4_K_M.gguf
scp en_US-lessac-medium.onnx pi@raspberrypi.local:~/comm-device/models/
scp en_US-lessac-medium.onnx.json pi@raspberrypi.local:~/comm-device/models/
```

### Verify all models are present

```bash
ls -lh models/
# Expected output:
#   classifier.pkl            (created after running train_classifier.py)
#   en_US-lessac-medium.onnx
#   en_US-lessac-medium.onnx.json
#   face_landmarker.task
#   gemma3-1b-Q4_K_M.gguf
```

---

## Step 4 — Wire the GPIO Feedback Buttons

Wire two momentary push buttons between the GPIO pins and GND. The firmware uses **pull-up** resistors internally, so pressing a button pulls the pin **low**.

```
Pi 5 GPIO Header
─────────────────────────
Pin 11  (GPIO 17 BCM)  ──┬── [CORRECT button] ──┐
                          └── 10 kΩ pull-up       │
                                                   │
Pin 13  (GPIO 27 BCM)  ──┬── [WRONG button]   ──┤
                          └── 10 kΩ pull-up       │
                                                   │
Pin 6   (GND)          ────────────────────────────┘
```

| Button | BCM Pin | Physical Pin | Action |
|---|---|---|---|
| CORRECT | GPIO 17 | Pin 11 | Confirms the prediction was right |
| WRONG | GPIO 27 | Pin 13 | Marks the prediction as wrong (flips to opposite label) |

> If you skip the buttons entirely, the device still works — every prediction is auto-accepted after the feedback window (default: 5 seconds).

---

## Step 5 — Connect the SPI Display (MHS-3.5")

If you are using an HDMI monitor instead, skip this section.

The display appears as `/dev/fb1` after the Pi's SPI overlay is enabled. To enable:

```bash
sudo raspi-config
# → Interface Options → SPI → Enable
# Reboot when prompted
```

Verify the display device exists after reboot:
```bash
ls /dev/fb*
# Should show /dev/fb0 and /dev/fb1
```

The app targets `/dev/fb1` automatically (set by `SDL_FBDEV` in `display.py`).

---

## Step 6 — Train the Expression Classifier

Before first use, train the sklearn classifier on a handful of sample frames:

```bash
source .venv/bin/activate
python scripts/train_classifier.py
```

This generates `models/classifier.pkl`. Re-run this script anytime after collecting enough corrected feedback data to retrain with better accuracy.

---

## Step 7 — Run the App

```bash
source .venv/bin/activate
python -m comm_device
```

The app starts all threads automatically:
- **Camera thread** — captures frames at 640×480
- **Expression thread** — runs MediaPipe head-gesture detection on every frame
- **Event/LLM thread** — fires when YES/NO is detected with ≥60% confidence (3-second cooldown between events)
- **Display thread** — renders live label + confidence bar on the SPI screen

To stop the app cleanly, press **Ctrl+C**. The app catches `SIGINT`/`SIGTERM` and shuts down all threads gracefully.

---

## Step 8 — Connect a Bluetooth Speaker (Optional)

```bash
bluetoothctl
> power on
> agent on
> scan on
# Wait until your speaker appears, note its MAC address
> pair XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> quit
```

The audio router detects the Bluetooth sink via `pactl` and routes TTS audio to it automatically. If no Bluetooth sink is found, it falls back to `aplay` on the default ALSA device.

---

## Configuration — Environment Variables

All settings have sensible defaults. Override any of them by exporting environment variables before running the app, or by adding them to a `.env` file sourced before launch.

| Variable | Default | Description |
|---|---|---|
| `COMM_DISPLAY_WIDTH` | `480` | Display resolution width |
| `COMM_DISPLAY_HEIGHT` | `320` | Display resolution height |
| `COMM_MODEL_PATH` | `models/gemma3-1b-Q4_K_M.gguf` | Path to Gemma GGUF model |
| `COMM_VOICE_MODEL_PATH` | `models/en_US-lessac-medium.onnx` | Path to Piper voice model |
| `COMM_CLASSIFIER_PATH` | `models/classifier.pkl` | Path to sklearn classifier |
| `COMM_FACE_LANDMARKER_PATH` | `models/face_landmarker.task` | Path to MediaPipe model |
| `COMM_IMX500_CONFIG_PATH` | *(empty)* | JSON config for IMX500 face-detection; leave empty to skip |
| `COMM_DB_PATH` | `artifacts/comm_device.sqlite3` | SQLite database path |
| `COMM_FEEDBACK_WINDOW_SECONDS` | `5` | Seconds to wait for button feedback after prediction |
| `COMM_EXPR_CONF_THRESHOLD` | `0.6` | Minimum confidence to trigger LLM+TTS pipeline |
| `COMM_LLM_COOLDOWN_SECONDS` | `3.0` | Minimum seconds between successive LLM calls |

Example — run with a longer feedback window and higher confidence threshold:

```bash
COMM_FEEDBACK_WINDOW_SECONDS=10 COMM_EXPR_CONF_THRESHOLD=0.75 python -m comm_device
```

---

## Retraining the Classifier from Collected Feedback

Every corrected prediction is stored in `artifacts/comm_device.sqlite3`. After using the device and pressing the buttons to correct mistakes, retrain:

```bash
source .venv/bin/activate
python scripts/train_classifier.py
```

To export raw training data as CSV:

```bash
python scripts/export_training_data.py
```

---

## Run at Boot (systemd)

To start the app automatically when the Pi powers on:

```bash
sudo nano /etc/systemd/system/comm-device.service
```

Paste the following (replace `pi` with your actual username):

```ini
[Unit]
Description=Comm Device AAC
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/comm-device
ExecStart=/home/pi/comm-device/.venv/bin/python -m comm_device
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable comm-device
sudo systemctl start comm-device

# Check status
sudo systemctl status comm-device

# View live logs
journalctl -u comm-device -f
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `picamera2 not available` in logs | picamera2 not installed or camera not connected | Confirm the camera ribbon is seated; `sudo apt install python3-picamera2` |
| `RPi.GPIO not available` in logs | Not running on real Pi hardware, or package missing | `sudo apt install python3-rpi.gpio` |
| No audio playback | `piper` binary not found or Bluetooth not connected | Re-run `install.sh`; check `bluetoothctl` connection |
| Display shows nothing on `/dev/fb1` | SPI not enabled | Run `sudo raspi-config` → Interface Options → SPI → Enable → reboot |
| `GGUF model not found` warning | Model download failed during install | Manually run the `curl` command from `install.sh` for the GGUF file |
| `Face landmarker model not found` | Model missing | Manually run the `curl` command from `install.sh` for `face_landmarker.task` |
| LLM responses are slow | Expected on Pi 5 without GPU | Normal; Q4_K_M quantisation keeps it to ~2–5 s per response |
| `classifier.pkl` missing | Never trained | Run `python scripts/train_classifier.py` |
| App crashes on startup with pygame error | No display attached | Attach HDMI monitor or SPI display; the app falls back to a regular window if `/dev/fb1` is unavailable |
