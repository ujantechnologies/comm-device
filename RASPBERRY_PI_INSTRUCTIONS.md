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

The `install.sh` script handles everything: system packages, Python virtual environment, and all Python dependencies.

```bash
chmod +x install.sh
./install.sh
```

The script auto-detects whether you are on **Bullseye** or **Bookworm** and picks the correct package names. When it finishes you will see:

```
Install complete. Activate venv with: source .venv/bin/activate
Run: python -m comm_device
```

> **Updating after a code change?** Use `update.sh` instead — it runs `git pull` and re-syncs Python packages without re-downloading the ~870 MB of model files:
> ```bash
> chmod +x update.sh
> ./update.sh
> ```

---

## Step 3b — Hugging Face Model Downloads

The app requires three model files that must be downloaded separately. Both HuggingFace repositories are **public — no account or token is required**.

### Model sources

| File | Repository | Size |
|---|---|---|
| `models/gemma3-1b-Q4_K_M.gguf` | [`lmstudio-community/gemma-3-1b-it-GGUF`](https://huggingface.co/lmstudio-community/gemma-3-1b-it-GGUF) | ~806 MB |
| `models/en_US-lessac-medium.onnx` | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) | ~63 MB |
| `models/en_US-lessac-medium.onnx.json` | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) | <1 MB |
| `models/face_landmarker.task` | Google Storage (MediaPipe) | ~2 MB |

### Option A — Download directly on the Pi with curl

```bash
mkdir -p models

# Gemma 3 1B GGUF (~806 MB)
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

# MediaPipe face landmarker
curl -fsSL \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
    -o models/face_landmarker.task
```

### Option B — Use the Hugging Face CLI (resumable, better for slow connections)

```bash
source .venv/bin/activate
pip install huggingface-hub

# Gemma 3 1B GGUF
huggingface-cli download \
    lmstudio-community/gemma-3-1b-it-GGUF \
    gemma-3-1b-it-Q4_K_M.gguf \
    --local-dir models \
    --local-dir-use-symlinks False

# Rename to match the expected filename in config.py
mv models/gemma-3-1b-it-Q4_K_M.gguf models/gemma3-1b-Q4_K_M.gguf

# Piper voice model
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

```bash
# On your laptop / desktop
pip install huggingface-hub
huggingface-cli download lmstudio-community/gemma-3-1b-it-GGUF \
    gemma-3-1b-it-Q4_K_M.gguf --local-dir .
huggingface-cli download rhasspy/piper-voices \
    en/en_US/lessac/medium/en_US-lessac-medium.onnx \
    en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
    --local-dir .

# Copy to the Pi over SSH (replace pi@raspberrypi.local with your Pi's address)
scp gemma-3-1b-it-Q4_K_M.gguf \
    pi@raspberrypi.local:~/comm-device/models/gemma3-1b-Q4_K_M.gguf
scp en_US-lessac-medium.onnx \
    pi@raspberrypi.local:~/comm-device/models/
scp en_US-lessac-medium.onnx.json \
    pi@raspberrypi.local:~/comm-device/models/
```

### Verify all models are present

```bash
ls -lh models/
# Expected:
#   classifier.pkl              (after running train_classifier.py)
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

Training mode controls on the touchscreen:
- **MODE** — switch between COMM and TRAIN
- **INT** — cycle the current target intent label
- **INT (double-tap)** — add a new intent label by voice (speak a short label)
- **ASK** — record a caregiver-spoken training question for the next sample
- **REV** — browse saved training videos and play the selected one
- **DEL** — delete the currently selected training video
- **RST** — clear all saved training videos and reset training samples/model
- **REC** — in COMM mode, listen for a caregiver question; in TRAIN mode, start recording a training sample
- **FIT** — train the personalized ASL intent model from collected samples

To train the gesture that starts the training-question flow, use the `speak_question` intent in TRAIN mode:
- Tap **INT** until `speak_question` is selected.
- Perform that same gesture repeatedly while collecting samples with **REC**.
- Tap **FIT** after collecting enough examples.
- After the model is trained, that gesture can trigger the spoken training question without tapping REC.

By default, training capture stores 5-second response clips because `COMM_TRAINING_CLIP_SECONDS` defaults to `5`.
Caregiver microphone questions now stop early when speech ends, up to the max time set by `COMM_MIC_QUESTION_SECONDS`.

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

All settings have sensible defaults. Override any of them by exporting environment variables before running the app.

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
| `COMM_TRAINING_INTENTS_FILE` | `artifacts/training_intents.txt` | Persisted caregiver-added intent labels |
| `COMM_TRAINING_VIDEO_DIR` | `artifacts/training_videos` | Directory for saved training video review artifacts |
| `COMM_TRAINING_CLIP_SECONDS` | `5` | Duration of each saved training response clip |
| `COMM_TRAINING_STRICT_LOCAL_LLM` | `true` | Require local GGUF model for training questions |
| `COMM_TRAINING_QUESTION_TEMPERATURE` | `0.65` | Sampling temperature for generated training questions |
| `COMM_TRAINING_QUESTION_TRIGGER_ENABLED` | `true` | Enable gesture-triggered question playback in TRAIN mode |
| `COMM_TRAINING_QUESTION_TRIGGER_INTENT` | `speak_question` | Intent label that triggers question playback |
| `COMM_TRAINING_QUESTION_TRIGGER_CONFIDENCE` | `0.8` | Minimum confidence for the gesture trigger |
| `COMM_TRAINING_QUESTION_TRIGGER_COOLDOWN_SECONDS` | `5.0` | Minimum time between gesture triggers |
| `COMM_TRAINING_TRIGGER_EVAL_INTERVAL_SECONDS` | `1.0` | How often to evaluate the gesture trigger window |
| `COMM_TRAINING_TRIGGER_WINDOW_FRAMES` | `24` | Number of frames used for the gesture trigger decision |
| `COMM_MIC_AUTO_STOP_ON_SILENCE` | `true` | Stop caregiver question recording after speech ends |
| `COMM_MIC_MIN_QUESTION_SECONDS` | `1.5` | Minimum question-recording time before silence can stop it |
| `COMM_MIC_SILENCE_SECONDS` | `1.0` | Length of trailing silence that ends caregiver question recording |
| `COMM_MIC_SILENCE_RMS_THRESHOLD` | `350` | Loudness threshold used for speech/silence detection |
| `COMM_FBDEV` | `/dev/fb0` | Framebuffer device for the SPI display. Pi 5: `/dev/fb0` (default). Pi 4 and earlier: `/dev/fb1` |

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

---

## Desktop Icon & Autostart (Raspberry Pi OS Desktop)

### Add a desktop icon you can tap to launch

```bash
# 1. Edit the launcher — replace 'kiranr' with your actual username
nano ~/comm-device/comm-device.desktop

# 2. Copy to the desktop
cp ~/comm-device/comm-device.desktop ~/Desktop/comm-device.desktop
chmod +x ~/Desktop/comm-device.desktop

# 3. Trust the launcher (Raspberry Pi OS will ask on first double-tap)
#    Just tap "Execute" when prompted
```

### Auto-launch when the desktop starts

```bash
mkdir -p ~/.config/autostart
cp ~/comm-device/comm-device.desktop ~/.config/autostart/
```

The app will open automatically on the next login/reboot.

### Run from SSH with display & audio access

```bash
cd ~/comm-device
./run.sh
```

`run.sh` automatically finds the active Wayland socket and PipeWire session so display and audio work over SSH without extra setup.

### Quit the app

- **Touch/click** the red **✕** button in the top-right corner of the screen
- **Keyboard**: press **ESC** or **Q**
- **SSH**: `pkill -f comm_device`

---

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
| `E: Unable to locate package libatlas-base-dev` during `install.sh` | Raspberry Pi OS **Bookworm** — package removed in Debian 12 | `install.sh` is now Bookworm-aware and skips this package automatically. Pull latest and re-run `./install.sh` |
| `E: Package 'jq' has no installation candidate` | Stale apt cache or minimal image | Run `sudo apt-get update` then re-run `./install.sh`; jq failure is now non-fatal |
| `E: Package 'pulseaudio-utils' has no installation candidate` | Bookworm replaced PulseAudio with PipeWire | Install `pipewire-pulse` instead: `sudo apt-get install pipewire-pulse` |
| `E: Unable to locate package portaudio19-dev` | Bookworm — not needed anyway | Safe to ignore; `portaudio19-dev` is not required by this project |
| `python3-rpi.gpio` selected as `python3-rpi-lgpio` | Expected on Bookworm — same package, renamed | No action needed |
| `picamera2 not available` in logs | picamera2 not installed or camera not connected | Confirm the camera ribbon is seated; `sudo apt install python3-picamera2` |
| `RPi.GPIO not available` in logs | Not running on real Pi hardware, or package missing | `sudo apt install python3-rpi-lgpio` (Bookworm) or `python3-rpi.gpio` (Bullseye) |
| No audio playback | `piper` binary not found or Bluetooth not connected | Install `piper-tts` in the venv: `pip install piper-tts`; check `bluetoothctl` connection |
| Display shows nothing on `/dev/fb1` | SPI not enabled | Run `sudo raspi-config` → Interface Options → SPI → Enable → reboot |
| `GGUF model not found` warning | Model not yet downloaded | Follow Step 3b to download model files |
| `Face landmarker model not found` | Model missing | Run the curl command from Step 3b Option A |
| LLM responses are slow | Expected on Pi 5 without GPU | Normal; Q4_K_M quantisation keeps it to ~2–5 s per response |
| `classifier.pkl` missing | Never trained | Run `python scripts/train_classifier.py` |
| `pygame.error: fbdev not available` / `windows not available` crash | No SPI display and no desktop session running | Fixed in latest code — pull and restart. The app now falls back to text-only mode on headless setups. |
| App crashes on startup with pygame error | No display attached | Attach HDMI monitor or SPI display; the app falls back to a regular window if `/dev/fb1` is unavailable |
