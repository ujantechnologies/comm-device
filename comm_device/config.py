from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    display_width: int = int(os.getenv("COMM_DISPLAY_WIDTH", "480"))
    display_height: int = int(os.getenv("COMM_DISPLAY_HEIGHT", "320"))
    # Gemma 3 1B GGUF (quantized) — download via install.sh
    model_path: str = os.getenv("COMM_MODEL_PATH", "models/gemma3-1b-Q4_K_M.gguf")
    # Piper voice model (.onnx) — download via install.sh
    voice_model_path: str = os.getenv("COMM_VOICE_MODEL_PATH", "models/en_US-lessac-medium.onnx")
    # Sklearn classifier persisted via train_classifier.py
    classifier_path: str = os.getenv("COMM_CLASSIFIER_PATH", "models/classifier.pkl")
    # MediaPipe face landmarker task file — download via install.sh
    face_landmarker_path: str = os.getenv("COMM_FACE_LANDMARKER_PATH", "models/face_landmarker.task")
    # Optional IMX500 face-detection JSON (leave empty to skip IMX500 DNN)
    imx500_config_path: str = os.getenv("COMM_IMX500_CONFIG_PATH", "")
    db_path: str = os.getenv("COMM_DB_PATH", "artifacts/comm_device.sqlite3")
    # Seconds to wait for GPIO feedback after a prediction
    feedback_window_seconds: int = int(os.getenv("COMM_FEEDBACK_WINDOW_SECONDS", "5"))
    # Minimum confidence to fire LLM+TTS pipeline
    expression_confidence_threshold: float = float(os.getenv("COMM_EXPR_CONF_THRESHOLD", "0.6"))
    # Minimum seconds between successive LLM invocations
    llm_cooldown_seconds: float = float(os.getenv("COMM_LLM_COOLDOWN_SECONDS", "3.0"))
    # Optional explicit audio output target (PipeWire node name), e.g.
    # bluez_output.CF_57_28_DC_04_87.1
    audio_output_target: str = os.getenv("COMM_AUDIO_OUTPUT_TARGET", "")
    # Framebuffer device for the SPI display.
    # Pi 5: legacy fb0 (bcm2708_fb doesn't load, SPI display takes fb0).
    # Pi 4 and earlier: fb1 (VC4 firmware claims fb0 first).
    fbdev: str = os.getenv("COMM_FBDEV", "/dev/fb0")


def load_config() -> AppConfig:
    return AppConfig()
