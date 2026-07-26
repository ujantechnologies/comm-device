from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    display_width: int = int(os.getenv("COMM_DISPLAY_WIDTH", "480"))
    display_height: int = int(os.getenv("COMM_DISPLAY_HEIGHT", "320"))
    model_path: str = os.getenv("COMM_MODEL_PATH", "models/gemma3-1b-Q4_K_M.gguf")
    voice_model_path: str = os.getenv("COMM_VOICE_MODEL_PATH", "models/en_US-lessac-medium.onnx")
    classifier_path: str = os.getenv("COMM_CLASSIFIER_PATH", "models/classifier.pkl")
    db_path: str = os.getenv("COMM_DB_PATH", "artifacts/comm_device.sqlite3")
    capture_device: str = os.getenv("COMM_CAPTURE_DEVICE", "imx500")
    feedback_window_seconds: int = int(os.getenv("COMM_FEEDBACK_WINDOW_SECONDS", "5"))


def load_config() -> AppConfig:
    return AppConfig()
