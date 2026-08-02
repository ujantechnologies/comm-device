from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    display_width: int = int(os.getenv("COMM_DISPLAY_WIDTH", "480"))
    display_height: int = int(os.getenv("COMM_DISPLAY_HEIGHT", "320"))
    # Gemma 3 1B GGUF (quantized) — download via install.sh
    model_path: str = os.getenv("COMM_MODEL_PATH", "models/gemma3-1b-Q4_K_M.gguf")
    # Piper voice model (.onnx) — download via install.sh
    voice_model_path: str = os.getenv("COMM_VOICE_MODEL_PATH", "models/en_US-lessac-medium.onnx")
    # Optional dedicated voice model for caregiver question playback (Voice 1)
    question_voice_model_path: str = os.getenv(
        "COMM_QUESTION_VOICE_MODEL_PATH",
        os.getenv("COMM_VOICE_MODEL_PATH", "models/en_US-lessac-medium.onnx"),
    )
    # Optional dedicated voice model for user response playback (Voice 2)
    response_voice_model_path: str = os.getenv(
        "COMM_RESPONSE_VOICE_MODEL_PATH",
        os.getenv("COMM_VOICE_MODEL_PATH", "models/en_US-lessac-medium.onnx"),
    )
    # Sklearn classifier persisted via train_classifier.py
    classifier_path: str = os.getenv("COMM_CLASSIFIER_PATH", "models/classifier.pkl")
    # MediaPipe face landmarker task file — download via install.sh
    face_landmarker_path: str = os.getenv("COMM_FACE_LANDMARKER_PATH", "models/face_landmarker.task")
    # Optional IMX500 face-detection JSON (leave empty to skip IMX500 DNN)
    imx500_config_path: str = os.getenv("COMM_IMX500_CONFIG_PATH", "")
    # Optional explicit audio input target (PipeWire source), e.g.
    # bluez_input.CF:57:28:DC:04:87
    audio_input_target: str = os.getenv("COMM_AUDIO_INPUT_TARGET", "")
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
    # Personalized ASL intent classifier artifacts.
    asl_intent_model_path: str = os.getenv(
        "COMM_ASL_INTENT_MODEL_PATH", "models/asl_intent_classifier.pkl"
    )
    asl_intent_dataset_path: str = os.getenv(
        "COMM_ASL_INTENT_DATASET_PATH", "artifacts/asl_intent_samples.npz"
    )
    # Default capture timings for slower ASL responses.
    asl_response_window_seconds: int = int(os.getenv("COMM_ASL_RESPONSE_WINDOW_SECONDS", "10"))
    asl_warmup_seconds: float = float(os.getenv("COMM_ASL_WARMUP_SECONDS", "1.5"))
    asl_training_intents: str = os.getenv(
        "COMM_ASL_TRAINING_INTENTS", "yes,no,water,pain,rest,help,speak_question"
    )
    training_intents_file: str = os.getenv(
        "COMM_TRAINING_INTENTS_FILE", "artifacts/training_intents.txt"
    )
    training_clip_seconds: int = int(os.getenv("COMM_TRAINING_CLIP_SECONDS", "5"))
    training_video_dir: str = os.getenv(
        "COMM_TRAINING_VIDEO_DIR", "artifacts/training_videos"
    )
    training_strict_local_llm: bool = _env_bool("COMM_TRAINING_STRICT_LOCAL_LLM", True)
    training_question_temperature: float = float(
        os.getenv("COMM_TRAINING_QUESTION_TEMPERATURE", "0.65")
    )
    training_question_trigger_enabled: bool = _env_bool(
        "COMM_TRAINING_QUESTION_TRIGGER_ENABLED", True
    )
    training_question_trigger_intent: str = os.getenv(
        "COMM_TRAINING_QUESTION_TRIGGER_INTENT", "speak_question"
    )
    training_question_trigger_confidence: float = float(
        os.getenv("COMM_TRAINING_QUESTION_TRIGGER_CONFIDENCE", "0.8")
    )
    training_question_trigger_cooldown_seconds: float = float(
        os.getenv("COMM_TRAINING_QUESTION_TRIGGER_COOLDOWN_SECONDS", "5.0")
    )
    training_trigger_eval_interval_seconds: float = float(
        os.getenv("COMM_TRAINING_TRIGGER_EVAL_INTERVAL_SECONDS", "1.0")
    )
    training_trigger_window_frames: int = int(
        os.getenv("COMM_TRAINING_TRIGGER_WINDOW_FRAMES", "24")
    )
    whisper_model_name: str = os.getenv("COMM_WHISPER_MODEL", "tiny")
    mic_question_seconds: int = int(os.getenv("COMM_MIC_QUESTION_SECONDS", "10"))
    mic_auto_stop_on_silence: bool = _env_bool("COMM_MIC_AUTO_STOP_ON_SILENCE", True)
    mic_min_question_seconds: float = float(
        os.getenv("COMM_MIC_MIN_QUESTION_SECONDS", "1.5")
    )
    mic_silence_seconds: float = float(os.getenv("COMM_MIC_SILENCE_SECONDS", "1.0"))
    mic_silence_rms_threshold: int = int(
        os.getenv("COMM_MIC_SILENCE_RMS_THRESHOLD", "350")
    )
    # Framebuffer device for the SPI display.
    # Pi 5: legacy fb0 (bcm2708_fb doesn't load, SPI display takes fb0).
    # Pi 4 and earlier: fb1 (VC4 firmware claims fb0 first).
    fbdev: str = os.getenv("COMM_FBDEV", "/dev/fb0")


def load_config() -> AppConfig:
    return AppConfig()
