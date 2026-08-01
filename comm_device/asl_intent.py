from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .camera import CameraService

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    intent: str
    confidence: float


def extract_window_feature(frames: list[np.ndarray]) -> np.ndarray:
    """Create a compact feature vector from a response window.

    Uses low-res grayscale summary + temporal motion statistics so it can run
    on Raspberry Pi without requiring heavy landmark models.
    """
    if not frames:
        return np.zeros((1154,), dtype=np.float32)

    # Downsample to keep feature size manageable.
    gray_frames = []
    for frame in frames:
        # frame is RGB uint8; luma approximation
        gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.float32)
        # Nearest-neighbor resize by slicing for speed
        h, w = gray.shape
        ys = np.linspace(0, h - 1, 24).astype(np.int32)
        xs = np.linspace(0, w - 1, 24).astype(np.int32)
        gray_small = gray[np.ix_(ys, xs)] / 255.0
        gray_frames.append(gray_small)

    stack = np.stack(gray_frames, axis=0)  # T x 24 x 24
    mean_img = stack.mean(axis=0).reshape(-1)
    std_img = stack.std(axis=0).reshape(-1)

    if len(stack) > 1:
        diffs = np.abs(np.diff(stack, axis=0))
        motion_mean = float(diffs.mean())
        motion_std = float(diffs.std())
    else:
        motion_mean = 0.0
        motion_std = 0.0

    feat = np.concatenate(
        [
            mean_img,
            std_img,
            np.array([motion_mean, motion_std], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return feat


def capture_response_window(
    camera: CameraService,
    seconds: int,
    fps_limit: float = 8.0,
    warmup_seconds: float = 1.0,
) -> np.ndarray:
    """Capture a timed response window and return one feature vector.

    warmup_seconds gives slower users time to begin signing.
    """
    if warmup_seconds > 0:
        time.sleep(warmup_seconds)

    frames: list[np.ndarray] = []
    deadline = time.monotonic() + seconds
    min_interval = 1.0 / max(1.0, fps_limit)
    next_ts = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_ts:
            time.sleep(min(0.01, next_ts - now))
            continue

        fd = camera.read()
        if fd is not None:
            frames.append(fd.frame.copy())
        next_ts = now + min_interval

    return extract_window_feature(frames)


class AslIntentStore:
    """Stores raw training samples and trains an intent classifier."""

    def __init__(self, dataset_path: str, model_path: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def append_sample(self, intent: str, feature: np.ndarray) -> None:
        X, y = self.load_samples()
        if X.size == 0:
            X = feature.reshape(1, -1)
            y = np.array([intent], dtype=object)
        else:
            X = np.vstack([X, feature.reshape(1, -1)])
            y = np.concatenate([y, np.array([intent], dtype=object)])
        np.savez(self.dataset_path, X=X.astype(np.float32), y=y)

    def load_samples(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.dataset_path.exists():
            return np.empty((0, 1154), dtype=np.float32), np.empty((0,), dtype=object)
        data = np.load(self.dataset_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"]
        return X, y

    def train(self, min_samples: int = 20) -> bool:
        X, y = self.load_samples()
        if len(X) < min_samples:
            logger.warning("Need at least %d samples to train; found %d", min_samples, len(X))
            return False
        unique = np.unique(y)
        if len(unique) < 2:
            logger.warning("Need at least 2 intent classes to train; found %d", len(unique))
            return False

        model: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True)),
            ]
        )
        model.fit(X, y)
        joblib.dump(model, self.model_path)
        logger.info("Saved ASL intent model to %s", self.model_path)
        return True


class AslIntentClassifier:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self._model: Optional[Pipeline] = None
        if self.model_path.exists():
            self._model = joblib.load(self.model_path)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, feature: np.ndarray) -> IntentPrediction:
        if self._model is None:
            return IntentPrediction(intent="unknown", confidence=0.0)

        probs = self._model.predict_proba(feature.reshape(1, -1))[0]
        classes = self._model.classes_
        idx = int(np.argmax(probs))
        return IntentPrediction(intent=str(classes[idx]), confidence=float(probs[idx]))


def record_mic_audio(output_path: str, seconds: int, target: str = "") -> bool:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["pw-record"]
    if target:
        cmd.extend(["--target", target])
    cmd.append(str(out))

    proc = subprocess.Popen(cmd)
    try:
        time.sleep(seconds)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)

    return out.exists() and out.stat().st_size > 0


def transcribe_audio(path: str, model_name: str = "tiny", language: str = "en") -> str:
    try:
        import whisper
    except ImportError:
        logger.warning("whisper not installed; returning empty transcription")
        return ""

    model = whisper.load_model(model_name)
    result = model.transcribe(path, fp16=False, language=language or None)
    return str(result.get("text", "")).strip()


def play_audio(path: str, target: str = "") -> None:
    if target:
        subprocess.run(["pw-play", "--target", target, path], check=False)
    else:
        subprocess.run(["pw-play", path], check=False)


def parse_intents(raw: str) -> list[str]:
    intents = [x.strip() for x in raw.split(",") if x.strip()]
    return intents


def format_intent_counts(labels: Iterable[str]) -> str:
    vals = list(labels)
    if not vals:
        return "<none>"
    parts: list[str] = []
    for intent in sorted(set(vals)):
        parts.append(f"{intent}:{vals.count(intent)}")
    return ", ".join(parts)
