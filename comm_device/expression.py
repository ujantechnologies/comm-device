from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    logger.warning("mediapipe not available — expression detection using synthetic output")

# Tuning knobs
_WINDOW = 15          # frames held in sliding window
_NOD_THRESHOLD = 0.04  # min normalised Y range to register a nod (YES)
_SHAKE_THRESHOLD = 0.04  # min normalised X range to register a shake (NO)
_DOMINANCE = 1.5      # primary axis must be this many times larger than secondary

# Nose tip landmark index in MediaPipe Face Landmarker output
_NOSE_IDX = 1


class ExpressionLabel(str, Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


@dataclass
class ExpressionResult:
    label: ExpressionLabel
    confidence: float
    features: Optional[np.ndarray] = None  # stored for offline retraining


class ExpressionService:
    """Detects yes/no head gestures via MediaPipe Face Landmarker.

    Uses a sliding window over the nose-tip (x, y) trajectory:
    - Significant vertical motion  → nod  → YES
    - Significant horizontal motion → shake → NO

    Falls back to synthetic output when MediaPipe is unavailable.
    """

    def __init__(self, model_path: str = "models/face_landmarker.task") -> None:
        self._landmarker: Optional[object] = None
        self._nose_x: deque[float] = deque(maxlen=_WINDOW)
        self._nose_y: deque[float] = deque(maxlen=_WINDOW)

        if _HAS_MEDIAPIPE:
            self._init_landmarker(model_path)

    def _init_landmarker(self, model_path: str) -> None:
        import os
        if not os.path.exists(model_path):
            logger.warning(
                "Face landmarker model not found at %s; run install.sh to download it",
                model_path,
            )
            return
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        logger.info("Face landmarker loaded from %s", model_path)

    def infer(self, frame: np.ndarray, frame_id: int) -> ExpressionResult:
        if _HAS_MEDIAPIPE and self._landmarker is not None:
            return self._infer_real(frame)
        return self._infer_synthetic(frame_id)

    # ------------------------------------------------------------------
    # Real MediaPipe inference
    # ------------------------------------------------------------------

    def _infer_real(self, frame: np.ndarray) -> ExpressionResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            self._nose_x.append(0.5)
            self._nose_y.append(0.5)
            return ExpressionResult(label=ExpressionLabel.UNCERTAIN, confidence=0.0)

        lms = result.face_landmarks[0]
        nose = lms[_NOSE_IDX]
        self._nose_x.append(float(nose.x))
        self._nose_y.append(float(nose.y))

        features = np.array(
            [nose.x, nose.y, nose.z,
             max(self._nose_x) - min(self._nose_x),
             max(self._nose_y) - min(self._nose_y)],
            dtype=np.float32,
        )
        label, conf = self._classify_motion()
        return ExpressionResult(label=label, confidence=conf, features=features)

    def _classify_motion(self) -> tuple[ExpressionLabel, float]:
        if len(self._nose_y) < _WINDOW:
            return ExpressionLabel.UNCERTAIN, 0.0

        y_range = float(max(self._nose_y) - min(self._nose_y))
        x_range = float(max(self._nose_x) - min(self._nose_x))

        if y_range >= _NOD_THRESHOLD and y_range > x_range * _DOMINANCE:
            conf = min(1.0, y_range / (_NOD_THRESHOLD * 2))
            return ExpressionLabel.YES, round(conf, 3)

        if x_range >= _SHAKE_THRESHOLD and x_range > y_range * _DOMINANCE:
            conf = min(1.0, x_range / (_SHAKE_THRESHOLD * 2))
            return ExpressionLabel.NO, round(conf, 3)

        return ExpressionLabel.UNCERTAIN, 0.0

    # ------------------------------------------------------------------
    # Synthetic fallback (development / CI)
    # ------------------------------------------------------------------

    def _infer_synthetic(self, frame_id: int) -> ExpressionResult:
        if frame_id % 40 == 0:
            return ExpressionResult(label=ExpressionLabel.NO, confidence=0.82)
        if frame_id % 20 == 0:
            return ExpressionResult(label=ExpressionLabel.YES, confidence=0.87)
        return ExpressionResult(label=ExpressionLabel.UNCERTAIN, confidence=0.40)
