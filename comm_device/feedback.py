from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .expression import ExpressionLabel

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO  # type: ignore[import]
    _HAS_GPIO = True
except (ImportError, RuntimeError):
    _HAS_GPIO = False
    logger.warning("RPi.GPIO not available — feedback auto-accepts all predictions")

_PIN_CORRECT = 17  # BCM — physical pin 11
_PIN_WRONG = 27    # BCM — physical pin 13

_OPPOSITE: dict[ExpressionLabel, ExpressionLabel] = {
    ExpressionLabel.YES: ExpressionLabel.NO,
    ExpressionLabel.NO: ExpressionLabel.YES,
    ExpressionLabel.UNCERTAIN: ExpressionLabel.UNCERTAIN,
}


@dataclass
class FeedbackEvent:
    frame_id: int
    predicted_label: ExpressionLabel
    corrected_label: Optional[ExpressionLabel]  # None → auto-accepted (timeout)


class FeedbackService:
    """Waits up to *window_seconds* for a GPIO button press after each prediction.

    Wiring:
      GPIO 17 (BCM) — CORRECT button (pull-up, press to GND)
      GPIO 27 (BCM) — WRONG  button (pull-up, press to GND)

    When WRONG is pressed the corrected label is the opposite of the prediction.
    No press within the window → prediction accepted as-is.
    """

    def __init__(self, window_seconds: int = 5) -> None:
        self._window = window_seconds
        if _HAS_GPIO:
            GPIO.cleanup()  # release any pins left claimed by a previous crashed run
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(_PIN_CORRECT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(_PIN_WRONG, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def collect(
        self,
        frame_id: int,
        predicted_label: ExpressionLabel,
    ) -> FeedbackEvent:
        if not _HAS_GPIO:
            return FeedbackEvent(
                frame_id=frame_id,
                predicted_label=predicted_label,
                corrected_label=predicted_label,
            )

        correct_ev = threading.Event()
        wrong_ev = threading.Event()

        GPIO.add_event_detect(_PIN_CORRECT, GPIO.FALLING, callback=lambda _: correct_ev.set(), bouncetime=200)
        GPIO.add_event_detect(_PIN_WRONG, GPIO.FALLING, callback=lambda _: wrong_ev.set(), bouncetime=200)

        deadline = time.monotonic() + self._window
        corrected: Optional[ExpressionLabel] = predicted_label

        while time.monotonic() < deadline:
            if correct_ev.is_set():
                logger.debug("Feedback: CORRECT")
                break
            if wrong_ev.is_set():
                corrected = _OPPOSITE[predicted_label]
                logger.info(
                    "Feedback: WRONG — corrected %s → %s",
                    predicted_label.value,
                    corrected.value,
                )
                break
            time.sleep(0.05)
        else:
            logger.debug("Feedback: timeout — auto-accepted %s", predicted_label.value)

        GPIO.remove_event_detect(_PIN_CORRECT)
        GPIO.remove_event_detect(_PIN_WRONG)

        return FeedbackEvent(
            frame_id=frame_id,
            predicted_label=predicted_label,
            corrected_label=corrected,
        )

    def cleanup(self) -> None:
        if _HAS_GPIO:
            GPIO.cleanup()
            logger.info("GPIO cleaned up")
