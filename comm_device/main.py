from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from typing import Optional

from .audio_router import AudioRouter
from .camera import CameraService, FrameData
from .config import load_config
from .data_store import DataStore
from .display import DisplayService
from .expression import ExpressionLabel, ExpressionResult, ExpressionService
from .feedback import FeedbackService
from .llm import LlmService
from .tts import TtsService

logger = logging.getLogger(__name__)

# Inter-thread queues
_frame_q: queue.Queue[FrameData] = queue.Queue(maxsize=2)
_result_q: queue.Queue[ExpressionResult] = queue.Queue(maxsize=1)
_event_q: queue.Queue[ExpressionResult] = queue.Queue(maxsize=1)
_display_q: queue.Queue[tuple[ExpressionResult, str, str]] = queue.Queue(maxsize=1)
# Separate queue so the display loop gets raw frames without competing with the
# expression thread, which exclusively drains _frame_q.
_display_frame_q: queue.Queue["np.ndarray"] = queue.Queue(maxsize=1)
_stop = threading.Event()


# ------------------------------------------------------------------
# Worker threads
# ------------------------------------------------------------------

def _camera_thread(camera: CameraService) -> None:
    while not _stop.is_set():
        data = camera.read()
        if data is None:
            time.sleep(0.01)
            continue
        try:
            _frame_q.put_nowait(data)
        except queue.Full:
            pass  # newer frames are more valuable; drop the stale queue entry


def _expression_thread(svc: ExpressionService) -> None:
    last_fire = 0.0
    cooldown = 3.0  # seconds between YES/NO events to avoid flooding LLM

    while not _stop.is_set():
        try:
            data = _frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        result = svc.infer(data.frame, data.frame_id)

        # Share raw frame with the display loop (non-blocking replace)
        try:
            _display_frame_q.put_nowait(data.frame)
        except queue.Full:
            try:
                _display_frame_q.get_nowait()
            except queue.Empty:
                pass
            _display_frame_q.put_nowait(data.frame)

        # Push latest result for display (non-blocking replace)
        try:
            _result_q.put_nowait(result)
        except queue.Full:
            try:
                _result_q.get_nowait()
            except queue.Empty:
                pass
            try:
                _result_q.put_nowait(result)
            except queue.Full:
                pass

        now = time.monotonic()
        if (
            result.label in (ExpressionLabel.YES, ExpressionLabel.NO)
            and result.confidence >= 0.6
            and now - last_fire >= cooldown
        ):
            last_fire = now
            try:
                _event_q.put_nowait(result)
            except queue.Full:
                pass


def _put_display(event: Optional[ExpressionResult], question: str, response: str) -> None:
    """Non-blocking update of the display queue."""
    sentinel = event or ExpressionResult(
        label=ExpressionLabel.UNCERTAIN, confidence=0.0, frame_id=0
    )
    try:
        _display_q.put_nowait((sentinel, question, response))
    except queue.Full:
        try:
            _display_q.get_nowait()
        except queue.Empty:
            pass
        try:
            _display_q.put_nowait((sentinel, question, response))
        except queue.Full:
            pass


def _llm_tts_thread(
    llm: LlmService,
    tts: TtsService,
    store: DataStore,
    feedback: FeedbackService,
    audio: AudioRouter,
) -> None:
    history: list[tuple[str, str]] = []  # (question, answer) pairs

    # Generate and speak the opening question before waiting for any gesture
    question = llm.generate_question(history)
    logger.info("Opening question: %s", question)
    _put_display(None, question, "")
    audio_path = tts.synthesize(question)
    audio.play(audio_path)

    while not _stop.is_set():
        try:
            event = _event_q.get(timeout=0.1)
        except queue.Empty:
            continue

        answer = event.label.value  # "yes" or "no"
        logger.info("User answered '%s' (conf=%.2f) to: %s", answer, event.confidence, question)

        history.append((question, answer))

        response_text = llm.generate_response(event.label)
        logger.info("LLM response for %s: %s", event.label.value, response_text)
        store.save_response(event.label, response_text)

        fb = feedback.collect(event.frame_id, event.label)
        store.save_prediction(
            event.frame_id,
            event.label,
            event.confidence,
            corrected_label=fb.corrected_label,
            features=event.features,
        )

        # Show the user's response and the final response on the display
        _put_display(event, question, response_text)

        response_audio = tts.synthesize(response_text)
        audio.play(response_audio)

        # Generate and speak the follow-up question
        question = llm.generate_question(history)
        logger.info("Next question: %s", question)
        _put_display(None, question, response_text)
        audio_path = tts.synthesize(question)
        audio.play(audio_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    cfg = load_config()

    camera = CameraService(imx500_config_path=cfg.imx500_config_path)
    expression = ExpressionService(model_path=cfg.face_landmarker_path)
    llm = LlmService(cfg.model_path)
    tts = TtsService(cfg.voice_model_path)
    display = DisplayService(cfg.display_width, cfg.display_height, cfg.fbdev)
    store = DataStore(cfg.db_path)
    feedback = FeedbackService(cfg.feedback_window_seconds)
    audio = AudioRouter()

    camera.start()

    threads = [
        threading.Thread(target=_camera_thread, args=(camera,), daemon=True, name="camera"),
        threading.Thread(target=_expression_thread, args=(expression,), daemon=True, name="expression"),
        threading.Thread(
            target=_llm_tts_thread,
            args=(llm, tts, store, feedback, audio),
            daemon=True,
            name="llm_tts",
        ),
    ]
    for t in threads:
        t.start()
    logger.info("All threads started")

    def _shutdown(sig: int, _frame: object) -> None:
        logger.info("Signal %d received — shutting down", sig)
        _stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    current_result = ExpressionResult(label=ExpressionLabel.UNCERTAIN, confidence=0.0)
    current_question = ""
    current_response = ""
    display_frame_id = 0

    try:
        while not _stop.is_set():
            display_frame_id += 1

            try:
                current_result = _result_q.get_nowait()
            except queue.Empty:
                pass

            try:
                ev_result, current_question, current_response = _display_q.get_nowait()
                current_result = ev_result
            except queue.Empty:
                pass

            # Grab the most recent raw frame for display (best-effort)
            raw_frame = None
            try:
                raw_frame = _display_frame_q.get_nowait()
            except queue.Empty:
                pass

            display.render(
                display_frame_id,
                current_result,
                current_question,
                current_response,
                raw_frame,
            )
            if display.should_quit:
                logger.info("Quit requested from display — shutting down")
                break
            time.sleep(1 / 15)  # ~15 FPS

    finally:
        _stop.set()
        camera.stop()
        feedback.cleanup()
        display.quit()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    run()
