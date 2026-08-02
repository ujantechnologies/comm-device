from __future__ import annotations

import faulthandler
import logging
import queue
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from .asl_intent import (
    AslIntentStore,
    extract_window_feature,
    format_intent_counts,
    parse_intents,
    record_mic_audio,
    transcribe_audio,
)
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
_CRASH_LOG_FH: Optional[object] = None


def _setup_logging() -> Path:
    """Configure console + file logging with crash traces.

    Returns:
        Path to the main log file.
    """
    log_dir = Path("artifacts")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "comm_device.log"
    crash_path = log_dir / "comm_device_crash.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Capture unhandled exceptions in main thread.
    def _handle_excepthook(exc_type: object, exc_value: object, exc_tb: object) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        root.error(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )

    sys.excepthook = _handle_excepthook

    # Capture unhandled exceptions from background threads.
    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        root.error(
            "Unhandled thread exception in '%s'",
            args.thread.name if args.thread else "<unknown>",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _handle_thread_exception

    # Dump fatal signals (segfault, bus error, etc.) to a dedicated file.
    global _CRASH_LOG_FH
    _CRASH_LOG_FH = crash_path.open("a", encoding="utf-8")
    faulthandler.enable(file=_CRASH_LOG_FH, all_threads=True)

    root.info("Logging initialized. Main log: %s | Crash log: %s", log_path, crash_path)
    return log_path

# Inter-thread queues
_frame_q: queue.Queue[FrameData] = queue.Queue(maxsize=2)
_result_q: queue.Queue[ExpressionResult] = queue.Queue(maxsize=1)
_event_q: queue.Queue[ExpressionResult] = queue.Queue(maxsize=1)
_display_q: queue.Queue[tuple[ExpressionResult, str, str]] = queue.Queue(maxsize=1)
_question_q: queue.Queue[str] = queue.Queue(maxsize=1)
_question_capture_req_q: queue.Queue[bool] = queue.Queue(maxsize=1)
_question_capture_result_q: queue.Queue[tuple[bool, str, str]] = queue.Queue(maxsize=1)
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
    tts_response: TtsService,
    store: DataStore,
    feedback: FeedbackService,
    audio: AudioRouter,
) -> None:
    # In app mode we do not auto-speak generated prompts at startup.
    # Caregiver-led questions can come from the microphone workflow scripts,
    # while this loop focuses on producing spoken responses from detected intent.
    question = ""

    while not _stop.is_set():
        # Keep the most recent caregiver question from mic capture.
        try:
            while True:
                question = _question_q.get_nowait()
        except queue.Empty:
            pass

        try:
            event = _event_q.get(timeout=0.1)
        except queue.Empty:
            continue

        answer = event.label.value  # "yes" or "no"
        logger.info("User answered '%s' (conf=%.2f) to: %s", answer, event.confidence, question)

        if not question.strip():
            # Ignore gestures until caregiver question is captured in COMM mode.
            logger.info("Ignoring gesture because no caregiver question is active")
            continue

        if question.strip():
            response_text = llm.generate_intent_response(question, answer)
        else:
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

        response_audio = tts_response.synthesize(response_text)
        audio.play(response_audio)


def _question_capture_thread(
    cfg: object,
    tts_question: TtsService,
    audio: AudioRouter,
) -> None:
    while not _stop.is_set():
        try:
            _question_capture_req_q.get(timeout=0.1)
        except queue.Empty:
            continue

        q_audio = "/tmp/comm_question.wav"
        ok = record_mic_audio(
            output_path=q_audio,
            seconds=max(1, getattr(cfg, "mic_question_seconds")),
            target=getattr(cfg, "audio_input_target"),
        )

        if not ok:
            try:
                _question_capture_result_q.put_nowait((False, "", "Mic capture failed. Check Bluetooth mic target."))
            except queue.Full:
                pass
            continue

        q_text = transcribe_audio(
            q_audio,
            model_name=getattr(cfg, "whisper_model_name"),
            language="en",
        ).strip()

        if not q_text:
            try:
                _question_capture_result_q.put_nowait((False, "", "No speech recognized. Tap REC and try again."))
            except queue.Full:
                pass
            continue

        q_out = tts_question.synthesize(q_text, output_path="artifacts/question.wav")
        audio.play(q_out)
        try:
            _question_capture_result_q.put_nowait((True, q_text, "Question captured. Waiting for user response..."))
        except queue.Full:
            pass


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run() -> None:
    _setup_logging()
    cfg = load_config()

    camera = CameraService(imx500_config_path=cfg.imx500_config_path)
    expression = ExpressionService(model_path=cfg.face_landmarker_path)
    llm = LlmService(cfg.model_path)
    tts_question = TtsService(cfg.question_voice_model_path)
    tts_response = TtsService(cfg.response_voice_model_path)
    display = DisplayService(cfg.display_width, cfg.display_height, cfg.fbdev)
    store = DataStore(cfg.db_path)
    feedback = FeedbackService(cfg.feedback_window_seconds)
    audio = AudioRouter(output_target=cfg.audio_output_target)
    training_store = AslIntentStore(
        dataset_path=cfg.asl_intent_dataset_path,
        model_path=cfg.asl_intent_model_path,
    )

    camera.start()

    threads = [
        threading.Thread(target=_camera_thread, args=(camera,), daemon=True, name="camera"),
        threading.Thread(target=_expression_thread, args=(expression,), daemon=True, name="expression"),
        threading.Thread(
            target=_llm_tts_thread,
            args=(llm, tts_response, store, feedback, audio),
            daemon=True,
            name="llm_tts",
        ),
        threading.Thread(
            target=_question_capture_thread,
            args=(cfg, tts_question, audio),
            daemon=True,
            name="question_capture",
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

    # In-app training mode state
    training_mode = False
    intent_labels = parse_intents(cfg.asl_training_intents) or ["yes", "no", "help"]
    intent_idx = 0
    training_status = "COMM mode"
    capture_active = False
    capture_warmup_end = 0.0
    capture_end = 0.0
    capture_frames: list["np.ndarray"] = []
    question_capture_active = False

    # Show any existing dataset stats once at startup
    X0, y0 = training_store.load_samples()
    if len(X0):
        training_status = f"samples={len(X0)} [{format_intent_counts(y0)}]"

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

            action = display.consume_action()
            if action == "toggle_mode":
                training_mode = not training_mode
                mode_name = "TRAIN" if training_mode else "COMM"
                training_status = f"{mode_name} mode"
            elif action == "next_intent":
                intent_idx = (intent_idx + 1) % len(intent_labels)
                training_status = f"intent={intent_labels[intent_idx]}"
            elif action == "capture_sample" and training_mode and not capture_active:
                capture_active = True
                now = time.monotonic()
                capture_warmup_end = now + max(0.0, cfg.asl_warmup_seconds)
                capture_end = capture_warmup_end + max(1, cfg.asl_response_window_seconds)
                capture_frames = []
                training_status = (
                    f"recording intent={intent_labels[intent_idx]} "
                    f"warmup={cfg.asl_warmup_seconds:.1f}s "
                    f"window={cfg.asl_response_window_seconds}s"
                )
            elif action == "capture_sample" and not training_mode and not question_capture_active:
                question_capture_active = True
                training_status = f"Listening to caregiver question ({cfg.mic_question_seconds}s)..."
                try:
                    _question_capture_req_q.put_nowait(True)
                except queue.Full:
                    pass
            elif action == "fit_model" and training_mode:
                min_samples = max(20, len(intent_labels) * 6)
                ok = training_store.train(min_samples=min_samples)
                if ok:
                    training_status = f"trained model at {cfg.asl_intent_model_path}"
                else:
                    Xc, _yc = training_store.load_samples()
                    training_status = f"need >= {min_samples} samples (have {len(Xc)})"

            if capture_active and training_mode:
                now = time.monotonic()
                if now < capture_warmup_end:
                    remaining = max(0.0, capture_warmup_end - now)
                    training_status = (
                        f"warmup {remaining:.1f}s intent={intent_labels[intent_idx]}"
                    )
                elif now <= capture_end:
                    if raw_frame is not None:
                        capture_frames.append(raw_frame.copy())
                    remaining = max(0.0, capture_end - now)
                    training_status = (
                        f"capturing {remaining:.1f}s intent={intent_labels[intent_idx]} "
                        f"frames={len(capture_frames)}"
                    )
                else:
                    capture_active = False
                    if capture_frames:
                        feat = extract_window_feature(capture_frames)
                        intent = intent_labels[intent_idx]
                        training_store.append_sample(intent, feat)
                        Xc, yc = training_store.load_samples()
                        training_status = (
                            f"saved '{intent}' total={len(Xc)} "
                            f"[{format_intent_counts(yc)}]"
                        )
                    else:
                        training_status = "capture failed: no frames"

            if question_capture_active and not training_mode:
                try:
                    ok, q_text, status = _question_capture_result_q.get_nowait()
                    question_capture_active = False
                    training_status = status
                    if ok and q_text:
                        current_question = q_text
                        try:
                            _question_q.put_nowait(q_text)
                        except queue.Full:
                            try:
                                _question_q.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                _question_q.put_nowait(q_text)
                            except queue.Full:
                                pass
                        _put_display(None, current_question, current_response)
                except queue.Empty:
                    pass

            display.render(
                display_frame_id,
                current_result,
                "TRAIN" if training_mode else "COMM",
                current_question,
                current_response,
                intent_labels[intent_idx],
                training_status,
                raw_frame,
            )
            if display.should_quit:
                logger.info("Quit requested from display — shutting down")
                break
            time.sleep(1 / 15)  # ~15 FPS

    except Exception:
        logger.exception("Fatal crash in main loop")
        raise
    finally:
        _stop.set()
        camera.stop()
        feedback.cleanup()
        display.quit()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    run()
