from __future__ import annotations

import logging
import queue
import signal
import threading
import time
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

# Inter-thread queues
_frame_q: queue.Queue[FrameData] = queue.Queue(maxsize=2)
_result_q: queue.Queue[ExpressionResult] = queue.Queue(maxsize=1)
_event_q: queue.Queue[ExpressionResult] = queue.Queue(maxsize=1)
_display_q: queue.Queue[tuple[ExpressionResult, str, str]] = queue.Queue(maxsize=1)
_question_q: queue.Queue[str] = queue.Queue(maxsize=1)
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

                q_audio = "/tmp/comm_question.wav"
                ok = record_mic_audio(
                    output_path=q_audio,
                    seconds=max(1, cfg.mic_question_seconds),
                    target=cfg.audio_input_target,
                )
                if ok:
                    q_text = transcribe_audio(
                        q_audio,
                        model_name=cfg.whisper_model_name,
                        language="en",
                    )
                    q_text = q_text.strip()
                    if q_text:
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
                        q_out = tts_question.synthesize(current_question, output_path="artifacts/question.wav")
                        audio.play(q_out)
                        training_status = "Question captured. Waiting for user ASL response..."
                    else:
                        training_status = "No speech recognized. Tap REC and try again."
                else:
                    training_status = "Mic capture failed. Check Bluetooth mic target."
                question_capture_active = False
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

    finally:
        _stop.set()
        camera.stop()
        feedback.cleanup()
        display.quit()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    run()
