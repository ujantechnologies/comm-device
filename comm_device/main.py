from __future__ import annotations

import faulthandler
import logging
import queue
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from .asl_intent import (
    AslIntentClassifier,
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


def _safe_name(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw.strip().lower())
    return safe[:24] or "intent"


def _save_training_video_npz(
    video_dir: str,
    frames: list[np.ndarray],
    intent: str,
) -> tuple[bool, str]:
    if not frames:
        return False, "No frames available for training video."

    out_dir = Path(video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_{_safe_name(intent)}.npz"

    try:
        np.savez_compressed(out_path, frames=np.stack(frames, axis=0))
        return True, str(out_path)
    except Exception as exc:
        logger.error("Failed to save training video artifact: %s", exc)
        return False, "Failed to save training video artifact."


def _play_training_video_npz(
    display: DisplayService,
    path: str,
    result: ExpressionResult,
    mode: str,
    question: str,
    response: str,
    training_intent: str,
) -> tuple[bool, str]:
    video_path = Path(path)
    if not video_path.exists():
        return False, "Training video file not found."

    try:
        data = np.load(video_path, allow_pickle=False)
        frames = data["frames"]
    except Exception as exc:
        logger.error("Failed to load training video: %s", exc)
        return False, "Could not read training video file."

    if len(frames) == 0:
        return False, "Training video has no frames."

    for idx, frame in enumerate(frames):
        display.render(
            idx + 1,
            result,
            mode,
            question,
            response,
            training_intent,
            "Reviewing video... tap any button to stop",
            frame,
        )
        if display.should_quit:
            return False, "Playback interrupted: app close requested."

        action = display.consume_action()
        if action:
            break

        time.sleep(1 / 8)

    return True, f"Reviewed {len(frames)} training frames."


def _delete_files(paths: list[str]) -> int:
    removed = 0
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            removed += 1
        except Exception as exc:
            logger.error("Failed to remove artifact %s: %s", path, exc)
    return removed


def _format_training_video_list(
    videos: list[dict[str, object]],
    selected_idx: int,
) -> str:
    if not videos:
        return "No training videos saved."

    lines = [f"Videos {selected_idx + 1}/{len(videos)}"]
    for idx, video in enumerate(videos[:4]):
        prefix = ">" if idx == selected_idx else " "
        intent = str(video["intent"])[:7]
        seconds = int(round(float(video["duration_seconds"])))
        lines.append(f"{prefix}{idx + 1}. {intent} {seconds}s")
    if len(videos) > 4:
        lines.append(f"... +{len(videos) - 4} more")
    return "\n".join(lines)


def _normalize_intent_label(raw: str) -> str:
    label = raw.strip().lower().replace(" ", "_")
    label = re.sub(r"[^a-z0-9_]+", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label[:24]


def _load_extra_intents(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Could not read training intents file %s: %s", p, exc)
        return []

    out: list[str] = []
    for line in lines:
        val = _normalize_intent_label(line)
        if val:
            out.append(val)
    return out


def _save_extra_intents(path: str, intents: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(intents) + "\n", encoding="utf-8")


def _current_extra_intents(base_intents: list[str], all_intents: list[str]) -> list[str]:
    return [intent for intent in all_intents if intent not in base_intents]


def _model_update_summary(model_path: str) -> str:
    p = Path(model_path)
    if not p.exists():
        return "model file missing"
    stat = p.stat()
    mtime = time.strftime("%H:%M:%S", time.localtime(stat.st_mtime))
    return f"model {stat.st_size // 1024}KB @ {mtime}"


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
            auto_stop_on_silence=bool(getattr(cfg, "mic_auto_stop_on_silence", True)),
            min_seconds=float(getattr(cfg, "mic_min_question_seconds", 1.5)),
            silence_seconds=float(getattr(cfg, "mic_silence_seconds", 1.0)),
            silence_rms_threshold=int(getattr(cfg, "mic_silence_rms_threshold", 350)),
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
    training_trigger_classifier = AslIntentClassifier(cfg.asl_intent_model_path)

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
    base_intents = parse_intents(cfg.asl_training_intents) or ["yes", "no", "help"]
    extra_intents = _load_extra_intents(cfg.training_intents_file)
    intent_labels = list(dict.fromkeys(base_intents + extra_intents))
    intent_idx = 0
    training_status = "COMM mode"
    capture_active = False
    capture_warmup_end = 0.0
    capture_end = 0.0
    capture_frames: list["np.ndarray"] = []
    capture_question = ""
    manual_training_question = ""
    training_question_from_caregiver = False
    training_question_history: list[tuple[str, str]] = []
    trigger_frames: list[np.ndarray] = []
    trigger_last_fire = 0.0
    trigger_next_eval = 0.0
    trigger_warned_unavailable = False
    review_video_idx = -1
    training_question_capture_mode = ""
    question_capture_active = False
    pending_delete_intent = ""
    pending_delete_intent_until = 0.0

    # Show any existing dataset stats once at startup
    X0, y0 = training_store.load_samples()
    if len(X0):
        training_status = f"samples={len(X0)} [{format_intent_counts(y0)}]"

    def _begin_training_capture(source: str) -> None:
        nonlocal capture_active
        nonlocal capture_warmup_end
        nonlocal capture_end
        nonlocal capture_frames
        nonlocal capture_question
        nonlocal manual_training_question
        nonlocal training_question_from_caregiver
        nonlocal current_question
        nonlocal current_response
        nonlocal training_status

        if capture_active:
            logger.info("Ignoring training capture request from %s because capture is already active", source)
            return

        target_intent = intent_labels[intent_idx]
        logger.info(
            "Starting training capture source=%s target_intent=%s caregiver_question=%s",
            source,
            target_intent,
            bool(manual_training_question.strip()),
        )

        if manual_training_question.strip():
            question_text = manual_training_question.strip()
            training_question_from_caregiver = True
        else:
            training_question_from_caregiver = False
            if cfg.training_strict_local_llm:
                ok_q, question_text = llm.generate_training_question(
                    training_question_history,
                    target_intent,
                    temperature=cfg.training_question_temperature,
                )
                if not ok_q:
                    logger.warning("Training question generation failed for intent=%s: %s", target_intent, question_text)
                    training_status = question_text
                    return
            else:
                if llm.is_local_model_ready:
                    ok_q, question_text = llm.generate_training_question(
                        training_question_history,
                        target_intent,
                        temperature=cfg.training_question_temperature,
                    )
                else:
                    ok_q, question_text = True, llm.generate_question(training_question_history)

                if not ok_q or not question_text:
                    logger.warning("Training question generation returned no usable question for intent=%s", target_intent)
                    training_status = "Could not generate training question."
                    return

        current_question = question_text
        current_response = ""
        capture_question = question_text
        _put_display(None, current_question, current_response)

        try:
            q_out = tts_question.synthesize(
                question_text,
                output_path="artifacts/training_question.wav",
            )
            audio.play(q_out)
            logger.info("Training question spoken intent=%s text=%s", target_intent, question_text)
        except Exception as exc:
            logger.error("Training question playback error: %s", exc)
            training_status = "Question generated but playback failed."
            return

        capture_active = True
        now = time.monotonic()
        capture_warmup_end = now + max(0.0, cfg.asl_warmup_seconds)
        capture_end = capture_warmup_end + max(1, cfg.training_clip_seconds)
        capture_frames = []
        if training_question_from_caregiver:
            training_status = (
                f"{source} trigger: caregiver question, intent={target_intent} "
                f"window={cfg.training_clip_seconds}s"
            )
            return
        training_status = (
            f"{source} trigger: recording intent={target_intent} "
            f"warmup={cfg.asl_warmup_seconds:.1f}s "
            f"window={cfg.training_clip_seconds}s"
        )

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
                logger.info("UI action=toggle_mode current_mode=%s", "TRAIN" if training_mode else "COMM")
                training_mode = not training_mode
                mode_name = "TRAIN" if training_mode else "COMM"
                training_status = f"{mode_name} mode"
                trigger_frames = []
                review_video_idx = -1
                if training_mode and cfg.training_strict_local_llm and not llm.is_local_model_ready:
                    training_status = (
                        "TRAIN mode needs local LLM model for questions. "
                        "Check COMM_MODEL_PATH and restart."
                    )
                pending_delete_intent = ""
                pending_delete_intent_until = 0.0
            elif action == "next_intent":
                intent_idx = (intent_idx + 1) % len(intent_labels)
                logger.info("UI action=next_intent selected_intent=%s", intent_labels[intent_idx])
                training_status = f"intent={intent_labels[intent_idx]}"
            elif action == "add_intent" and training_mode and not question_capture_active:
                logger.info("UI action=add_intent mode=TRAIN")
                question_capture_active = True
                training_question_capture_mode = "intent"
                training_status = (
                    f"Speak new intent name (up to {cfg.mic_question_seconds}s)..."
                )
                try:
                    _question_capture_req_q.put_nowait(True)
                except queue.Full:
                    pass
            elif action == "capture_sample" and training_mode and not capture_active:
                logger.info("UI action=capture_sample mode=TRAIN")
                _begin_training_capture("REC")
            elif action == "capture_sample" and not training_mode and not question_capture_active:
                logger.info("UI action=capture_sample mode=COMM")
                question_capture_active = True
                training_question_capture_mode = "comm"
                training_status = (
                    f"Listening to caregiver question (up to {cfg.mic_question_seconds}s)..."
                )
                try:
                    _question_capture_req_q.put_nowait(True)
                except queue.Full:
                    pass
            elif action == "ask_question" and training_mode and not question_capture_active:
                logger.info("UI action=ask_question mode=TRAIN")
                question_capture_active = True
                training_question_capture_mode = "train"
                training_status = (
                    f"Listening for caregiver training question (up to {cfg.mic_question_seconds}s)..."
                )
                try:
                    _question_capture_req_q.put_nowait(True)
                except queue.Full:
                    pass
            elif action == "review_video" and training_mode:
                logger.info("UI action=review_video mode=TRAIN")
                videos = store.list_training_videos()
                if not videos:
                    training_status = "No saved training video to review."
                    current_response = ""
                else:
                    review_video_idx = (review_video_idx + 1) % len(videos)
                    current_response = _format_training_video_list(videos, review_video_idx)
                    current_question = str(videos[review_video_idx]["question_text"])
                    ok_play, status = _play_training_video_npz(
                        display,
                        str(videos[review_video_idx]["file_path"]),
                        current_result,
                        "TRAIN",
                        current_question,
                        current_response,
                        str(videos[review_video_idx]["intent"]),
                    )
                    training_status = status
                    if ok_play:
                        _put_display(None, current_question, current_response)
            elif action == "delete_video" and training_mode:
                logger.info("UI action=delete_video mode=TRAIN")
                videos = store.list_training_videos()
                if not videos:
                    training_status = "No saved training video to delete."
                else:
                    review_video_idx = min(review_video_idx, len(videos) - 1)
                    selected_video = videos[review_video_idx]
                    removed_path = store.delete_training_video(int(selected_video["id"]))
                    if not removed_path:
                        training_status = "Failed to delete training video metadata."
                    else:
                        removed_count = _delete_files([removed_path])
                        remaining = store.list_training_videos()
                        if remaining:
                            review_video_idx = min(review_video_idx, len(remaining) - 1)
                            current_response = _format_training_video_list(remaining, review_video_idx)
                        else:
                            review_video_idx = -1
                            current_response = ""
                        training_status = (
                            f"Deleted video {selected_video['intent']} ({removed_count} file removed)."
                        )
            elif action == "reset_training" and training_mode:
                logger.info("UI action=reset_training mode=TRAIN")
                removed_paths = store.delete_all_training_videos()
                removed_files = _delete_files(removed_paths)
                training_store.clear()
                training_trigger_classifier = AslIntentClassifier(cfg.asl_intent_model_path)
                review_video_idx = -1
                manual_training_question = ""
                training_question_from_caregiver = False
                training_question_history = []
                current_question = ""
                current_response = ""
                capture_question = ""
                training_status = (
                    f"Reset training: cleared {len(removed_paths)} videos, removed {removed_files} files."
                )
                _put_display(None, current_question, current_response)
            elif action == "fit_model" and training_mode:
                logger.info("UI action=fit_model mode=TRAIN")
                min_samples = max(
                    max(1, cfg.training_fit_min_total_samples),
                    len(intent_labels) * max(0, cfg.training_fit_min_per_intent),
                )
                before_summary = _model_update_summary(cfg.asl_intent_model_path)
                videos_before = len(store.list_training_videos())
                Xc_before, yc_before = training_store.load_samples()
                logger.info(
                    "FIT start samples=%d labels=[%s] videos=%d model_before=%s",
                    len(Xc_before),
                    format_intent_counts(yc_before),
                    videos_before,
                    before_summary,
                )
                logger.info(
                    "FIT requirement total_samples>=%d intents=%d per_intent=%d",
                    min_samples,
                    len(intent_labels),
                    max(0, cfg.training_fit_min_per_intent),
                )
                ok = training_store.train(min_samples=min_samples)
                if ok:
                    training_trigger_classifier = AslIntentClassifier(cfg.asl_intent_model_path)
                    trigger_warned_unavailable = False
                    Xc_after, yc_after = training_store.load_samples()
                    after_summary = _model_update_summary(cfg.asl_intent_model_path)
                    videos_after = len(store.list_training_videos())
                    logger.info(
                        "FIT success samples=%d labels=[%s] videos=%d model_after=%s",
                        len(Xc_after),
                        format_intent_counts(yc_after),
                        videos_after,
                        after_summary,
                    )
                    logger.info("Training classifier reloaded from %s", cfg.asl_intent_model_path)
                    training_status = (
                        f"FIT ok samples={len(Xc_after)} videos={videos_after} | {after_summary}"
                    )
                else:
                    Xc, _yc = training_store.load_samples()
                    logger.warning("Training fit rejected: samples=%d required>=%d", len(Xc), min_samples)
                    training_status = f"need >= {min_samples} samples (have {len(Xc)})"
            elif action == "delete_intent" and training_mode:
                current_intent = intent_labels[intent_idx]
                logger.info("UI action=delete_intent mode=TRAIN intent=%s", current_intent)
                now_confirm = time.monotonic()
                if (
                    pending_delete_intent != current_intent
                    or now_confirm > pending_delete_intent_until
                ):
                    pending_delete_intent = current_intent
                    pending_delete_intent_until = now_confirm + 2.0
                    training_status = (
                        f"Confirm delete intent '{current_intent}': double-tap DEL again in 2s"
                    )
                elif len(intent_labels) <= 2:
                    training_status = "Keep at least 2 intents for training."
                elif current_intent == cfg.training_question_trigger_intent:
                    training_status = "Cannot delete trigger intent; change trigger config first."
                else:
                    removed_samples = training_store.delete_intent(current_intent)
                    removed_video_paths = store.delete_training_videos_by_intent(current_intent)
                    removed_video_files = _delete_files(removed_video_paths)

                    intent_labels.pop(intent_idx)
                    intent_idx = max(0, min(intent_idx, len(intent_labels) - 1))

                    extra_intents = _current_extra_intents(base_intents, intent_labels)
                    try:
                        _save_extra_intents(cfg.training_intents_file, extra_intents)
                    except Exception as exc:
                        logger.error("Failed to persist intents after deletion: %s", exc)

                    manual_training_question = ""
                    capture_question = ""
                    training_question_history = [
                        (q, i) for (q, i) in training_question_history if i != current_intent
                    ]
                    training_trigger_classifier = AslIntentClassifier(cfg.asl_intent_model_path)

                    training_status = (
                        f"Deleted intent '{current_intent}' samples={removed_samples} "
                        f"videos={removed_video_files}"
                    )
                    current_response = f"Intents: {len(intent_labels)}"
                    _put_display(None, current_question, current_response)
                if training_status.startswith("Deleted intent"):
                    pending_delete_intent = ""
                    pending_delete_intent_until = 0.0

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
                        if capture_question:
                            training_question_history.append((capture_question, intent))

                        ok_video, video_result = _save_training_video_npz(
                            cfg.training_video_dir,
                            capture_frames,
                            intent,
                        )
                        if ok_video:
                            store.save_training_video(
                                intent=intent,
                                question_text=capture_question,
                                file_path=video_result,
                                frame_count=len(capture_frames),
                                duration_seconds=float(cfg.training_clip_seconds),
                            )
                            logger.info(
                                "Training sample saved intent=%s frames=%d clip=%ss artifact=%s",
                                intent,
                                len(capture_frames),
                                cfg.training_clip_seconds,
                                video_result,
                            )

                        Xc, yc = training_store.load_samples()
                        base_status = (
                            f"saved '{intent}' total={len(Xc)} "
                            f"[{format_intent_counts(yc)}]"
                        )
                        if ok_video:
                            training_status = f"{base_status} video saved"
                        else:
                            training_status = f"{base_status} | {video_result}"
                    else:
                        logger.warning("Training capture finished with no frames")
                        training_status = "capture failed: no frames"
                    capture_question = ""

            if question_capture_active:
                try:
                    ok, q_text, status = _question_capture_result_q.get_nowait()
                    question_capture_active = False
                    mode = training_question_capture_mode
                    training_question_capture_mode = ""
                    training_status = status
                    if ok and q_text and mode == "comm":
                        logger.info("COMM question captured text=%s", q_text)
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
                    elif ok and q_text and mode == "train":
                        logger.info("TRAIN caregiver question captured text=%s", q_text)
                        manual_training_question = q_text
                        current_question = q_text
                        current_response = "Caregiver question ready. Tap REC or use gesture."
                        training_status = "Caregiver training question saved."
                        _put_display(None, current_question, current_response)
                    elif ok and q_text and mode == "intent":
                        label = _normalize_intent_label(q_text)
                        logger.info("Intent-add speech captured raw=%s normalized=%s", q_text, label)
                        if not label:
                            training_status = "Intent add failed: say a short one-word label."
                        elif label in intent_labels:
                            intent_idx = intent_labels.index(label)
                            training_status = f"Intent already exists: {label}"
                        else:
                            intent_labels.append(label)
                            if label not in base_intents:
                                extra_intents = [x for x in intent_labels if x not in base_intents]
                                try:
                                    _save_extra_intents(cfg.training_intents_file, extra_intents)
                                except Exception as exc:
                                    logger.error("Failed to persist custom intents: %s", exc)
                                    training_status = (
                                        f"Added intent {label} (not persisted: write error)."
                                    )
                                    continue
                            intent_idx = len(intent_labels) - 1
                            training_status = f"Added intent: {label}"
                        current_response = f"Intents: {len(intent_labels)}"
                        _put_display(None, current_question, current_response)
                    elif not ok:
                        logger.warning("Question capture failed mode=%s status=%s", mode, status)
                except queue.Empty:
                    pass

            if training_mode and cfg.training_question_trigger_enabled and not capture_active:
                if raw_frame is not None:
                    trigger_frames.append(raw_frame.copy())
                max_frames = max(8, cfg.training_trigger_window_frames)
                if len(trigger_frames) > max_frames:
                    trigger_frames = trigger_frames[-max_frames:]

                now = time.monotonic()
                if len(trigger_frames) >= max_frames and now >= trigger_next_eval:
                    trigger_next_eval = now + max(0.2, cfg.training_trigger_eval_interval_seconds)

                    if not training_trigger_classifier.ready:
                        if not trigger_warned_unavailable:
                            training_status = "Gesture trigger unavailable: train model then tap FIT."
                            trigger_warned_unavailable = True
                    else:
                        trigger_warned_unavailable = False
                        pred = training_trigger_classifier.predict(
                            extract_window_feature(trigger_frames)
                        )
                        if (
                            pred.intent == cfg.training_question_trigger_intent
                            and pred.confidence >= cfg.training_question_trigger_confidence
                            and now - trigger_last_fire >= cfg.training_question_trigger_cooldown_seconds
                        ):
                            logger.info(
                                "Gesture trigger fired intent=%s confidence=%.3f threshold=%.3f",
                                pred.intent,
                                pred.confidence,
                                cfg.training_question_trigger_confidence,
                            )
                            trigger_last_fire = now
                            _begin_training_capture("GESTURE")
            elif not training_mode:
                trigger_frames = []
                trigger_warned_unavailable = False

            if (
                pending_delete_intent
                and time.monotonic() > pending_delete_intent_until
            ):
                pending_delete_intent = ""
                pending_delete_intent_until = 0.0

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
                logger.info("Quit requested from display (%s) — shutting down", display.quit_reason or "unknown")
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
