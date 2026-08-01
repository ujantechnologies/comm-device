"""Runtime loop: mic question -> ASL intent -> spoken response.

This is a practical first version for communication support where users can be
slow to respond.

Steps per turn:
1) Record question from microphone and transcribe with Whisper.
2) Wait for and capture an extended ASL response window from camera.
3) Classify intent from personalized model.
4) Convert intent into spoken response with LLM + TTS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comm_device.asl_intent import (  # noqa: E402
    AslIntentClassifier,
    capture_response_window,
    record_mic_audio,
    transcribe_audio,
)
from comm_device.audio_router import AudioRouter  # noqa: E402
from comm_device.camera import CameraService  # noqa: E402
from comm_device.config import load_config  # noqa: E402
from comm_device.llm import LlmService  # noqa: E402
from comm_device.tts import TtsService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mic->ASL->intent->speech communication loop.")
    parser.add_argument("--mic-seconds", type=int, default=6)
    parser.add_argument("--response-seconds", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--language", default="en")
    parser.add_argument("--once", action="store_true", help="Run one turn then exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    llm = LlmService(cfg.model_path)
    tts = TtsService(cfg.voice_model_path)
    audio = AudioRouter(output_target=cfg.audio_output_target)
    classifier = AslIntentClassifier(cfg.asl_intent_model_path)
    camera = CameraService(imx500_config_path=cfg.imx500_config_path)

    if not classifier.ready:
        print(
            "ASL intent model not found. Run training first:\n"
            "  python scripts/asl_training_mode.py --intents yes,no,water,pain,rest"
        )
        return 1

    camera.start()
    turn = 0

    try:
        while True:
            turn += 1
            question_wav = f"/tmp/question_{turn}.wav"
            ok = record_mic_audio(
                output_path=question_wav,
                seconds=args.mic_seconds,
                target=cfg.audio_input_target,
            )
            if not ok:
                print("No microphone audio captured. Check Bluetooth mic target.")
                if args.once:
                    return 1
                continue

            question_text = transcribe_audio(
                question_wav,
                model_name=args.whisper_model,
                language=args.language,
            )
            if not question_text:
                question_text = "I could not understand the question clearly."
            print(f"Question: {question_text}")

            print("User can now respond with ASL...")
            feature = capture_response_window(
                camera,
                seconds=args.response_seconds,
                warmup_seconds=args.warmup_seconds,
            )
            pred = classifier.predict(feature)
            intent = pred.intent
            print(f"Predicted intent: {intent} (confidence={pred.confidence:.2f})")

            response = llm.generate_intent_response(question_text, intent)
            print(f"Response: {response}")

            out_path = tts.synthesize(response)
            audio.play(out_path)

            if args.once:
                return 0
    finally:
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
