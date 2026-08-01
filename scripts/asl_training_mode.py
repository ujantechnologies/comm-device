"""Interactive ASL intent training mode.

Workflow:
1) Caregiver/user says prompt or sets context.
2) User responds with ASL gesture/motion slowly.
3) Script captures response window and stores a labeled sample.
4) After collecting enough samples per intent, train classifier.

Example:
  python scripts/asl_training_mode.py --intents yes,no,water,pain,rest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comm_device.asl_intent import (  # noqa: E402
    AslIntentStore,
    capture_response_window,
    format_intent_counts,
    parse_intents,
)
from comm_device.camera import CameraService  # noqa: E402
from comm_device.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train personalized ASL intent classifier.")
    parser.add_argument("--intents", required=True, help="Comma-separated intent labels")
    parser.add_argument("--samples-per-intent", type=int, default=15)
    parser.add_argument("--response-seconds", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    intents = parse_intents(args.intents)
    if len(intents) < 2:
        print("Provide at least 2 intents with --intents")
        return 1

    cfg = load_config()
    store = AslIntentStore(
        dataset_path=cfg.asl_intent_dataset_path,
        model_path=cfg.asl_intent_model_path,
    )

    camera = CameraService(imx500_config_path=cfg.imx500_config_path)
    camera.start()

    try:
        print("Training mode started.")
        print(f"Intents: {', '.join(intents)}")
        print(
            f"Collecting {args.samples_per_intent} samples per intent; "
            f"response window {args.response_seconds}s"
        )

        for intent in intents:
            print(f"\n=== Intent: {intent} ===")
            for i in range(args.samples_per_intent):
                input(
                    f"Sample {i + 1}/{args.samples_per_intent} for '{intent}'. "
                    "Press Enter when user is ready to sign..."
                )
                feature = capture_response_window(
                    camera,
                    seconds=args.response_seconds,
                    fps_limit=args.fps,
                    warmup_seconds=args.warmup_seconds,
                )
                store.append_sample(intent, feature)
                print(f"Saved sample {i + 1} for '{intent}'")

        X, y = store.load_samples()
        print(f"\nCollected {len(X)} total samples")
        print(f"Label distribution: {format_intent_counts(y)}")

        trained = store.train(min_samples=max(20, len(intents) * 6))
        if not trained:
            print("Training did not complete. Collect more balanced samples.")
            return 1

        print(f"Model saved to: {cfg.asl_intent_model_path}")
        return 0
    finally:
        camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
