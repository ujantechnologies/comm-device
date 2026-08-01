"""Record short audio from PipeWire, optionally play back, and transcribe with Whisper.

Examples:
  python scripts/transcribe_mic.py --seconds 6 --model tiny
  python scripts/transcribe_mic.py --target bluez_input.CF:57:28:DC:04:87 --playback
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _record_wav(output: Path, seconds: int, target: str) -> None:
    cmd = ["pw-record"]
    if target:
        cmd.extend(["--target", target])
    cmd.append(str(output))

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


def _play_wav(path: Path) -> None:
    subprocess.run(["pw-play", str(path)], check=False)


def _transcribe(path: Path, model_name: str, language: str) -> str:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed in this environment. "
            "Run: pip install openai-whisper"
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(path),
        fp16=False,  # CPU on Pi uses FP32
        language=language or None,
    )
    return str(result.get("text", "")).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record and transcribe microphone input.")
    parser.add_argument("--seconds", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument(
        "--target",
        default="",
        help="PipeWire source target name (e.g. bluez_input.CF:57:28:DC:04:87)",
    )
    parser.add_argument("--model", default="tiny", help="Whisper model: tiny|base|small|...")
    parser.add_argument("--language", default="en", help="Language code, e.g. en")
    parser.add_argument("--playback", action="store_true", help="Play back recorded audio")
    parser.add_argument(
        "--output",
        default="/tmp/mic-test.wav",
        help="Output WAV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {args.seconds}s to {out}...")
    if args.target:
        print(f"Using target source: {args.target}")
    _record_wav(out, args.seconds, args.target)

    if not out.exists() or out.stat().st_size == 0:
        print("Recording failed: no audio file created.")
        return 1

    print(f"Recorded {out.stat().st_size} bytes")

    if args.playback:
        print("Playing back recording...")
        _play_wav(out)

    print(f"Transcribing with Whisper model '{args.model}'...")
    try:
        text = _transcribe(out, args.model, args.language)
    except Exception as exc:
        print(f"Transcription failed: {exc}")
        return 1

    print("Transcription:")
    print(text if text else "<empty>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
