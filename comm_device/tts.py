from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class TtsService:
    """Synthesises speech via the Piper TTS binary.

    When the piper binary or voice model is absent the text is written to a
    .txt file so the audio_router can log it instead of crashing.
    """

    def __init__(self, voice_model_path: str) -> None:
        self.voice_model_path = voice_model_path
        self._piper = shutil.which("piper") or shutil.which("piper-tts")
        if self._piper is None:
            logger.warning("piper binary not found — TTS will write text placeholder")

    def synthesize(
        self, text: str, output_path: str = "artifacts/response.wav"
    ) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if self._piper and Path(self.voice_model_path).exists():
            try:
                subprocess.run(
                    [self._piper, "--model", self.voice_model_path, "--output_file", str(out)],
                    input=text.encode("utf-8"),
                    check=True,
                    capture_output=True,
                )
                return str(out)
            except subprocess.CalledProcessError as exc:
                logger.error("Piper failed: %s", exc.stderr.decode(errors="replace"))

        # Fallback: write text file so the pipeline doesn't stall
        txt = out.with_suffix(".txt")
        txt.write_text(text + "\n", encoding="utf-8")
        return str(txt)
