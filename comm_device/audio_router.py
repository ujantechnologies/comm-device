import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(self, output_target: str = "") -> None:
        # Optional explicit output target (e.g. bluez_output....)
        self._output_target = output_target.strip()

    def detect_bluetooth_sink(self) -> Optional[str]:
        # If explicitly configured, trust the configured target first.
        if self._output_target:
            return self._output_target

        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None

        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and "bluez" in parts[1]:
                return parts[1]
        return None

    def play(self, file_path: str) -> None:
        # Prefer Bluetooth sink if connected
        sink = self.detect_bluetooth_sink()
        if sink:
            # PipeWire native target (works even without paplay/pactl)
            try:
                result = subprocess.run(
                    ["pw-play", "--target", sink, file_path],
                    check=False,
                    capture_output=True,
                )
                if result.returncode == 0:
                    return
                logger.debug(
                    "pw-play --target failed (rc=%d): %s",
                    result.returncode,
                    result.stderr.decode(errors="replace").strip(),
                )
            except FileNotFoundError:
                pass

            try:
                subprocess.run(["paplay", "--device", sink, file_path], check=False)
                return
            except FileNotFoundError:
                pass

        # Pi 5 / Trixie: PipeWire native (pw-play) first, then PulseAudio compat (paplay)
        for cmd in (["pw-play", file_path], ["paplay", file_path]):
            try:
                result = subprocess.run(cmd, check=False, capture_output=True)
                if result.returncode == 0:
                    return
                logger.debug("%s failed (rc=%d): %s", cmd[0], result.returncode,
                             result.stderr.decode(errors="replace").strip())
            except FileNotFoundError:
                pass

        # ALSA fallback (Pi 4 and earlier, or USB audio)
        try:
            result = subprocess.run(["aplay", file_path], check=False, capture_output=True)
            if result.returncode == 0:
                return
            logger.debug("aplay failed (rc=%d): %s", result.returncode,
                         result.stderr.decode(errors="replace").strip())
        except FileNotFoundError:
            pass

        logger.warning(
            "Audio playback failed — no working output device found. "
            "Connect a Bluetooth speaker (Step 8) or a USB audio adapter. "
            "Generated artifact: %s", file_path
        )
