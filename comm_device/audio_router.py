import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class AudioRouter:
    def detect_bluetooth_sink(self) -> Optional[str]:
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
            try:
                subprocess.run(["paplay", "--device", sink, file_path], check=False)
                return
            except FileNotFoundError:
                pass

        # Pi 5 uses PipeWire — try paplay on the default sink before aplay
        try:
            result = subprocess.run(
                ["paplay", file_path],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                return
            logger.debug("paplay failed (rc=%d): %s", result.returncode,
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
