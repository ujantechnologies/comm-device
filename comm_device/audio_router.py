import subprocess
from typing import Optional


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
        sink = self.detect_bluetooth_sink()
        if sink:
            try:
                subprocess.run(["paplay", "--device", sink, file_path], check=False)
                return
            except FileNotFoundError:
                pass

        try:
            subprocess.run(["aplay", file_path], check=False)
        except FileNotFoundError:
            print(f"audio playback command not found; generated artifact at {file_path}")
