from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameData:
    frame_id: int
    face_bbox: Optional[tuple[int, int, int, int]]


class CameraService:
    def __init__(self) -> None:
        self._running = False
        self._frame_id = 0

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> Optional[FrameData]:
        if not self._running:
            return None
        self._frame_id += 1
        # Placeholder synthetic bbox until IMX500 metadata integration is implemented.
        return FrameData(frame_id=self._frame_id, face_bbox=(120, 60, 180, 180))
