from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False
    logger.warning("picamera2 not available — using synthetic frame source")

_FRAME_W, _FRAME_H = 640, 480


@dataclass
class FrameData:
    frame_id: int
    frame: np.ndarray              # HxWx3 RGB uint8
    face_bbox: Optional[tuple[int, int, int, int]] = None  # x, y, w, h pixels


class CameraService:
    """Wraps picamera2 with IMX500 face-detection metadata.

    When picamera2 is unavailable (dev machines) it returns synthetic frames
    so the rest of the pipeline can be exercised offline.
    """

    def __init__(self, imx500_config_path: str = "") -> None:
        self._frame_id = 0
        self._running = False
        self._cam: Optional[object] = None
        self._imx500: Optional[object] = None

        if _HAS_PICAMERA2:
            self._init_picamera2(imx500_config_path)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_picamera2(self, imx500_config_path: str) -> None:
        if imx500_config_path:
            self._imx500 = IMX500(imx500_config_path)
            intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
            intrinsics.task = "object detection"
            self._cam = Picamera2(self._imx500.camera_num)
        else:
            self._cam = Picamera2()

        cfg = self._cam.create_preview_configuration(
            main={"size": (_FRAME_W, _FRAME_H), "format": "RGB888"},
            buffer_count=4,
        )
        self._cam.configure(cfg)
        logger.info("Camera initialised (%s)", "IMX500" if imx500_config_path else "standard")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        if _HAS_PICAMERA2 and self._cam:
            self._cam.start()

    def stop(self) -> None:
        self._running = False
        if _HAS_PICAMERA2 and self._cam:
            self._cam.stop()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read(self) -> Optional[FrameData]:
        if not self._running:
            return None
        self._frame_id += 1
        if _HAS_PICAMERA2 and self._cam:
            return self._read_real()
        return self._read_synthetic()

    def _read_real(self) -> FrameData:
        request = self._cam.capture_request()
        try:
            frame: np.ndarray = request.make_array("main")
            bbox = self._parse_imx500_bbox(request)
        finally:
            request.release()
        return FrameData(frame_id=self._frame_id, frame=frame, face_bbox=bbox)

    def _parse_imx500_bbox(
        self, request: object
    ) -> Optional[tuple[int, int, int, int]]:
        if self._imx500 is None:
            return None
        try:
            outputs = self._imx500.get_outputs(request.get_metadata())
            if not outputs or len(outputs[0]) == 0:
                return None
            b = outputs[0][0]  # first detection: [y1, x1, y2, x2] normalised
            x = int(b[1] * _FRAME_W)
            y = int(b[0] * _FRAME_H)
            w = int((b[3] - b[1]) * _FRAME_W)
            h = int((b[2] - b[0]) * _FRAME_H)
            return (x, y, w, h)
        except Exception:
            return None

    def _read_synthetic(self) -> FrameData:
        frame = np.zeros((_FRAME_H, _FRAME_W, 3), dtype=np.uint8)
        cx = _FRAME_W // 2 + int(60 * np.sin(self._frame_id * 0.05))
        frame[80:400, cx - 100:cx + 100] = [200, 160, 130]
        return FrameData(
            frame_id=self._frame_id,
            frame=frame,
            face_bbox=(cx - 100, 80, 200, 320),
        )
