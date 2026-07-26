from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .expression import ExpressionLabel, ExpressionResult

logger = logging.getLogger(__name__)

# Target the MHS-3.5" SPI framebuffer; override via environment variables.
os.environ.setdefault("SDL_VIDEODRIVER", "fbdev")
os.environ.setdefault("SDL_FBDEV", "/dev/fb1")
os.environ.setdefault("SDL_MOUSEDRV", "TSLIB")
os.environ.setdefault("SDL_MOUSE_RELATIVE", "0")

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False
    logger.warning("pygame not available — display will be text-only")

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 220, 80)
_RED = (220, 50, 50)
_GREY = (180, 180, 180)
_DARK = (18, 18, 18)

_LABEL_COLORS: dict[ExpressionLabel, tuple[int, int, int]] = {
    ExpressionLabel.YES: _GREEN,
    ExpressionLabel.NO: _RED,
    ExpressionLabel.UNCERTAIN: _GREY,
}


class DisplayService:
    """Renders to a pygame surface on /dev/fb1 (MHS-3.5" 480×320).

    Layout (landscape 480×320):
      Left half  — live camera frame (scaled)
      Right half — large YES/NO label + confidence bar
      Bottom strip — LLM response text
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._screen: Optional[object] = None
        self._font_xl: Optional[object] = None
        self._font_sm: Optional[object] = None

        if _HAS_PYGAME:
            self._init_pygame()

    def _init_pygame(self) -> None:
        pygame.init()
        try:
            self._screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.FULLSCREEN | pygame.NOFRAME,
            )
        except pygame.error:
            # Dev machine without a framebuffer — open a regular window instead
            os.environ["SDL_VIDEODRIVER"] = ""
            pygame.display.quit()
            pygame.display.init()
            self._screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Comm Device")
        pygame.mouse.set_visible(False)
        self._font_xl = pygame.font.SysFont("monospace", 90, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 16)
        logger.info("Display initialised at %dx%d", self.width, self.height)

    def render(
        self,
        frame_id: int,
        result: ExpressionResult,
        response: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        if not _HAS_PYGAME or self._screen is None:
            print(
                f"frame={frame_id} label={result.label.value} "
                f"conf={result.confidence:.2f} | {response}"
            )
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        self._screen.fill(_DARK)
        half = self.width // 2

        # Left: camera feed
        if frame is not None:
            try:
                cam_h = self.height - 32
                crop = frame[:cam_h, :half]
                surf = pygame.surfarray.make_surface(
                    np.ascontiguousarray(crop.swapaxes(0, 1))
                )
                self._screen.blit(surf, (0, 0))
            except Exception:
                pass

        # Right: YES / NO badge
        color = _LABEL_COLORS[result.label]
        badge = self._font_xl.render(result.label.value.upper(), True, color)
        bx = half + (half - badge.get_width()) // 2
        by = (self.height // 2 - badge.get_height()) // 2
        self._screen.blit(badge, (bx, by))

        # Confidence bar
        bar_w = max(1, int(result.confidence * (half - 20)))
        pygame.draw.rect(
            self._screen, color,
            pygame.Rect(half + 10, by + badge.get_height() + 6, bar_w, 8),
        )

        # Bottom: response text (up to 2 lines)
        if response:
            for i, line in enumerate(self._wrap(response, 54)[:2]):
                txt = self._font_sm.render(line, True, _WHITE)
                self._screen.blit(txt, (4, self.height - 34 + i * 17))

        # Frame counter
        fc = self._font_sm.render(f"#{frame_id}", True, (60, 60, 60))
        self._screen.blit(fc, (self.width - fc.get_width() - 4, 4))

        pygame.display.flip()

    def quit(self) -> None:
        if _HAS_PYGAME:
            pygame.quit()

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        words, lines, line = text.split(), [], []
        for word in words:
            if sum(len(w) + 1 for w in line) + len(word) <= width:
                line.append(word)
            else:
                if line:
                    lines.append(" ".join(line))
                line = [word]
        if line:
            lines.append(" ".join(line))
        return lines
