from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .expression import ExpressionLabel, ExpressionResult

logger = logging.getLogger(__name__)

# Auto-detect display backend.
# If a Wayland or X11 session is active (e.g. Pi desktop), use it so the app
# opens a window on screen. Fall back to fbdev for the SPI panel on headless setups.
if "WAYLAND_DISPLAY" in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
elif "DISPLAY" in os.environ:
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
else:
    os.environ.setdefault("SDL_VIDEODRIVER", "fbdev")
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

    def __init__(self, width: int, height: int, fbdev: str = "/dev/fb0") -> None:
        self.width = width
        self.height = height
        self._screen: Optional[object] = None
        self._font_xl: Optional[object] = None
        self._font_sm: Optional[object] = None
        self._should_quit = False
        self._last_action = ""
        # Close button — top-right corner, finger-friendly 48×48 px
        self._close_rect: Optional[object] = None
        self._mode_rect: Optional[object] = None
        self._intent_rect: Optional[object] = None
        self._rec_rect: Optional[object] = None
        self._fit_rect: Optional[object] = None

        if _HAS_PYGAME:
            os.environ["SDL_FBDEV"] = fbdev
            self._init_pygame()

    @property
    def should_quit(self) -> bool:
        return self._should_quit

    def consume_action(self) -> str:
        action = self._last_action
        self._last_action = ""
        return action

    def _init_pygame(self) -> None:
        pygame.init()
        try:
            self._screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.FULLSCREEN | pygame.NOFRAME,
            )
        except pygame.error:
            # No framebuffer — try a regular windowed display (e.g. X11/Wayland)
            try:
                os.environ["SDL_VIDEODRIVER"] = ""
                pygame.display.quit()
                pygame.display.init()
                self._screen = pygame.display.set_mode((self.width, self.height))
            except pygame.error:
                # Headless environment — run without any visual output
                logger.warning(
                    "No display available (fbdev and windowed both failed) "
                    "— running in text-only mode"
                )
                pygame.quit()
                return

        pygame.display.set_caption("Comm Device")
        pygame.mouse.set_visible(True)  # show cursor so touch targets are visible
        self._font_xl = pygame.font.SysFont("monospace", 90, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 16)
        self._close_rect = pygame.Rect(self.width - 48, 0, 48, 48)
        self._mode_rect = pygame.Rect(4, 4, 76, 28)
        self._intent_rect = pygame.Rect(86, 4, 88, 28)
        self._rec_rect = pygame.Rect(self.width - 172, self.height - 32, 80, 28)
        self._fit_rect = pygame.Rect(self.width - 86, self.height - 32, 80, 28)
        logger.info("Display initialised at %dx%d", self.width, self.height)

    def render(
        self,
        frame_id: int,
        result: ExpressionResult,
        mode: str,
        question: str,
        response: str,
        training_intent: str = "",
        training_status: str = "",
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
                self._should_quit = True
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._should_quit = True
                    return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                pos = (
                    event.x if event.type == pygame.FINGERDOWN else event.pos[0],
                    event.y if event.type == pygame.FINGERDOWN else event.pos[1],
                )
                if event.type == pygame.FINGERDOWN:
                    # FINGERDOWN x/y are 0-1 normalised; convert to pixels
                    pos = (int(event.x * self.width), int(event.y * self.height))
                if self._close_rect and self._close_rect.collidepoint(pos):
                    self._should_quit = True
                    return
                if self._mode_rect and self._mode_rect.collidepoint(pos):
                    self._last_action = "toggle_mode"
                if self._intent_rect and self._intent_rect.collidepoint(pos):
                    self._last_action = "next_intent"
                if self._rec_rect and self._rec_rect.collidepoint(pos):
                    self._last_action = "capture_sample"
                if self._fit_rect and self._fit_rect.collidepoint(pos):
                    self._last_action = "fit_model"

        self._screen.fill(_DARK)
        half = self.width // 2

        # Left: camera feed (rotated 90° clockwise for sideways-mounted camera)
        if frame is not None:
            try:
                cam_h = self.height - 32
                crop = frame
                surf = pygame.surfarray.make_surface(
                    np.ascontiguousarray(crop.swapaxes(0, 1))
                )
                # rotate() is CCW; -90 = 90° clockwise
                surf = pygame.transform.rotate(surf, -90)
                surf = pygame.transform.scale(surf, (half, cam_h))
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

        # Bottom: current question (spoken prompt)
        if question:
            for i, line in enumerate(self._wrap(f"Q: {question}", 54)[:2]):
                txt = self._font_sm.render(line, True, _WHITE)
                self._screen.blit(txt, (4, self.height - 34 + i * 17))

        # Right panel: final response text
        if response:
            lines = self._wrap(response, 22)[:4]
            for i, line in enumerate(lines):
                txt = self._font_sm.render(line, True, _WHITE)
                self._screen.blit(txt, (half + 10, self.height // 2 + 12 + i * 17))

        # Frame counter
        fc = self._font_sm.render(f"#{frame_id}", True, (60, 60, 60))
        self._screen.blit(fc, (self.width - fc.get_width() - 52, 4))

        # Mode and training controls
        if self._mode_rect:
            pygame.draw.rect(self._screen, (40, 80, 160), self._mode_rect, border_radius=5)
            mode_txt = self._font_sm.render(f"MODE:{mode}", True, _WHITE)
            self._screen.blit(mode_txt, (self._mode_rect.x + 4, self._mode_rect.y + 6))

        if self._intent_rect:
            pygame.draw.rect(self._screen, (90, 90, 90), self._intent_rect, border_radius=5)
            itxt = self._font_sm.render(f"INT:{training_intent[:6]}", True, _WHITE)
            self._screen.blit(itxt, (self._intent_rect.x + 4, self._intent_rect.y + 6))

        if self._rec_rect:
            pygame.draw.rect(self._screen, (150, 70, 30), self._rec_rect, border_radius=5)
            rec_txt = self._font_sm.render("REC", True, _WHITE)
            self._screen.blit(rec_txt, (self._rec_rect.x + 24, self._rec_rect.y + 6))

        if self._fit_rect:
            pygame.draw.rect(self._screen, (40, 130, 70), self._fit_rect, border_radius=5)
            fit_txt = self._font_sm.render("FIT", True, _WHITE)
            self._screen.blit(fit_txt, (self._fit_rect.x + 24, self._fit_rect.y + 6))

        if training_status:
            stxt = self._font_sm.render(training_status[:54], True, _WHITE)
            self._screen.blit(stxt, (4, self.height - 52))

        # Close button — always on top
        if self._close_rect:
            pygame.draw.rect(self._screen, (180, 30, 30), self._close_rect, border_radius=6)
            x_surf = self._font_sm.render("X", True, _WHITE)
            xr = x_surf.get_rect(center=self._close_rect.center)
            self._screen.blit(x_surf, xr)

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
