from __future__ import annotations

import logging
import os
import time
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
        self._font_md: Optional[object] = None
        self._font_sm: Optional[object] = None
        self._should_quit = False
        self._quit_reason = ""
        self._last_action = ""
        self._started_at = time.monotonic()
        self._close_guard_seconds = 3.0
        self._last_intent_tap_at = 0.0
        self._intent_double_tap_seconds = 0.6
        # Close button — top-right corner, finger-friendly 48×48 px
        self._close_rect: Optional[object] = None
        self._mode_rect: Optional[object] = None
        self._intent_rect: Optional[object] = None
        self._ask_rect: Optional[object] = None
        self._review_rect: Optional[object] = None
        self._delete_rect: Optional[object] = None
        self._reset_rect: Optional[object] = None
        self._rec_rect: Optional[object] = None
        self._fit_rect: Optional[object] = None

        if _HAS_PYGAME:
            os.environ["SDL_FBDEV"] = fbdev
            self._init_pygame()

    @property
    def should_quit(self) -> bool:
        return self._should_quit

    @property
    def quit_reason(self) -> str:
        return self._quit_reason

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
        self._font_md = pygame.font.SysFont("monospace", 20, bold=True)
        self._font_sm = pygame.font.SysFont("monospace", 16)
        self._close_rect = pygame.Rect(self.width - 48, 0, 48, 48)
        # Large touch targets for 3.5" screen
        self._mode_rect = pygame.Rect(4, 4, 60, 40)
        self._intent_rect = pygame.Rect(68, 4, 90, 40)
        self._ask_rect = pygame.Rect(162, 4, 40, 40)
        self._review_rect = pygame.Rect(206, 4, 40, 40)
        self._delete_rect = pygame.Rect(250, 4, 40, 40)
        self._reset_rect = pygame.Rect(294, 4, 40, 40)
        self._rec_rect = pygame.Rect(338, 4, 40, 40)
        self._fit_rect = pygame.Rect(382, 4, 40, 40)
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
                self._quit_reason = "pygame quit event"
                logger.info("Display requested shutdown: %s", self._quit_reason)
                self._should_quit = True
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._quit_reason = f"keyboard key={event.key}"
                    logger.info("Display requested shutdown: %s", self._quit_reason)
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
                    uptime = time.monotonic() - self._started_at
                    if uptime < self._close_guard_seconds:
                        logger.warning(
                            "Ignoring close tap during startup guard: pos=%s uptime=%.2fs",
                            pos,
                            uptime,
                        )
                    else:
                        self._quit_reason = f"close button tap pos={pos} uptime={uptime:.2f}s"
                        logger.info("Display requested shutdown: %s", self._quit_reason)
                        self._should_quit = True
                        return
                if self._mode_rect and self._mode_rect.collidepoint(pos):
                    self._last_action = "toggle_mode"
                if self._intent_rect and self._intent_rect.collidepoint(pos):
                    now = time.monotonic()
                    if now - self._last_intent_tap_at <= self._intent_double_tap_seconds:
                        self._last_action = "add_intent"
                        self._last_intent_tap_at = 0.0
                    else:
                        self._last_action = "next_intent"
                        self._last_intent_tap_at = now
                if self._ask_rect and self._ask_rect.collidepoint(pos):
                    self._last_action = "ask_question"
                if self._review_rect and self._review_rect.collidepoint(pos):
                    self._last_action = "review_video"
                if self._delete_rect and self._delete_rect.collidepoint(pos):
                    self._last_action = "delete_video"
                if self._reset_rect and self._reset_rect.collidepoint(pos):
                    self._last_action = "reset_training"
                if self._rec_rect and self._rec_rect.collidepoint(pos):
                    self._last_action = "capture_sample"
                if self._fit_rect and self._fit_rect.collidepoint(pos):
                    self._last_action = "fit_model"
                if self._last_action:
                    logger.info(
                        "Display action=%s event=%s pos=%s",
                        self._last_action,
                        pygame.event.event_name(event.type),
                        pos,
                    )

        self._screen.fill(_DARK)
        # Slightly smaller camera panel to make controls/text clearer on 3.5" screen.
        cam_w = int(self.width * 0.42)
        right_x = cam_w + 6
        right_w = self.width - right_x - 4

        # Left: camera feed (rotated 90° clockwise for sideways-mounted camera)
        if frame is not None:
            try:
                cam_h = self.height - 94
                crop = frame
                surf = pygame.surfarray.make_surface(
                    np.ascontiguousarray(crop.swapaxes(0, 1))
                )
                # rotate() is CCW; -90 = 90° clockwise
                surf = pygame.transform.rotate(surf, -90)
                surf = pygame.transform.scale(surf, (cam_w, cam_h))
                self._screen.blit(surf, (0, 50))
            except Exception:
                pass

        # Right: YES / NO badge
        color = _LABEL_COLORS[result.label]
        badge = self._font_xl.render(result.label.value.upper(), True, color)
        bx = right_x + (right_w - badge.get_width()) // 2
        by = 58
        self._screen.blit(badge, (bx, by))

        # Confidence bar
        bar_w = max(1, int(result.confidence * (right_w - 20)))
        pygame.draw.rect(
            self._screen, color,
            pygame.Rect(right_x + 10, by + badge.get_height() + 6, min(bar_w, right_w - 20), 10),
        )

        # Bottom: current question (spoken prompt)
        if question:
            for i, line in enumerate(self._wrap(f"Q: {question}", 54)[:2]):
                txt = self._font_sm.render(line, True, _WHITE)
                self._screen.blit(txt, (4, self.height - 34 + i * 17))

        # Right panel: final response text
        if response:
            lines = self._wrap(response, 22)[:5]
            for i, line in enumerate(lines):
                txt = self._font_sm.render(line, True, _WHITE)
                self._screen.blit(txt, (right_x + 10, by + badge.get_height() + 24 + i * 17))

        # Frame counter
        fc = self._font_sm.render(f"#{frame_id}", True, (60, 60, 60))
        self._screen.blit(fc, (self.width - fc.get_width() - 52, 4))

        # Mode and training controls
        if self._mode_rect:
            pygame.draw.rect(self._screen, (40, 80, 160), self._mode_rect, border_radius=5)
            mode_txt = self._font_md.render("MODE", True, _WHITE)
            mode_rect = mode_txt.get_rect(center=(self._mode_rect.centerx, self._mode_rect.y + 14))
            self._screen.blit(mode_txt, mode_rect)
            mode_val = self._font_sm.render(mode, True, _WHITE)
            mode_val_rect = mode_val.get_rect(center=(self._mode_rect.centerx, self._mode_rect.y + 30))
            self._screen.blit(mode_val, mode_val_rect)

        if self._intent_rect:
            pygame.draw.rect(self._screen, (90, 90, 90), self._intent_rect, border_radius=5)
            itxt = self._font_md.render("INT", True, _WHITE)
            itxt_rect = itxt.get_rect(center=(self._intent_rect.centerx, self._intent_rect.y + 14))
            self._screen.blit(itxt, itxt_rect)
            ival = self._font_sm.render(training_intent[:9], True, _WHITE)
            ival_rect = ival.get_rect(center=(self._intent_rect.centerx, self._intent_rect.y + 30))
            self._screen.blit(ival, ival_rect)

        if self._ask_rect:
            pygame.draw.rect(self._screen, (110, 80, 30), self._ask_rect, border_radius=5)
            ask_txt = self._font_md.render("ASK", True, _WHITE)
            ask_rect = ask_txt.get_rect(center=self._ask_rect.center)
            self._screen.blit(ask_txt, ask_rect)

        if self._review_rect:
            pygame.draw.rect(self._screen, (70, 90, 140), self._review_rect, border_radius=5)
            rev_txt = self._font_md.render("REV", True, _WHITE)
            rev_rect = rev_txt.get_rect(center=self._review_rect.center)
            self._screen.blit(rev_txt, rev_rect)

        if self._delete_rect:
            pygame.draw.rect(self._screen, (155, 55, 55), self._delete_rect, border_radius=5)
            del_txt = self._font_md.render("DEL", True, _WHITE)
            del_rect = del_txt.get_rect(center=self._delete_rect.center)
            self._screen.blit(del_txt, del_rect)

        if self._reset_rect:
            pygame.draw.rect(self._screen, (120, 30, 120), self._reset_rect, border_radius=5)
            rst_txt = self._font_md.render("RST", True, _WHITE)
            rst_rect = rst_txt.get_rect(center=self._reset_rect.center)
            self._screen.blit(rst_txt, rst_rect)

        if self._rec_rect:
            pygame.draw.rect(self._screen, (150, 70, 30), self._rec_rect, border_radius=5)
            rec_txt = self._font_md.render("REC", True, _WHITE)
            rec_rect = rec_txt.get_rect(center=self._rec_rect.center)
            self._screen.blit(rec_txt, rec_rect)

        if self._fit_rect:
            pygame.draw.rect(self._screen, (40, 130, 70), self._fit_rect, border_radius=5)
            fit_txt = self._font_md.render("FIT", True, _WHITE)
            fit_rect = fit_txt.get_rect(center=self._fit_rect.center)
            self._screen.blit(fit_txt, fit_rect)

        if training_status:
            stxt = self._font_sm.render(training_status[:54], True, _WHITE)
            self._screen.blit(stxt, (4, self.height - 20))

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
