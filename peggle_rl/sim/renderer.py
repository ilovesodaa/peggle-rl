"""
Peggle Pygame renderer.

Renders the game state to a Pygame surface or to a numpy array for
headless RL training. Supports both human-visible window and RGB array mode.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from peggle_rl.sim.engine import PeggleGame

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from peggle_rl.levels.parser import PegType, BrickType
from peggle_rl.sim.physics import (
    BOARD_WIDTH, BOARD_HEIGHT, LAUNCHER_X, LAUNCHER_Y,
    ANGLE_MIN, ANGLE_MAX, brick_endpoints,
)


# ---------------------------------------------------------------------------
# Colors (matching Peggle Deluxe aesthetic)
# ---------------------------------------------------------------------------

BG_COLOR = (15, 15, 40)            # Dark blue background
WALL_COLOR = (60, 60, 80)

PEG_COLORS = {
    PegType.NORMAL: (50, 120, 220),     # Blue
    PegType.GOAL: (255, 140, 0),        # Orange
    PegType.POWERUP: (0, 200, 80),      # Green
    PegType.SCORE: (180, 50, 220),      # Purple
}

PEG_LIT_COLORS = {
    PegType.NORMAL: (100, 180, 255),
    PegType.GOAL: (255, 200, 50),
    PegType.POWERUP: (50, 255, 130),
    PegType.SCORE: (230, 120, 255),
}

BRICK_COLORS = PEG_COLORS.copy()
BRICK_LIT_COLORS = PEG_LIT_COLORS.copy()

BALL_COLOR = (230, 230, 230)
BALL_OUTLINE = (255, 255, 255)
BUCKET_COLOR = (180, 180, 200)
ROD_COLOR = (100, 100, 120)
LAUNCHER_COLOR = (200, 200, 200)

# HUD
HUD_BG = (0, 0, 0, 160)
HUD_TEXT = (255, 255, 255)
ORANGE_TEXT = (255, 160, 0)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class PeggleRenderer:
    """
    Renders PeggleGame state to a Pygame surface.

    Usage:
        renderer = PeggleRenderer(game)
        renderer.init()
        ...
        renderer.render()  # draws to window
        rgb = renderer.render_array()  # returns numpy array
        renderer.close()
    """

    def __init__(self, game: PeggleGame, scale: float = 1.0):
        self.game = game
        self.scale = scale
        self.width = int(BOARD_WIDTH * scale)
        self.height = int(BOARD_HEIGHT * scale)
        self._surface: pygame.Surface | None = None
        self._display: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._clock: pygame.time.Clock | None = None
        self._initialized = False
        self._aim_angle: float = 0.0  # For drawing aim line

    def init(self, headless: bool = False) -> None:
        """Initialize Pygame (display or headless)."""
        if not HAS_PYGAME:
            raise ImportError("pygame is required for rendering")

        if not pygame.get_init():
            pygame.init()

        if headless:
            self._surface = pygame.Surface((self.width, self.height))
        else:
            self._display = pygame.display.set_mode(
                (self.width, self.height))
            pygame.display.set_caption("Peggle RL")
            self._surface = self._display
            self._clock = pygame.time.Clock()

        pygame.font.init()
        self._font = pygame.font.SysFont("consolas", int(14 * self.scale))
        self._initialized = True

    def close(self) -> None:
        if self._initialized and HAS_PYGAME:
            pygame.quit()
            self._initialized = False

    def set_aim_angle(self, angle: float) -> None:
        """Set the current aim angle for drawing the aim line."""
        self._aim_angle = angle

    def render(self, fps: int = 60) -> None:
        """Render to the display window."""
        if not self._initialized:
            self.init()

        self._draw()

        if self._display is not None:
            pygame.display.flip()
            if self._clock:
                self._clock.tick(fps)

    def render_array(self) -> np.ndarray:
        """Render and return as (H, W, 3) uint8 numpy array."""
        if not self._initialized:
            self.init(headless=True)

        self._draw()
        arr = pygame.surfarray.array3d(self._surface)
        # pygame gives (W, H, 3), transpose to (H, W, 3)
        return arr.transpose(1, 0, 2)

    def _s(self, val: float) -> int:
        """Scale a coordinate."""
        return int(val * self.scale)

    def _draw(self) -> None:
        """Draw the full game state."""
        surf = self._surface
        game = self.game
        s = self._s

        # Background
        surf.fill(BG_COLOR)

        # Rods
        for rod in game.rods:
            pygame.draw.line(
                surf, ROD_COLOR,
                (s(rod.x1), s(rod.y1)),
                (s(rod.x2), s(rod.y2)),
                max(1, s(4))
            )

        # Bricks
        for brick in game.bricks:
            if brick.hit:
                continue
            color = (BRICK_LIT_COLORS if brick.lit
                     else BRICK_COLORS).get(brick.peg_type, BRICK_COLORS[PegType.NORMAL])
            x1, y1, x2, y2 = brick_endpoints(
                brick.x, brick.y, brick.length, brick.angle)
            thickness = max(2, s(brick.width))
            pygame.draw.line(
                surf, color,
                (s(x1), s(y1)), (s(x2), s(y2)),
                thickness
            )

        # Pegs
        for peg in game.pegs:
            if peg.hit:
                continue
            color = (PEG_LIT_COLORS if peg.lit
                     else PEG_COLORS).get(peg.peg_type, PEG_COLORS[PegType.NORMAL])
            pygame.draw.circle(
                surf, color,
                (s(peg.x), s(peg.y)),
                max(2, s(peg.radius))
            )

        # Bucket
        bx = s(game.bucket.x)
        by = s(game.bucket.y)
        bw = s(game.bucket.width)
        bh = s(15)
        pygame.draw.rect(
            surf, BUCKET_COLOR,
            (bx - bw // 2, by, bw, bh),
            2
        )

        # Launcher
        lx = s(LAUNCHER_X)
        ly = s(LAUNCHER_Y)
        pygame.draw.circle(surf, LAUNCHER_COLOR, (lx, ly), s(12), 2)

        # Aim line (when aiming)
        from peggle_rl.sim.engine import GameState
        if game.state == GameState.AIMING:
            aim_rad = math.radians(self._aim_angle)
            end_x = LAUNCHER_X + math.sin(aim_rad) * 60
            end_y = LAUNCHER_Y + math.cos(aim_rad) * 60
            pygame.draw.line(
                surf, (255, 255, 100),
                (lx, ly), (s(end_x), s(end_y)), 2
            )

        # Ball
        if game.ball.active:
            bpos = game.ball.pos
            pygame.draw.circle(
                surf, BALL_COLOR,
                (s(bpos.x), s(bpos.y)),
                max(2, s(game.ball.radius))
            )
            pygame.draw.circle(
                surf, BALL_OUTLINE,
                (s(bpos.x), s(bpos.y)),
                max(2, s(game.ball.radius)), 1
            )

        # HUD
        self._draw_hud(surf)

    def _draw_hud(self, surf: pygame.Surface) -> None:
        """Draw score, balls remaining, orange count."""
        if self._font is None:
            return

        game = self.game
        y = 5

        texts = [
            (f"Score: {game.score:,}", HUD_TEXT),
            (f"Balls: {game.balls_remaining}", HUD_TEXT),
            (f"Orange: {game.orange_remaining}", ORANGE_TEXT),
            (f"Shot: {game.shot_count}", HUD_TEXT),
        ]

        for text, color in texts:
            rendered = self._font.render(text, True, color)
            surf.blit(rendered, (5, y))
            y += rendered.get_height() + 2
