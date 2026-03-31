"""
Peggle physics engine.

Simulates ball-peg/brick/wall collisions with Peggle Deluxe physics:
  - Gravity: 0.05 pixels/tick^2 (from Haggle SDK PhysObj.mGravity)
  - 60 ticks per second
  - Ball radius: 7.5 pixels
  - Board: 800 x 600 pixels (playable area ~580 from top)
  - Walls at x=0 and x=800, ceiling at y=0
  - Ball lost when y > 600 (unless caught by bucket)

Collision detection uses spatial hashing for performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peggle_rl.sim.engine import Peg, Brick

# ---------------------------------------------------------------------------
# Constants (matching Peggle Deluxe)
# ---------------------------------------------------------------------------

BOARD_WIDTH = 800
BOARD_HEIGHT = 600
PLAYABLE_HEIGHT = 580          # Below this is the bucket zone

GRAVITY = 0.05                 # pixels/tick^2 (from Haggle mGravity)
TICKS_PER_SECOND = 60
DT = 1.0 / TICKS_PER_SECOND

BALL_RADIUS = 7.5
BALL_RESTITUTION = 0.7         # Default bounce coefficient
BALL_MAX_SPEED = 20.0          # Terminal velocity clamp
BALL_INITIAL_SPEED = 8.0       # Launch speed

WALL_LEFT = 0.0
WALL_RIGHT = 800.0
CEILING_Y = 0.0

# Bucket (free ball catcher) - moves back and forth
BUCKET_Y = 585.0
BUCKET_WIDTH = 100.0
BUCKET_SPEED = 1.5             # pixels/tick
BUCKET_MIN_X = 100.0
BUCKET_MAX_X = 700.0

# Launcher
LAUNCHER_X = 400.0
LAUNCHER_Y = 25.0
ANGLE_MIN = -97.0              # degrees (from Haggle)
ANGLE_MAX = 97.0


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> Vec2:
        return Vec2(self.x * s, self.y * s)

    def __rmul__(self, s: float) -> Vec2:
        return Vec2(self.x * s, self.y * s)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalized(self) -> Vec2:
        mag = self.length()
        if mag < 1e-12:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def reflect(self, normal: Vec2) -> Vec2:
        """Reflect vector across a normal."""
        d = self.dot(normal)
        return Vec2(self.x - 2 * d * normal.x, self.y - 2 * d * normal.y)

    def copy(self) -> Vec2:
        return Vec2(self.x, self.y)


def angle_to_vec(angle_deg: float) -> Vec2:
    """Convert launch angle (0=down, negative=left, positive=right) to direction."""
    rad = math.radians(angle_deg)
    # In Peggle, 0 degrees = straight down, angle rotates the aim
    return Vec2(math.sin(rad), math.cos(rad))


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------

@dataclass
class Ball:
    pos: Vec2 = field(default_factory=lambda: Vec2(LAUNCHER_X, LAUNCHER_Y))
    vel: Vec2 = field(default_factory=Vec2)
    radius: float = BALL_RADIUS
    active: bool = False
    stuck_timer: int = 0       # Counter to detect stuck balls

    def launch(self, angle_deg: float, speed: float = BALL_INITIAL_SPEED):
        """Launch from the cannon at the given angle."""
        direction = angle_to_vec(angle_deg)
        self.vel = direction * speed
        self.active = True
        self.stuck_timer = 0

    def update(self):
        """Apply gravity and move the ball one tick."""
        if not self.active:
            return

        # Gravity (downward)
        self.vel.y += GRAVITY

        # Clamp speed
        speed = self.vel.length()
        if speed > BALL_MAX_SPEED:
            self.vel = self.vel.normalized() * BALL_MAX_SPEED

        # Move
        self.pos = self.pos + self.vel


# ---------------------------------------------------------------------------
# Bucket (free ball catcher)
# ---------------------------------------------------------------------------

@dataclass
class Bucket:
    x: float = 400.0           # Center x
    width: float = BUCKET_WIDTH
    y: float = BUCKET_Y
    speed: float = BUCKET_SPEED
    direction: int = 1          # 1 = right, -1 = left

    def update(self):
        self.x += self.speed * self.direction
        if self.x >= BUCKET_MAX_X:
            self.x = BUCKET_MAX_X
            self.direction = -1
        elif self.x <= BUCKET_MIN_X:
            self.x = BUCKET_MIN_X
            self.direction = 1

    def contains(self, ball: Ball) -> bool:
        """Check if the ball falls into the bucket."""
        if ball.pos.y < self.y - ball.radius:
            return False
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        return left <= ball.pos.x <= right


# ---------------------------------------------------------------------------
# Spatial hash for collision detection
# ---------------------------------------------------------------------------

class SpatialHash:
    """Grid-based spatial hash for fast broad-phase collision detection."""

    def __init__(self, cell_size: float = 40.0):
        self.cell_size = cell_size
        self._grid: dict[tuple[int, int], list[int]] = {}

    def clear(self):
        self._grid.clear()

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, idx: int, x: float, y: float, radius: float):
        """Insert an object covering a circle (x, y, radius)."""
        x0 = int((x - radius) // self.cell_size)
        x1 = int((x + radius) // self.cell_size)
        y0 = int((y - radius) // self.cell_size)
        y1 = int((y + radius) // self.cell_size)
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                key = (gx, gy)
                if key not in self._grid:
                    self._grid[key] = []
                self._grid[key].append(idx)

    def insert_rect(self, idx: int, x: float, y: float,
                    hw: float, hh: float):
        """Insert an object covering a rectangle centered at (x, y)."""
        x0 = int((x - hw) // self.cell_size)
        x1 = int((x + hw) // self.cell_size)
        y0 = int((y - hh) // self.cell_size)
        y1 = int((y + hh) // self.cell_size)
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                key = (gx, gy)
                if key not in self._grid:
                    self._grid[key] = []
                self._grid[key].append(idx)

    def query(self, x: float, y: float, radius: float) -> set[int]:
        """Get all object indices near a circle."""
        result: set[int] = set()
        x0 = int((x - radius) // self.cell_size)
        x1 = int((x + radius) // self.cell_size)
        y0 = int((y - radius) // self.cell_size)
        y1 = int((y + radius) // self.cell_size)
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                key = (gx, gy)
                if key in self._grid:
                    result.update(self._grid[key])
        return result


# ---------------------------------------------------------------------------
# Collision functions
# ---------------------------------------------------------------------------

def collide_ball_circle(ball: Ball, cx: float, cy: float,
                        cr: float, restitution: float = BALL_RESTITUTION
                        ) -> bool:
    """
    Circle-circle collision between ball and a circular peg.
    Returns True if collision occurred, modifies ball velocity.
    """
    dx = ball.pos.x - cx
    dy = ball.pos.y - cy
    dist_sq = dx * dx + dy * dy
    min_dist = ball.radius + cr

    if dist_sq >= min_dist * min_dist:
        return False
    if dist_sq < 1e-12:
        # Ball is exactly at peg center - push it out
        ball.pos.x = cx + min_dist
        ball.vel.x = abs(ball.vel.length()) * restitution
        return True

    dist = math.sqrt(dist_sq)
    # Normal from peg center to ball center
    nx = dx / dist
    ny = dy / dist

    # Separate ball from peg (push out)
    overlap = min_dist - dist
    ball.pos.x += nx * overlap
    ball.pos.y += ny * overlap

    # Reflect velocity
    normal = Vec2(nx, ny)
    vel_dot_n = ball.vel.dot(normal)
    if vel_dot_n < 0:  # Only if moving toward the peg
        ball.vel = ball.vel - normal * (2 * vel_dot_n)
        ball.vel = ball.vel * restitution

    return True


def collide_ball_segment(ball: Ball, x1: float, y1: float,
                         x2: float, y2: float, thickness: float = 4.0,
                         restitution: float = BALL_RESTITUTION
                         ) -> bool:
    """
    Collision between ball and a line segment (used for bricks and rods).
    The segment has a given thickness (half-width on each side).
    Returns True if collision occurred.
    """
    # Vector from p1 to p2
    ex = x2 - x1
    ey = y2 - y1
    seg_len_sq = ex * ex + ey * ey
    if seg_len_sq < 1e-12:
        # Degenerate segment - treat as circle
        return collide_ball_circle(ball, x1, y1, thickness, restitution)

    # Project ball center onto segment
    t = ((ball.pos.x - x1) * ex + (ball.pos.y - y1) * ey) / seg_len_sq
    t = clamp(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = x1 + t * ex
    closest_y = y1 + t * ey

    dx = ball.pos.x - closest_x
    dy = ball.pos.y - closest_y
    dist_sq = dx * dx + dy * dy
    min_dist = ball.radius + thickness

    if dist_sq >= min_dist * min_dist:
        return False
    if dist_sq < 1e-12:
        # Use segment normal
        seg_len = math.sqrt(seg_len_sq)
        nx = -ey / seg_len
        ny = ex / seg_len
        ball.pos.x = closest_x + nx * min_dist
        ball.pos.y = closest_y + ny * min_dist
        ball.vel = ball.vel.reflect(Vec2(nx, ny)) * restitution
        return True

    dist = math.sqrt(dist_sq)
    nx = dx / dist
    ny = dy / dist

    # Push out
    overlap = min_dist - dist
    ball.pos.x += nx * overlap
    ball.pos.y += ny * overlap

    # Reflect
    normal = Vec2(nx, ny)
    vel_dot_n = ball.vel.dot(normal)
    if vel_dot_n < 0:
        ball.vel = ball.vel - normal * (2 * vel_dot_n)
        ball.vel = ball.vel * restitution

    return True


def collide_ball_walls(ball: Ball) -> bool:
    """Bounce ball off left/right walls and ceiling. Returns True if hit."""
    hit = False

    # Left wall
    if ball.pos.x - ball.radius < WALL_LEFT:
        ball.pos.x = WALL_LEFT + ball.radius
        ball.vel.x = abs(ball.vel.x) * BALL_RESTITUTION
        hit = True

    # Right wall
    if ball.pos.x + ball.radius > WALL_RIGHT:
        ball.pos.x = WALL_RIGHT - ball.radius
        ball.vel.x = -abs(ball.vel.x) * BALL_RESTITUTION
        hit = True

    # Ceiling
    if ball.pos.y - ball.radius < CEILING_Y:
        ball.pos.y = CEILING_Y + ball.radius
        ball.vel.y = abs(ball.vel.y) * BALL_RESTITUTION
        hit = True

    return hit


def ball_in_bucket(ball: Ball, bucket: Bucket) -> bool:
    """Check if ball has been caught by the moving bucket."""
    if ball.pos.y + ball.radius < bucket.y:
        return False
    left = bucket.x - bucket.width / 2
    right = bucket.x + bucket.width / 2
    return left <= ball.pos.x <= right


def ball_out_of_bounds(ball: Ball) -> bool:
    """Check if ball has fallen below the board."""
    return ball.pos.y - ball.radius > BOARD_HEIGHT


# ---------------------------------------------------------------------------
# Brick geometry helpers
# ---------------------------------------------------------------------------

def brick_endpoints(x: float, y: float, length: float,
                    angle: float) -> tuple[float, float, float, float]:
    """
    Get the two endpoints of a brick's center line.
    angle is in radians. The brick is centered at (x, y).
    Returns (x1, y1, x2, y2).
    """
    half_len = length / 2
    dx = math.cos(angle) * half_len
    dy = math.sin(angle) * half_len
    return (x - dx, y - dy, x + dx, y + dy)
