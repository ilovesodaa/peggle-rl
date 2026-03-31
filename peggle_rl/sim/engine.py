"""
Peggle game engine.

Manages the full game state: pegs, ball, bucket, scoring, and powers.
Drives the physics simulation and tracks game progression.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable

from peggle_rl.levels.parser import (
    Level, CircleEntry, BrickEntry, RodEntry, PolygonEntry,
    PegType, BrickType, Movement, MovementShape,
)
from peggle_rl.sim.physics import (
    Vec2, Ball, Bucket, SpatialHash,
    collide_ball_circle, collide_ball_segment, collide_ball_walls,
    ball_in_bucket, ball_out_of_bounds, brick_endpoints, angle_to_vec,
    BOARD_WIDTH, BOARD_HEIGHT, LAUNCHER_X, LAUNCHER_Y,
    BALL_INITIAL_SPEED, BALL_RADIUS, BALL_RESTITUTION,
    ANGLE_MIN, ANGLE_MAX, TICKS_PER_SECOND,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTING_BALLS = 10
MAX_TICKS_PER_SHOT = 60 * 30   # 30 seconds max per shot (prevent stuck balls)
STUCK_SPEED_THRESHOLD = 0.15   # Ball considered stuck if slower than this
STUCK_TICK_LIMIT = 300         # Remove ball after 5s of being slow

# Scoring (matching Peggle Deluxe)
BASE_SCORES = {
    PegType.NORMAL: 10,
    PegType.GOAL: 100,
    PegType.POWERUP: 10,
    PegType.SCORE: 500,
}

# Score multiplier based on remaining orange pegs
SCORE_MULTIPLIERS = [
    (25, 1), (20, 2), (15, 3), (10, 5),
    (5, 10), (3, 50), (1, 100), (0, 500),
]


class GameState(IntEnum):
    AIMING = auto()           # Waiting for player to shoot
    BALL_IN_PLAY = auto()     # Ball is moving, physics active
    SHOT_COMPLETE = auto()     # Ball left board or caught, processing
    LEVEL_COMPLETE = auto()    # All orange pegs cleared
    GAME_OVER = auto()         # No balls remaining, orange pegs remain


class Power(IntEnum):
    NONE = 0
    SUPER_GUIDE = 1      # Bjorn - shows trajectory
    MULTIBALL = 2        # Jimmy - 2 extra balls
    PYRAMID = 3          # Kat Tut - triangle flipper at bottom
    SPACE_BLAST = 4      # Splork - explodes nearby pegs
    FLIPPERS = 5         # Claude - flippers at bottom
    SPOOKY_BALL = 6      # Renfield - ball reappears at top
    FLOWER_POWER = 7     # Tula - bonus score flowers
    LUCKY_SPIN = 8       # Warren - slot machine
    FIREBALL = 9         # Cinderbottom - ball goes through pegs
    ZEN_BALL = 10        # Master Hu - auto-aims


# ---------------------------------------------------------------------------
# Peg / Brick game objects
# ---------------------------------------------------------------------------

@dataclass
class PegObject:
    """A peg (circle) in the game."""
    x: float
    y: float
    radius: float
    peg_type: PegType
    hit: bool = False
    lit: bool = False          # Visually lit after hit
    variable: bool = False     # Can be randomly assigned orange
    crumble: bool = False      # Disappears quickly after hit
    # Movement (for moving pegs)
    movement: Movement | None = None
    base_x: float = 0.0
    base_y: float = 0.0
    rolly: float | None = None
    bouncy: float | None = None


@dataclass
class BrickObject:
    """A brick (line segment) in the game."""
    x: float
    y: float
    length: float
    angle: float               # radians
    width: float               # thickness
    peg_type: PegType
    brick_type: BrickType = BrickType.STRAIGHT
    hit: bool = False
    lit: bool = False
    variable: bool = False
    crumble: bool = False
    movement: Movement | None = None
    base_x: float = 0.0
    base_y: float = 0.0
    rolly: float | None = None
    bouncy: float | None = None


@dataclass
class RodObject:
    """A rod (wall/obstacle line segment)."""
    x1: float
    y1: float
    x2: float
    y2: float


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------

class PeggleGame:
    """
    Full Peggle game simulation for a single level.

    Usage:
        game = PeggleGame(level_data)
        game.reset()
        while not game.is_terminal:
            obs = game.get_observation()
            game.shoot(angle_degrees)
            while game.state == GameState.BALL_IN_PLAY:
                game.tick()
    """

    def __init__(
        self,
        level: Level,
        power: Power = Power.NONE,
        seed: int | None = None,
        num_orange: int = 25,
    ):
        self.level = level
        self.power = power
        self.num_orange = num_orange
        self._rng = random.Random(seed)

        # Game objects (populated on reset)
        self.pegs: list[PegObject] = []
        self.bricks: list[BrickObject] = []
        self.rods: list[RodObject] = []
        self.ball = Ball()
        self.bucket = Bucket()
        self.spatial_hash = SpatialHash(cell_size=40.0)

        # Game state
        self.state = GameState.AIMING
        self.balls_remaining = STARTING_BALLS
        self.score = 0
        self.tick_count = 0
        self.shot_count = 0
        self.shot_tick = 0
        self.pegs_hit_this_shot: list[int] = []    # indices
        self.bricks_hit_this_shot: list[int] = []
        self.free_ball_earned = False
        self.power_active = False
        self.spooky_used = False

        # Stats
        self.total_pegs_hit = 0
        self.total_orange_hit = 0
        self.long_shots = 0    # Shots that hit 10+ pegs

    @property
    def orange_remaining(self) -> int:
        count = sum(1 for p in self.pegs
                    if p.peg_type == PegType.GOAL and not p.hit)
        count += sum(1 for b in self.bricks
                     if b.peg_type == PegType.GOAL and not b.hit)
        return count

    @property
    def total_pegs_remaining(self) -> int:
        count = sum(1 for p in self.pegs if not p.hit)
        count += sum(1 for b in self.bricks if not b.hit)
        return count

    @property
    def is_terminal(self) -> bool:
        return self.state in (GameState.LEVEL_COMPLETE, GameState.GAME_OVER)

    @property
    def level_won(self) -> bool:
        return self.state == GameState.LEVEL_COMPLETE

    def reset(self, seed: int | None = None) -> None:
        """Reset the game to initial state."""
        if seed is not None:
            self._rng = random.Random(seed)

        self._build_objects()
        self._assign_orange_pegs()
        self._build_spatial_hash()

        self.ball = Ball()
        self.bucket = Bucket()
        self.state = GameState.AIMING
        self.balls_remaining = STARTING_BALLS
        self.score = 0
        self.tick_count = 0
        self.shot_count = 0
        self.shot_tick = 0
        self.pegs_hit_this_shot.clear()
        self.bricks_hit_this_shot.clear()
        self.free_ball_earned = False
        self.power_active = False
        self.spooky_used = False
        self.total_pegs_hit = 0
        self.total_orange_hit = 0
        self.long_shots = 0

    def _build_objects(self) -> None:
        """Convert parsed level data into game objects."""
        self.pegs.clear()
        self.bricks.clear()
        self.rods.clear()

        for c in self.level.circles:
            ptype = c.peg_info.peg_type if c.peg_info else PegType.NORMAL
            variable = c.peg_info.variable if c.peg_info else False
            crumble = c.peg_info.crumble if c.peg_info else False
            peg = PegObject(
                x=c.x, y=c.y, radius=c.radius,
                peg_type=ptype, variable=variable, crumble=crumble,
                movement=c.movement, base_x=c.x, base_y=c.y,
                rolly=c.rolly, bouncy=c.bouncy,
            )
            self.pegs.append(peg)

        for b in self.level.bricks:
            ptype = b.peg_info.peg_type if b.peg_info else PegType.NORMAL
            variable = b.peg_info.variable if b.peg_info else False
            crumble = b.peg_info.crumble if b.peg_info else False
            brick = BrickObject(
                x=b.x, y=b.y, length=b.length, angle=b.angle,
                width=max(b.width, 4.0),
                peg_type=ptype, brick_type=b.brick_type,
                variable=variable, crumble=crumble,
                movement=b.movement, base_x=b.x, base_y=b.y,
                rolly=b.rolly, bouncy=b.bouncy,
            )
            self.bricks.append(brick)

        for r in self.level.rods:
            self.rods.append(RodObject(r.x1, r.y1, r.x2, r.y2))

    def _assign_orange_pegs(self) -> None:
        """
        Randomly assign orange pegs from the variable pool.

        In Peggle Deluxe, each level starts with exactly 25 orange targets
        randomly chosen from all variable pegs/bricks. Fixed-type pegs
        (already set to GOAL/GREEN/PURPLE) keep their assignment.
        """
        # Collect variable-eligible objects (type NORMAL + variable flag)
        variable_indices: list[tuple[str, int]] = []

        for i, p in enumerate(self.pegs):
            if p.variable and p.peg_type == PegType.NORMAL:
                variable_indices.append(("peg", i))
        for i, b in enumerate(self.bricks):
            if b.variable and b.peg_type == PegType.NORMAL:
                variable_indices.append(("brick", i))

        # Count how many are already orange
        already_orange = sum(
            1 for p in self.pegs if p.peg_type == PegType.GOAL
        ) + sum(
            1 for b in self.bricks if b.peg_type == PegType.GOAL
        )

        need_orange = max(0, self.num_orange - already_orange)

        if need_orange > 0 and variable_indices:
            chosen = self._rng.sample(
                variable_indices,
                min(need_orange, len(variable_indices))
            )
            for kind, idx in chosen:
                if kind == "peg":
                    self.pegs[idx].peg_type = PegType.GOAL
                else:
                    self.bricks[idx].peg_type = PegType.GOAL

    def _build_spatial_hash(self) -> None:
        """Build spatial hash from all active pegs and bricks."""
        self.spatial_hash.clear()

        for i, p in enumerate(self.pegs):
            if not p.hit:
                self.spatial_hash.insert(i, p.x, p.y, p.radius)

        # Bricks: index offset by len(pegs)
        offset = len(self.pegs)
        for i, b in enumerate(self.bricks):
            if not b.hit:
                x1, y1, x2, y2 = brick_endpoints(
                    b.x, b.y, b.length, b.angle)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                hw = abs(x2 - x1) / 2 + b.width
                hh = abs(y2 - y1) / 2 + b.width
                self.spatial_hash.insert_rect(
                    offset + i, cx, cy, hw, hh)

        # Rods: index offset by len(pegs) + len(bricks)
        rod_offset = offset + len(self.bricks)
        for i, r in enumerate(self.rods):
            cx = (r.x1 + r.x2) / 2
            cy = (r.y1 + r.y2) / 2
            hw = abs(r.x2 - r.x1) / 2 + 10
            hh = abs(r.y2 - r.y1) / 2 + 10
            self.spatial_hash.insert_rect(
                rod_offset + i, cx, cy, hw, hh)

    def shoot(self, angle_deg: float) -> None:
        """Launch the ball at the given angle (degrees, 0=down)."""
        if self.state != GameState.AIMING:
            return

        angle_deg = max(ANGLE_MIN, min(ANGLE_MAX, angle_deg))

        self.ball = Ball(
            pos=Vec2(LAUNCHER_X, LAUNCHER_Y),
            radius=BALL_RADIUS,
        )
        self.ball.launch(angle_deg)

        self.state = GameState.BALL_IN_PLAY
        self.shot_count += 1
        self.shot_tick = 0
        self.pegs_hit_this_shot.clear()
        self.bricks_hit_this_shot.clear()
        self.free_ball_earned = False
        self.power_active = False
        self.spooky_used = False

    def tick(self) -> None:
        """Advance the simulation by one tick."""
        if self.state != GameState.BALL_IN_PLAY:
            return

        self.tick_count += 1
        self.shot_tick += 1

        # Update moving objects
        self._update_movement()
        self.bucket.update()

        # Update ball physics
        self.ball.update()

        # Wall collisions
        collide_ball_walls(self.ball)

        # Peg/brick collisions
        self._check_collisions()

        # Check for ball leaving play
        if ball_out_of_bounds(self.ball):
            if ball_in_bucket(self.ball, self.bucket):
                self.free_ball_earned = True
            self._end_shot()
        elif self.shot_tick > MAX_TICKS_PER_SHOT:
            # Safety timeout
            self._end_shot()
        else:
            # Stuck detection
            speed = self.ball.vel.length()
            if speed < STUCK_SPEED_THRESHOLD:
                self.ball.stuck_timer += 1
                if self.ball.stuck_timer > STUCK_TICK_LIMIT:
                    self._end_shot()
            else:
                self.ball.stuck_timer = 0

    def tick_until_done(self) -> None:
        """Run ticks until the current shot resolves."""
        while self.state == GameState.BALL_IN_PLAY:
            self.tick()

    def _check_collisions(self) -> None:
        """Check ball against all nearby objects via spatial hash."""
        if not self.ball.active:
            return

        candidates = self.spatial_hash.query(
            self.ball.pos.x, self.ball.pos.y,
            self.ball.radius + 30  # padding for large pegs
        )

        peg_count = len(self.pegs)
        brick_count = len(self.bricks)
        rod_offset = peg_count + brick_count

        for idx in candidates:
            if idx < peg_count:
                # Peg collision
                peg = self.pegs[idx]
                if peg.hit:
                    continue
                restitution = peg.bouncy if peg.bouncy else BALL_RESTITUTION
                if collide_ball_circle(
                    self.ball, peg.x, peg.y, peg.radius, restitution
                ):
                    peg.hit = True
                    peg.lit = True
                    self.pegs_hit_this_shot.append(idx)
                    self._on_peg_hit(peg)

            elif idx < rod_offset:
                # Brick collision
                bi = idx - peg_count
                brick = self.bricks[bi]
                if brick.hit:
                    continue
                x1, y1, x2, y2 = brick_endpoints(
                    brick.x, brick.y, brick.length, brick.angle)
                restitution = brick.bouncy if brick.bouncy else BALL_RESTITUTION
                thickness = brick.width / 2
                if collide_ball_segment(
                    self.ball, x1, y1, x2, y2, thickness, restitution
                ):
                    brick.hit = True
                    brick.lit = True
                    self.bricks_hit_this_shot.append(bi)
                    self._on_brick_hit(brick)

            else:
                # Rod collision (never consumed, always present)
                ri = idx - rod_offset
                if ri < len(self.rods):
                    rod = self.rods[ri]
                    collide_ball_segment(
                        self.ball, rod.x1, rod.y1, rod.x2, rod.y2,
                        4.0, BALL_RESTITUTION
                    )

    def _on_peg_hit(self, peg: PegObject) -> None:
        """Handle scoring when a peg is hit."""
        self.total_pegs_hit += 1
        if peg.peg_type == PegType.GOAL:
            self.total_orange_hit += 1

        # Score
        base = BASE_SCORES.get(peg.peg_type, 10)
        multiplier = self._score_multiplier()
        self.score += base * multiplier

        # Green peg activates power
        if peg.peg_type == PegType.POWERUP:
            self.power_active = True
            self._activate_power()

    def _on_brick_hit(self, brick: BrickObject) -> None:
        """Handle scoring when a brick is hit."""
        self.total_pegs_hit += 1
        if brick.peg_type == PegType.GOAL:
            self.total_orange_hit += 1

        base = BASE_SCORES.get(brick.peg_type, 10)
        multiplier = self._score_multiplier()
        self.score += base * multiplier

        if brick.peg_type == PegType.POWERUP:
            self.power_active = True
            self._activate_power()

    def _score_multiplier(self) -> int:
        """Get current score multiplier based on orange pegs remaining."""
        remaining = self.orange_remaining
        for threshold, mult in SCORE_MULTIPLIERS:
            if remaining >= threshold:
                return mult
        return 1

    def _activate_power(self) -> None:
        """Activate the character's special power."""
        if self.power == Power.MULTIBALL:
            # In a real implementation, spawn 2 extra balls
            # For RL simplicity, award bonus points
            self.score += 1000
        elif self.power == Power.SPACE_BLAST:
            # Clear nearby pegs (simplified: clear pegs within 80px)
            self._space_blast()
        elif self.power == Power.FIREBALL:
            # Ball passes through pegs without bouncing (simplified)
            pass  # Would need physics mode change
        elif self.power == Power.SPOOKY_BALL:
            self.spooky_used = True
        # Other powers are either visual (SuperGuide), passive (ZenBall),
        # or complex (Flippers, Pyramid, FlowerPower, LuckySpin)

    def _space_blast(self) -> None:
        """Splork's SpaceBlast: clear pegs near the green peg."""
        blast_radius = 80.0
        for peg in self.pegs:
            if peg.hit or peg.peg_type == PegType.POWERUP:
                continue
            if peg.peg_type == PegType.POWERUP:
                continue
            # Find distance to the ball (approximate blast center)
            dx = peg.x - self.ball.pos.x
            dy = peg.y - self.ball.pos.y
            if dx * dx + dy * dy < blast_radius * blast_radius:
                peg.hit = True
                peg.lit = True
                self._on_peg_hit(peg)

    def _end_shot(self) -> None:
        """Process end of shot: remove hit pegs, check win/loss."""
        self.ball.active = False

        # Count hits this shot
        hits = len(self.pegs_hit_this_shot) + len(self.bricks_hit_this_shot)
        if hits >= 10:
            self.long_shots += 1

        # Spooky ball: ball reappears at top
        if self.spooky_used and not self.free_ball_earned:
            self.free_ball_earned = True

        # Deduct ball (unless free ball earned)
        if not self.free_ball_earned:
            self.balls_remaining -= 1

        # Rebuild spatial hash (removes hit pegs)
        self._build_spatial_hash()

        # Check win/loss
        if self.orange_remaining == 0:
            self.state = GameState.LEVEL_COMPLETE
            # Fever bonus: remaining balls * 10000
            self.score += self.balls_remaining * 10000
        elif self.balls_remaining <= 0:
            self.state = GameState.GAME_OVER
        else:
            self.state = GameState.AIMING

    def _update_movement(self) -> None:
        """Update positions of moving pegs/bricks."""
        for peg in self.pegs:
            if peg.movement and not peg.hit:
                self._apply_movement(peg, peg.movement)

        for brick in self.bricks:
            if brick.movement and not brick.hit:
                self._apply_movement(brick, brick.movement)

    def _apply_movement(self, obj, mov: Movement) -> None:
        """Apply movement path to an object."""
        if mov.time_period <= 0:
            return

        # Calculate phase from tick count
        total_cycle = mov.time_period + mov.pause1 + mov.pause2
        if total_cycle <= 0:
            return

        t = (self.tick_count % (total_cycle * TICKS_PER_SECOND // 100))
        phase = t / (total_cycle * TICKS_PER_SECOND / 100)

        r1 = mov.radius1 if mov.radius1 != 0 else 0
        r2 = mov.radius2 if mov.radius2 != 0 else r1

        if mov.shape == MovementShape.VERTICAL_CYCLE:
            offset_y = math.sin(phase * 2 * math.pi) * r2
            obj.y = obj.base_y + offset_y

        elif mov.shape == MovementShape.HORIZONTAL_CYCLE:
            offset_x = math.cos(phase * 2 * math.pi) * r1
            obj.x = obj.base_x + offset_x

        elif mov.shape == MovementShape.CIRCLE:
            angle = phase * 2 * math.pi
            obj.x = obj.base_x + math.cos(angle) * r1
            obj.y = obj.base_y + math.sin(angle) * r2

        elif mov.shape == MovementShape.ROTATE:
            # For pegs, rotation doesn't change position (they're circular)
            # For bricks, it would change angle - simplified here
            pass

    # ---------------------------------------------------------------------------
    # Observation / info for RL
    # ---------------------------------------------------------------------------

    def get_observation(self) -> dict:
        """
        Get current game state as an observation dictionary.
        This is consumed by the Gymnasium environment.
        """
        peg_data = []
        for p in self.pegs:
            if not p.hit:
                peg_data.append({
                    "x": p.x, "y": p.y, "radius": p.radius,
                    "type": int(p.peg_type), "lit": p.lit,
                })

        brick_data = []
        for b in self.bricks:
            if not b.hit:
                brick_data.append({
                    "x": b.x, "y": b.y, "length": b.length,
                    "angle": b.angle, "width": b.width,
                    "type": int(b.peg_type), "lit": b.lit,
                })

        return {
            "balls_remaining": self.balls_remaining,
            "score": self.score,
            "orange_remaining": self.orange_remaining,
            "total_remaining": self.total_pegs_remaining,
            "bucket_x": self.bucket.x,
            "pegs": peg_data,
            "bricks": brick_data,
            "state": int(self.state),
        }

    def get_info(self) -> dict:
        """Get auxiliary info (not part of obs, for logging/debugging)."""
        return {
            "shot_count": self.shot_count,
            "total_pegs_hit": self.total_pegs_hit,
            "total_orange_hit": self.total_orange_hit,
            "long_shots": self.long_shots,
            "level_won": self.level_won,
            "ticks": self.tick_count,
        }
