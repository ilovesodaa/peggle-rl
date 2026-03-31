"""
Peggle Gymnasium environments.

Two environments:
  - PeggleSim-v0: Simulator with vector observations (fast, for training)
  - PeggleSimRender-v0: Simulator with RGB image observations

Action spaces (configurable):
  - Continuous: Box(-97, 97) for angle + Discrete(2) for power activation
  - Discrete: Discrete(392) discretized angles (-97 to +97 in 0.5 deg steps)
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from peggle_rl.levels.parser import Level, PegType
from peggle_rl.levels.catalog import (
    LEVEL_ORDER, get_level_data, STAGE_LEVELS, STAGE_POWERS,
)
from peggle_rl.sim.engine import (
    PeggleGame, GameState, Power,
    STARTING_BALLS,
)
from peggle_rl.sim.physics import (
    BOARD_WIDTH, BOARD_HEIGHT, ANGLE_MIN, ANGLE_MAX,
    LAUNCHER_X, LAUNCHER_Y,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PEGS = 200          # Max pegs/bricks tracked in observation
DISCRETE_STEPS = 390    # -97 to +97 in 0.5-degree steps = 388 + 2 endpoints
ANGLE_STEP = 0.5


def _angle_from_discrete(action: int) -> float:
    """Convert discrete action index to angle in degrees."""
    return ANGLE_MIN + action * ANGLE_STEP


def _power_from_stage(stage_idx: int) -> Power:
    """Get the power for a given stage (0-indexed)."""
    power_map = {
        "SuperGuide": Power.SUPER_GUIDE,
        "Multiball": Power.MULTIBALL,
        "Pyramid": Power.PYRAMID,
        "SpaceBlast": Power.SPACE_BLAST,
        "Flippers": Power.FLIPPERS,
        "SpookyBall": Power.SPOOKY_BALL,
        "FlowerPower": Power.FLOWER_POWER,
        "LuckySpin": Power.LUCKY_SPIN,
        "Fireball": Power.FIREBALL,
        "ZenBall": Power.ZEN_BALL,
    }
    return power_map.get(STAGE_POWERS[stage_idx], Power.NONE)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def compute_reward(
    game: PeggleGame,
    prev_orange: int,
    prev_score: int,
    pegs_hit: int,
) -> float:
    """
    Compute step reward after a shot resolves.

    Components:
      - Orange peg hit: +10 per orange cleared
      - Score delta: +0.001 * (new_score - old_score)
      - Long shot bonus: +5 if 10+ pegs hit
      - Free ball: +3
      - Level complete: +50
      - Game over (loss): -10
      - Per-shot cost: -0.5 (encourages efficiency)
    """
    reward = -0.5  # Shot cost

    # Orange pegs cleared this shot
    orange_cleared = prev_orange - game.orange_remaining
    reward += orange_cleared * 10.0

    # Score delta
    score_delta = game.score - prev_score
    reward += score_delta * 0.001

    # Peg hits
    reward += pegs_hit * 0.2

    # Long shot
    if pegs_hit >= 10:
        reward += 5.0

    # Free ball
    if game.free_ball_earned:
        reward += 3.0

    # Terminal
    if game.level_won:
        reward += 50.0
    elif game.state == GameState.GAME_OVER:
        reward -= 10.0

    return reward


# ---------------------------------------------------------------------------
# PeggleSim-v0: Vector observation environment
# ---------------------------------------------------------------------------

class PeggleSimEnv(gym.Env):
    """
    Peggle simulator environment with vector observations.

    Observation space: Dict with:
      - "global": Box(5,) [balls_remaining, orange_remaining, total_remaining,
                           bucket_x (normalized), score (normalized)]
      - "pegs": Box(MAX_PEGS, 5) [x, y, radius, type, active] per peg/brick

    Action space (configurable):
      - "continuous": Box([-97], [97]) angle
      - "discrete": Discrete(DISCRETE_STEPS) angle bins
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        level_name: str | None = None,
        stage: int | None = None,
        action_mode: str = "continuous",  # "continuous" or "discrete"
        render_mode: str | None = None,
        max_shots: int = 50,
        seed: int | None = None,
        randomize_level: bool = False,
    ):
        super().__init__()

        self.action_mode = action_mode
        self.render_mode = render_mode
        self.max_shots = max_shots
        self.randomize_level = randomize_level
        self._seed = seed
        self._renderer = None

        # Level selection
        self._level_name = level_name
        self._stage = stage
        self._level_names: list[str] = []

        if level_name:
            self._level_names = [level_name]
        elif stage is not None:
            self._level_names = STAGE_LEVELS[stage]
        else:
            self._level_names = LEVEL_ORDER.copy()

        # Load first level to set up spaces
        self._current_level_name = self._level_names[0]
        self._level_data: Level = get_level_data(self._current_level_name)

        # Determine power
        self._power = Power.NONE
        if stage is not None:
            self._power = _power_from_stage(stage)

        # Game engine
        self.game = PeggleGame(
            self._level_data, power=self._power, seed=seed
        )

        # Action space
        if action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.float32(ANGLE_MIN),
                high=np.float32(ANGLE_MAX),
                shape=(1,),
                dtype=np.float32,
            )
        elif action_mode == "discrete":
            n_actions = int((ANGLE_MAX - ANGLE_MIN) / ANGLE_STEP) + 1
            self.action_space = spaces.Discrete(n_actions)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        # Observation space
        self.observation_space = spaces.Dict({
            "global": spaces.Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([20, 200, 400, 1, 1], dtype=np.float32),
            ),
            "pegs": spaces.Box(
                low=-1.0, high=2.0,
                shape=(MAX_PEGS, 5),
                dtype=np.float32,
            ),
        })

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build observation from game state."""
        game = self.game

        # Global state
        global_obs = np.array([
            game.balls_remaining / STARTING_BALLS,
            game.orange_remaining / 25.0,
            game.total_pegs_remaining / max(1, len(game.pegs) + len(game.bricks)),
            game.bucket.x / BOARD_WIDTH,
            min(game.score / 100000.0, 1.0),
        ], dtype=np.float32)

        # Peg observations: [x_norm, y_norm, radius_norm, type_onehot_orange, active]
        peg_obs = np.zeros((MAX_PEGS, 5), dtype=np.float32)
        idx = 0

        for p in game.pegs:
            if idx >= MAX_PEGS:
                break
            if p.hit:
                continue
            peg_obs[idx] = [
                p.x / BOARD_WIDTH,
                p.y / BOARD_HEIGHT,
                p.radius / 20.0,
                1.0 if p.peg_type == PegType.GOAL else 0.0,
                1.0,
            ]
            idx += 1

        for b in game.bricks:
            if idx >= MAX_PEGS:
                break
            if b.hit:
                continue
            peg_obs[idx] = [
                b.x / BOARD_WIDTH,
                b.y / BOARD_HEIGHT,
                b.length / 100.0,
                1.0 if b.peg_type == PegType.GOAL else 0.0,
                1.0,
            ]
            idx += 1

        return {"global": global_obs, "pegs": peg_obs}

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        # Optionally randomize level
        if self.randomize_level and len(self._level_names) > 1:
            self._current_level_name = self.np_random.choice(
                self._level_names)
            self._level_data = get_level_data(self._current_level_name)
            self.game = PeggleGame(
                self._level_data, power=self._power,
                seed=seed if seed is not None else self._seed,
            )

        self.game.reset(seed=seed)

        obs = self._get_obs()
        info = self.game.get_info()
        info["level_name"] = self._current_level_name
        return obs, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one shot (aim + simulate until ball resolves).
        """
        # Decode action to angle
        if self.action_mode == "continuous":
            if isinstance(action, np.ndarray):
                action = action.item()
            angle = float(np.clip(float(action), ANGLE_MIN, ANGLE_MAX))
        else:
            angle = _angle_from_discrete(int(action))

        # Record pre-shot state
        prev_orange = self.game.orange_remaining
        prev_score = self.game.score

        # Shoot and simulate
        self.game.shoot(angle)
        self.game.tick_until_done()

        # Compute reward
        pegs_hit = (len(self.game.pegs_hit_this_shot)
                    + len(self.game.bricks_hit_this_shot))
        reward = compute_reward(
            self.game, prev_orange, prev_score, pegs_hit
        )

        # Check termination
        terminated = self.game.is_terminal
        truncated = self.game.shot_count >= self.max_shots

        obs = self._get_obs()
        info = self.game.get_info()
        info["level_name"] = self._current_level_name
        info["pegs_hit_this_shot"] = pegs_hit
        info["angle"] = angle

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from peggle_rl.sim.renderer import PeggleRenderer
            self._renderer = PeggleRenderer(self.game, scale=1.0)
            self._renderer.init(headless=(self.render_mode == "rgb_array"))

        if self.render_mode == "human":
            self._renderer.render()
        elif self.render_mode == "rgb_array":
            return self._renderer.render_array()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_registered = False

def register_envs():
    """Register Peggle environments with Gymnasium (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    gym.register(
        id="PeggleSim-v0",
        entry_point="peggle_rl.sim.env:PeggleSimEnv",
        kwargs={"action_mode": "continuous"},
    )
    gym.register(
        id="PeggleSimDiscrete-v0",
        entry_point="peggle_rl.sim.env:PeggleSimEnv",
        kwargs={"action_mode": "discrete"},
    )
