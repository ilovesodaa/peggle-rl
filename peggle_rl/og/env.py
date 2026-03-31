"""
Gymnasium environment wrapping the real Peggle Deluxe game
via the Haggle mod named-pipe bridge.

Usage:
    env = PeggleOGEnv(action_mode="continuous")
    obs, info = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Requires:
  - Peggle Deluxe installed
  - Haggle mod loader (Haggle.exe) in the game directory
  - peggle-rl-bridge.dll + haggle-sdk.dll in the game's mods/ folder
  - pywin32 installed (pip install pywin32)
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from peggle_rl.og.pipe_client import (
    PeggleBridgeClient,
    PeggleBridgeError,
    GameState,
    PegInfo,
    ShotInfo,
)
from peggle_rl.sim.physics import ANGLE_MIN, ANGLE_MAX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PEGS = 200
DISCRETE_STEPS = 390
ANGLE_STEP = 0.5
BOARD_WIDTH = 800.0
BOARD_HEIGHT = 600.0
SHOT_TIMEOUT_MS = 15000


def _angle_from_discrete(action: int) -> float:
    return ANGLE_MIN + action * ANGLE_STEP


# ---------------------------------------------------------------------------
# OG Peggle Gymnasium env
# ---------------------------------------------------------------------------

class PeggleOGEnv(gym.Env):
    """
    Gymnasium environment for the real Peggle Deluxe game.

    Communicates with the running game via the Haggle named-pipe bridge.
    The game must be running and the bridge mod loaded before creating
    this environment.

    Observation space: Dict with:
      - "global": Box(3,) [orange_remaining_frac, total_remaining_frac, gun_angle_norm]
      - "pegs": Box(MAX_PEGS, 4) [x_norm, y_norm, type, active]

    Action space (configurable):
      - "continuous": Box([-97], [97]) for angle
      - "discrete": Discrete(N) for discretized angle bins
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        action_mode: str = "continuous",
        render_mode: str | None = None,
        max_shots: int = 50,
        shot_timeout_ms: int = SHOT_TIMEOUT_MS,
        connect_timeout: float = 30.0,
    ):
        super().__init__()

        self.action_mode = action_mode
        self.render_mode = render_mode
        self.max_shots = max_shots
        self.shot_timeout_ms = shot_timeout_ms
        self.connect_timeout = connect_timeout

        # Bridge client (connects lazily on reset)
        self._client: PeggleBridgeClient | None = None

        # Game tracking
        self._shot_count = 0
        self._prev_orange = 0
        self._prev_total = 0
        self._pegs: list[PegInfo] = []

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
                low=np.array([0, 0, -1], dtype=np.float32),
                high=np.array([1, 1, 1], dtype=np.float32),
            ),
            "pegs": spaces.Box(
                low=-1.0, high=2.0,
                shape=(MAX_PEGS, 4),
                dtype=np.float32,
            ),
        })

    def _ensure_connected(self) -> PeggleBridgeClient:
        if self._client is None or not self._client.connected:
            self._client = PeggleBridgeClient()
            self._client.connect(timeout=self.connect_timeout)
        return self._client

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build observation from bridge data."""
        client = self._ensure_connected()

        # Get pegs
        self._pegs = client.get_pegs()

        # Count stats
        active_pegs = [p for p in self._pegs if not p.hit]
        orange_active = sum(1 for p in active_pegs if p.peg_type == 1)
        total_active = len(active_pegs)

        # Get gun angle
        state_info = client.get_state()

        # Global observation
        max_orange = max(1, sum(1 for p in self._pegs if p.peg_type == 1))
        max_total = max(1, len(self._pegs))

        global_obs = np.array([
            orange_active / max_orange,
            total_active / max_total,
            state_info.gun_angle / ANGLE_MAX,
        ], dtype=np.float32)

        # Peg observations
        peg_obs = np.zeros((MAX_PEGS, 4), dtype=np.float32)
        idx = 0
        for p in self._pegs:
            if idx >= MAX_PEGS:
                break
            if p.hit:
                continue
            peg_obs[idx] = [
                p.x / BOARD_WIDTH,
                p.y / BOARD_HEIGHT,
                p.peg_type / 3.0,  # Normalize type
                1.0,  # Active
            ]
            idx += 1

        return {"global": global_obs, "pegs": peg_obs}

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        client = self._ensure_connected()

        # Reset the level in the real game
        client.reset_level()

        # Wait for the game to reach PreShot state
        time.sleep(0.5)
        try:
            client.wait_state(GameState.PRE_SHOT, timeout_ms=10000)
        except PeggleBridgeError:
            # May already be in PreShot
            pass

        self._shot_count = 0

        # Get initial peg count for reward tracking
        self._pegs = client.get_pegs()
        self._prev_orange = sum(1 for p in self._pegs
                                if p.peg_type == 1 and not p.hit)
        self._prev_total = sum(1 for p in self._pegs if not p.hit)

        obs = self._get_obs()
        info = {
            "shot_count": 0,
            "orange_remaining": self._prev_orange,
            "total_remaining": self._prev_total,
        }
        return obs, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Execute one shot in the real game."""
        client = self._ensure_connected()

        # Decode action to angle
        if self.action_mode == "continuous":
            angle = float(np.clip(action, ANGLE_MIN, ANGLE_MAX))
            if isinstance(action, np.ndarray):
                angle = float(action.item())
        else:
            angle = _angle_from_discrete(int(action))

        # Record pre-shot state
        prev_orange = self._prev_orange

        # Execute the shot
        state_after, shot_info = client.shoot_and_wait(
            angle, timeout_ms=self.shot_timeout_ms
        )
        self._shot_count += 1

        # Update peg info
        self._pegs = client.get_pegs()
        cur_orange = sum(1 for p in self._pegs
                         if p.peg_type == 1 and not p.hit)
        cur_total = sum(1 for p in self._pegs if not p.hit)

        # Compute reward
        orange_cleared = prev_orange - cur_orange
        pegs_cleared = self._prev_total - cur_total
        reward = self._compute_reward(
            orange_cleared, pegs_cleared, state_after
        )

        # Update tracking
        self._prev_orange = cur_orange
        self._prev_total = cur_total

        # Check termination
        terminated = state_after in (
            GameState.LEVEL_DONE,
            GameState.INIT_LEVEL,  # Game might auto-advance
        ) or cur_orange == 0
        truncated = self._shot_count >= self.max_shots

        obs = self._get_obs()
        info = {
            "shot_count": self._shot_count,
            "angle": angle,
            "orange_remaining": cur_orange,
            "total_remaining": cur_total,
            "orange_cleared": orange_cleared,
            "pegs_cleared": pegs_cleared,
            "game_state": state_after.name,
            "shot_info": {
                "pegs_hit_this_shot": shot_info.pegs_hit_this_shot,
                "total_pegs_hit": shot_info.total_pegs_hit,
                "orange_hit": shot_info.orange_hit,
            },
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        orange_cleared: int,
        pegs_cleared: int,
        state: GameState,
    ) -> float:
        """Compute reward after a shot."""
        reward = -0.5  # Shot cost

        # Orange pegs cleared
        reward += orange_cleared * 10.0

        # Total pegs cleared
        reward += pegs_cleared * 0.2

        # Long shot bonus
        if pegs_cleared >= 10:
            reward += 5.0

        # Level complete
        if state == GameState.LEVEL_DONE or self._prev_orange == 0:
            reward += 50.0

        return reward

    def render(self):
        # The real game is its own renderer
        pass

    def close(self):
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_og_envs():
    """Register OG Peggle environments with Gymnasium."""
    gym.register(
        id="PeggleOG-v0",
        entry_point="peggle_rl.og.env:PeggleOGEnv",
        kwargs={"action_mode": "continuous"},
    )
    gym.register(
        id="PeggleOGDiscrete-v0",
        entry_point="peggle_rl.og.env:PeggleOGEnv",
        kwargs={"action_mode": "discrete"},
    )
