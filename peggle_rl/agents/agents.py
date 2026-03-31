"""
Stable-Baselines3 agent wrappers for Peggle RL.

Provides ready-to-use training functions for:
  - PPO (discrete and continuous)
  - SAC (continuous)
  - DQN (discrete)

All agents use SB3 under the hood. These wrappers handle:
  - Observation flattening (Dict -> flat vector)
  - Hyperparameter defaults tuned for Peggle
  - Logging and checkpointing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Observation wrapper: flatten Dict obs to a single Box
# ---------------------------------------------------------------------------

class FlattenDictObs(gym.ObservationWrapper):
    """
    Flatten a Dict observation space into a single 1D Box.

    Required because SB3 algorithms (SAC, DQN) don't natively
    support Dict observation spaces.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict)

        # Calculate total flat dimension
        total_dim = 0
        self._keys: list[str] = []
        self._shapes: list[tuple[int, ...]] = []
        for key in sorted(env.observation_space.spaces.keys()):
            space = env.observation_space.spaces[key]
            assert isinstance(space, spaces.Box)
            self._keys.append(key)
            self._shapes.append(space.shape)
            total_dim += int(np.prod(space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def observation(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for key in self._keys:
            parts.append(obs[key].flatten())
        return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_env(
    env_id: str = "PeggleSim-v0",
    action_mode: str = "continuous",
    level_name: str | None = None,
    stage: int | None = None,
    randomize_level: bool = True,
    flatten: bool = True,
    **env_kwargs: Any,
) -> gym.Env:
    """
    Create a Peggle environment ready for SB3 training.

    Args:
        env_id: Environment ID (PeggleSim-v0, PeggleSimDiscrete-v0, PeggleOG-v0, etc.)
        action_mode: "continuous" or "discrete"
        level_name: Optional specific level to train on
        stage: Optional stage index (0-10) to train on
        randomize_level: Whether to randomize level on reset
        flatten: Whether to flatten Dict observations
        **env_kwargs: Additional kwargs passed to the environment
    """
    # Register environments
    from peggle_rl.sim.env import register_envs
    register_envs()

    kwargs: dict[str, Any] = {
        "action_mode": action_mode,
        **env_kwargs,
    }

    if "Sim" in env_id:
        if level_name:
            kwargs["level_name"] = level_name
        if stage is not None:
            kwargs["stage"] = stage
        kwargs["randomize_level"] = randomize_level

    env = gym.make(env_id, **kwargs)

    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenDictObs(env)

    return env


def create_ppo(
    env: gym.Env,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    tensorboard_log: str | None = None,
    **kwargs: Any,
):
    """
    Create a PPO agent tuned for Peggle.

    Works with both continuous and discrete action spaces.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for PPO. "
            "Install with: pip install stable-baselines3"
        )

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )


def create_sac(
    env: gym.Env,
    learning_rate: float = 3e-4,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    learning_starts: int = 1000,
    verbose: int = 1,
    tensorboard_log: str | None = None,
    **kwargs: Any,
):
    """
    Create a SAC agent for continuous action Peggle.

    SAC (Soft Actor-Critic) works only with continuous action spaces.
    Best for fine-grained angle control.
    """
    try:
        from stable_baselines3 import SAC
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for SAC. "
            "Install with: pip install stable-baselines3"
        )

    return SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        learning_starts=learning_starts,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )


def create_td3(
    env: gym.Env,
    learning_rate: float = 1e-3,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    learning_starts: int = 1000,
    verbose: int = 1,
    tensorboard_log: str | None = None,
    **kwargs: Any,
):
    """
    Create a TD3 agent for continuous action Peggle.

    TD3 (Twin Delayed DDPG) is another option for continuous control.
    """
    try:
        from stable_baselines3 import TD3
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for TD3. "
            "Install with: pip install stable-baselines3"
        )

    return TD3(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        learning_starts=learning_starts,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )


def create_dqn(
    env: gym.Env,
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.02,
    learning_starts: int = 500,
    verbose: int = 1,
    tensorboard_log: str | None = None,
    **kwargs: Any,
):
    """
    Create a DQN agent for discrete action Peggle.

    DQN works only with discrete action spaces (discretized angle bins).
    """
    try:
        from stable_baselines3 import DQN
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for DQN. "
            "Install with: pip install stable-baselines3"
        )

    return DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        learning_starts=learning_starts,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Random baseline agent
# ---------------------------------------------------------------------------

class RandomAgent:
    """Simple random baseline agent for benchmarking."""

    def __init__(self, env: gym.Env):
        self.action_space = env.action_space

    def predict(self, obs: Any, **kwargs) -> tuple[Any, None]:
        return self.action_space.sample(), None


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """
    Rule-based agent that aims at the densest cluster of orange pegs.

    Useful as a baseline to compare RL agents against.
    """

    def __init__(self, env: gym.Env, noise_std: float = 2.0):
        self.action_space = env.action_space
        self.noise_std = noise_std
        self._is_discrete = isinstance(env.action_space, spaces.Discrete)

    def predict(self, obs: Any, **kwargs) -> tuple[Any, None]:
        import numpy as np

        # obs is flattened: first 5 values are global, rest are pegs
        if isinstance(obs, dict):
            global_obs = obs["global"]
            peg_obs = obs["pegs"]
        else:
            global_obs = obs[:5]
            peg_obs = obs[5:].reshape(-1, 5)

        # Find orange pegs (type == 1.0 in peg obs)
        orange_mask = peg_obs[:, 3] > 0.5  # type index
        active_mask = peg_obs[:, 4] > 0.5  # active flag
        target_mask = orange_mask & active_mask

        if not target_mask.any():
            # No orange pegs visible - aim at any active peg
            target_mask = active_mask

        if not target_mask.any():
            # No pegs at all - random
            return self.action_space.sample(), None

        # Get target peg positions (normalized)
        targets = peg_obs[target_mask]
        # Average position of orange pegs
        avg_x = targets[:, 0].mean()  # normalized [0, 1]

        # Convert x position to angle
        # x=0 -> left wall -> negative angle, x=1 -> right wall -> positive
        angle = (avg_x - 0.5) * 2 * 97.0  # Scale to [-97, 97]

        # Add noise for exploration
        angle += np.random.normal(0, self.noise_std)
        angle = np.clip(angle, -97.0, 97.0)

        if self._is_discrete:
            action = int((angle - (-97.0)) / 0.5)
            action = max(0, min(action, self.action_space.n - 1))
            return action, None
        else:
            return np.array([angle], dtype=np.float32), None
