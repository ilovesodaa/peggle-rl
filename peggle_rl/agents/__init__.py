"""Peggle RL agents."""

from peggle_rl.agents.agents import (
    FlattenDictObs,
    make_env,
    create_ppo,
    create_sac,
    create_td3,
    create_dqn,
    RandomAgent,
    HeuristicAgent,
)

__all__ = [
    "FlattenDictObs",
    "make_env",
    "create_ppo",
    "create_sac",
    "create_td3",
    "create_dqn",
    "RandomAgent",
    "HeuristicAgent",
]
