#!/usr/bin/env python3
"""
Train a Peggle RL agent.

Usage:
    python scripts/train.py --config configs/ppo_continuous.yaml
    python scripts/train.py --algo ppo --action-mode continuous --total-timesteps 500000
    python scripts/train.py --algo sac --level level1 --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a Peggle RL agent")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3", "dqn"],
                        help="RL algorithm")
    parser.add_argument("--action-mode", type=str, default="continuous",
                        choices=["continuous", "discrete"],
                        help="Action space mode")
    parser.add_argument("--env", type=str, default="PeggleSim-v0",
                        help="Environment ID")
    parser.add_argument("--level", type=str, default=None,
                        help="Train on a specific level")
    parser.add_argument("--stage", type=int, default=None,
                        help="Train on a specific stage (0-10)")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Tensorboard log directory")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Evaluate every N timesteps")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    # Load config if provided (overrides CLI args)
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    algo = cfg.get("algo", args.algo)
    action_mode = cfg.get("action_mode", args.action_mode)
    env_id = cfg.get("env", args.env)
    level = cfg.get("level", args.level)
    stage = cfg.get("stage", args.stage)
    total_timesteps = cfg.get("total_timesteps", args.total_timesteps)
    save_dir = cfg.get("save_dir", args.save_dir)
    log_dir = cfg.get("log_dir", args.log_dir)
    save_freq = cfg.get("save_freq", args.save_freq)
    eval_freq = cfg.get("eval_freq", args.eval_freq)
    eval_episodes = cfg.get("eval_episodes", args.eval_episodes)
    seed = cfg.get("seed", args.seed)
    verbose = cfg.get("verbose", args.verbose)
    hyperparams = cfg.get("hyperparams", {})

    # Validate
    if algo in ("sac", "td3") and action_mode == "discrete":
        print(f"Error: {algo.upper()} requires continuous action mode")
        sys.exit(1)
    if algo == "dqn" and action_mode == "continuous":
        print("Error: DQN requires discrete action mode")
        sys.exit(1)

    # Adjust env ID for action mode
    if "Sim" in env_id and action_mode == "discrete" and "Discrete" not in env_id:
        env_id = env_id.replace("-v0", "Discrete-v0") if "Discrete" not in env_id else env_id

    # Create environment
    from peggle_rl.agents import make_env
    env = make_env(
        env_id=env_id,
        action_mode=action_mode,
        level_name=level,
        stage=stage,
        randomize_level=(level is None and stage is None),
        flatten=True,
        seed=seed,
    )

    # Create run name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_{action_mode}_{ts}"
    if level:
        run_name = f"{algo}_{level}_{ts}"
    elif stage is not None:
        run_name = f"{algo}_stage{stage}_{ts}"

    tb_log = os.path.join(log_dir, run_name)

    # Create agent
    from peggle_rl.agents import create_ppo, create_sac, create_td3, create_dqn

    agent_factory = {
        "ppo": create_ppo,
        "sac": create_sac,
        "td3": create_td3,
        "dqn": create_dqn,
    }

    model = agent_factory[algo](
        env,
        verbose=verbose,
        tensorboard_log=tb_log,
        **hyperparams,
    )

    print(f"Training {algo.upper()} on {env_id} ({action_mode})")
    print(f"  Level: {level or 'all'}")
    print(f"  Stage: {stage if stage is not None else 'all'}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Save dir: {save_dir}")
    print(f"  Tensorboard: {tb_log}")
    print()

    # Setup callbacks
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )

    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(save_dir, run_name),
            name_prefix="peggle",
        ),
    ]

    if eval_freq > 0:
        eval_env = make_env(
            env_id=env_id,
            action_mode=action_mode,
            level_name=level,
            stage=stage,
            randomize_level=(level is None and stage is None),
            flatten=True,
            seed=(seed + 1000) if seed else None,
        )
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, run_name, "best"),
                log_path=os.path.join(log_dir, run_name, "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                deterministic=True,
            )
        )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_dir, run_name, "final_model")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
