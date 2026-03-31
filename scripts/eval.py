#!/usr/bin/env python3
"""
Evaluate a trained Peggle RL agent.

Usage:
    python scripts/eval.py --model checkpoints/ppo_continuous_20240101/final_model.zip
    python scripts/eval.py --model checkpoints/best/best_model.zip --render
    python scripts/eval.py --baseline random --episodes 100
    python scripts/eval.py --baseline heuristic --level level1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Peggle RL agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained SB3 model (.zip)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3", "dqn"],
                        help="Algorithm (for loading correct model class)")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["random", "heuristic"],
                        help="Use a baseline agent instead of trained model")
    parser.add_argument("--action-mode", type=str, default="continuous",
                        choices=["continuous", "discrete"])
    parser.add_argument("--env", type=str, default="PeggleSim-v0")
    parser.add_argument("--level", type=str, default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the game (requires pygame)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    if args.model is None and args.baseline is None:
        print("Error: must specify --model or --baseline")
        sys.exit(1)

    action_mode = args.action_mode
    env_id = args.env

    if "Sim" in env_id and action_mode == "discrete" and "Discrete" not in env_id:
        env_id = env_id.replace("-v0", "Discrete-v0")

    # Create environment
    from peggle_rl.agents import make_env, RandomAgent, HeuristicAgent

    render_mode = "human" if args.render else None
    env = make_env(
        env_id=env_id,
        action_mode=action_mode,
        level_name=args.level,
        stage=args.stage,
        randomize_level=(args.level is None and args.stage is None),
        flatten=True,
        seed=args.seed,
        render_mode=render_mode,
    )

    # Load model or baseline
    if args.baseline == "random":
        model = RandomAgent(env)
        agent_name = "Random"
    elif args.baseline == "heuristic":
        model = HeuristicAgent(env)
        agent_name = "Heuristic"
    else:
        from stable_baselines3 import PPO, SAC, TD3, DQN
        algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "dqn": DQN}
        model = algo_cls[args.algo].load(args.model, env=env)
        agent_name = f"{args.algo.upper()} ({args.model})"

    print(f"Evaluating: {agent_name}")
    print(f"Environment: {env_id} ({action_mode})")
    print(f"Level: {args.level or 'random'}")
    print(f"Episodes: {args.episodes}")
    print()

    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    wins = 0
    orange_cleared_total = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if args.render:
                env.render()

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if info.get("level_won", False):
            wins += 1

        orange_remaining = info.get("orange_remaining", 0)
        orange_total = 25  # Default Peggle orange peg count
        orange_cleared_total += (orange_total - orange_remaining)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1:3d}: reward={total_reward:7.1f}, "
                  f"shots={steps:2d}, "
                  f"won={'yes' if info.get('level_won') else 'no':3s}, "
                  f"orange_left={orange_remaining}")

    # Summary
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)

    print(f"\n{'='*50}")
    print(f"Results over {args.episodes} episodes:")
    print(f"  Mean reward:   {rewards.mean():8.1f} +/- {rewards.std():6.1f}")
    print(f"  Mean shots:    {lengths.mean():8.1f} +/- {lengths.std():6.1f}")
    print(f"  Win rate:      {wins/args.episodes*100:8.1f}%")
    print(f"  Orange cleared:{orange_cleared_total/args.episodes:8.1f} / 25 avg")
    print(f"  Min reward:    {rewards.min():8.1f}")
    print(f"  Max reward:    {rewards.max():8.1f}")
    print(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()
