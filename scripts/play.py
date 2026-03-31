#!/usr/bin/env python3
"""
Play Peggle in the simulator with interactive controls or watch an agent.

Usage:
    python scripts/play.py                          # Interactive play
    python scripts/play.py --model checkpoints/best # Watch a trained agent
    python scripts/play.py --level spiral            # Play a specific level
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Play Peggle in the simulator")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (watch mode)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3", "dqn"])
    parser.add_argument("--level", type=str, default="level1",
                        help="Level to play")
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--action-mode", type=str, default="continuous")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between agent shots (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from peggle_rl.agents import make_env

    env = make_env(
        env_id="PeggleSim-v0",
        action_mode=args.action_mode,
        level_name=args.level,
        stage=args.stage,
        flatten=True,
        seed=args.seed,
        render_mode="human",
    )

    if args.model:
        # Watch a trained agent
        from stable_baselines3 import PPO, SAC, TD3, DQN
        algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "dqn": DQN}
        model = algo_cls[args.algo].load(args.model, env=env)

        obs, info = env.reset()
        print(f"Watching {args.algo.upper()} play {args.level}...")

        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
            time.sleep(args.delay)

        print(f"Game over! Reward: {total_reward:.1f}, "
              f"Won: {info.get('level_won', False)}")
    else:
        # Interactive play with keyboard
        import numpy as np
        try:
            import pygame
        except ImportError:
            print("Error: pygame required for interactive play. "
                  "Install with: pip install pygame")
            sys.exit(1)

        obs, info = env.reset()
        print(f"Playing {args.level} interactively!")
        print("Controls: LEFT/RIGHT to aim, SPACE to shoot, R to reset, Q to quit")

        angle = 0.0
        done = False
        total_reward = 0.0

        while not done:
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        angle = 0.0
                        total_reward = 0.0
                        print("Level reset!")
                    elif event.key == pygame.K_SPACE:
                        action = np.array([angle], dtype=np.float32)
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        print(f"Shot at {angle:.1f} deg -> reward={reward:.1f}, "
                              f"total={total_reward:.1f}")
                        if terminated or truncated:
                            print(f"Game over! Won: {info.get('level_won', False)}")
                            print("Press R to restart or Q to quit")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                angle = max(-97, angle - 1.0)
            if keys[pygame.K_RIGHT]:
                angle = min(97, angle + 1.0)

            time.sleep(1/60)

    env.close()


if __name__ == "__main__":
    main()
