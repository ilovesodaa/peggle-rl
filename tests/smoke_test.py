"""Quick smoke test for the Peggle RL simulator environment."""
import sys
sys.path.insert(0, r"C:\Users\Valou\Documents\peggle-rl")

from peggle_rl.levels.catalog import set_levels_dir
set_levels_dir(r"C:\Users\Valou\Documents\peggle_extracted\levels")

from peggle_rl.sim.env import register_envs
import gymnasium as gym
import numpy as np

register_envs()

# Test continuous env
env = gym.make("PeggleSim-v0", level_name="level1", seed=42)
obs, info = env.reset(seed=42)
print("Obs keys:", list(obs.keys()))
print("Global shape:", obs["global"].shape)
print("Pegs shape:", obs["pegs"].shape)

action = np.array([0.0], dtype=np.float32)
obs, reward, term, trunc, info = env.step(action)
print("Shot 1: reward=%.2f term=%s pegs_hit=%s" % (reward, term, info.get("pegs_hit_this_shot")))

# Run a full episode
obs, info = env.reset(seed=123)
total_reward = 0.0
for i in range(20):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if term or trunc:
        break

print("Episode: %d shots, reward=%.1f, won=%s" % (i + 1, total_reward, info.get("level_won", False)))
env.close()

# Test discrete env
env2 = gym.make("PeggleSimDiscrete-v0", level_name="spiral", seed=99)
obs, info = env2.reset(seed=99)
action = env2.action_space.sample()
obs, reward, term, trunc, info = env2.step(action)
print("Discrete shot: reward=%.2f" % reward)
env2.close()

# Test FlattenDictObs wrapper
from peggle_rl.agents import make_env
env3 = make_env(env_id="PeggleSim-v0", level_name="level1", flatten=True, seed=42)
obs, info = env3.reset(seed=42)
print("Flat obs shape:", obs.shape, "dtype:", obs.dtype)
env3.close()

print("\nALL SMOKE TESTS PASSED")
