# peggle-rl

Reinforcement learning for **Peggle Deluxe** -- featuring a Python physics simulator with all 55 adventure levels and a bridge to the original game via the [Haggle](https://github.com/PeggleCommunity/haggle) modding SDK.

## Features

- **Gymnasium environments** for both the simulator and the OG game
- **All 55 adventure levels** fully parsed from the original `.dat` binary format
- **Continuous and discrete action spaces** -- `Box(-97, 97)` or `Discrete(389)` angle bins
- **Physics engine** matching Peggle Deluxe (gravity, ball-peg collisions, bucket, walls)
- **10 character powers** modeled (SuperGuide, Multiball, SpaceBlast, Fireball, etc.)
- **Haggle mod bridge** -- C++ DLL that lets Python control the real game via named pipes
- **SB3 agent wrappers** for PPO, SAC, TD3, and DQN with tuned hyperparameters
- **Pre-built configs** for training and evaluation

## Quick Start

```bash
# Install (simulator only)
pip install -e ".[train,render]"

# Train a PPO agent on all levels
python scripts/train.py --config configs/ppo_continuous.yaml

# Evaluate
python scripts/eval.py --model checkpoints/ppo_continuous_*/final_model.zip --episodes 100

# Play interactively
python scripts/play.py --level spiral

# Compare against baselines
python scripts/eval.py --baseline random --episodes 100
python scripts/eval.py --baseline heuristic --episodes 100
```

## Environments

| Environment ID | Backend | Action Space | Description |
|---|---|---|---|
| `PeggleSim-v0` | Simulator | Continuous `Box(-97, 97)` | Fast training (vector obs) |
| `PeggleSimDiscrete-v0` | Simulator | Discrete `Discrete(389)` | Discretized angles |
| `PeggleOG-v0` | Real game | Continuous `Box(-97, 97)` | OG Peggle via Haggle bridge |
| `PeggleOGDiscrete-v0` | Real game | Discrete `Discrete(389)` | OG Peggle, discrete |

### Observation Space

```python
{
    "global": Box(5,),   # [balls_remaining, orange_remaining, total_remaining, bucket_x, score]
    "pegs": Box(200, 5), # [x, y, radius, is_orange, active] per peg
}
```

### Simulator Usage

```python
import gymnasium as gym
from peggle_rl.sim.env import register_envs

register_envs()
env = gym.make("PeggleSim-v0", level_name="spiral", render_mode="human")
obs, info = env.reset()

for _ in range(20):
    angle = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(angle)
    env.render()
    if terminated or truncated:
        break

env.close()
```

## OG Game Bridge (Haggle Mod)

To play against the real Peggle Deluxe:

1. **Install** Peggle Deluxe (the CD/retail version)
2. **Install** [Haggle Mod Loader](https://github.com/PeggleCommunity/haggle-mod-loader) in the game directory
3. **Download** `peggle-rl-bridge.dll` and `haggle-sdk.dll` from [Releases](../../releases) (or build from source)
4. **Place** both DLLs in the game's `mods/` folder
5. **Launch** via `Haggle.exe`
6. **Install** the Python client: `pip install -e ".[og]"`

```python
from peggle_rl.og import PeggleBridgeClient

with PeggleBridgeClient() as client:
    assert client.ping()
    state = client.get_state()
    client.set_angle(15.0)
    client.shoot()
```

### Building the Mod DLL

The release process is fully automated via two GitHub Actions workflows:

1. **`bump-tag.yml`** – Triggers on every push to `main`/`master` that changes `haggle_mod/`.
   Automatically bumps the patch version tag (e.g. `v1.0.0` → `v1.0.1`) and pushes it.
2. **`build-haggle-mod.yml`** – Triggers on every `v*` tag push.
   Builds the DLLs and publishes them as a GitHub Release.

To build locally:

```
cd haggle_mod
set HAGGLE_SDK_PATH=path\to\haggle
premake5 vs2022 --file=premake5.lua
msbuild build\peggle-rl-bridge.sln /p:Configuration=Release /p:Platform=x86
```

## Level Data

All 55 adventure levels are parsed from Peggle Deluxe's binary `.dat` format. The parser (`peggle_rl/levels/parser.py`) handles circles, bricks, rods, polygons, teleports, emitters, and movement data.

### Level Extraction

```python
from peggle_rl.levels.catalog import get_level_data, set_levels_dir

# Point to your extracted .dat files
set_levels_dir(r"C:\Users\You\Documents\peggle_extracted\levels")

level = get_level_data("spiral")
print(f"Circles: {len(level.circles)}, Bricks: {len(level.bricks)}")
```

### Stages

| # | Character | Power | Levels |
|---|---|---|---|
| 1 | Bjorn Unicorn | SuperGuide | level1-5 |
| 2 | Jimmy Lightning | Multiball | plinko, ram, threevalleys, spiral, owl |
| 3 | Kat Tut | Pyramid | totempole, infinity, eye, fourdoors, fever |
| 4 | Splork | SpaceBlast | theamoeban, 3d, bug, spaceballs, sunflower |
| 5 | Claude | Flippers | threehills, toggle, sine, infinitejest, sideaction |
| 6 | Renfield | SpookyBall | waves, spiderweb, blockers, baseball, vermin |
| 7 | Tula | FlowerPower | windmills, tulips, tubing, car, sunny |
| 8 | Warren | LuckySpin | monkey, pinball, cyclops, dice, cards |
| 9 | Cinderbottom | Fireball | paperclips, gateway, blocks, hollywoodcircles, bunches |
| 10 | Master Hu | ZenBall | spincycle, vortex, holes, yinyang, doublehelix |
| 11 | Bjorn (Master) | SuperGuide | whirlpool, crisscross, blindfold, aim, dna |

## Project Structure

```
peggle-rl/
  peggle_rl/
    levels/
      parser.py          # Binary .dat level parser (1200+ lines)
      catalog.py         # Level ordering, stage/character mappings
    sim/
      physics.py         # Physics engine (Vec2, Ball, Bucket, collisions)
      engine.py          # Game engine (PeggleGame, scoring, powers)
      renderer.py        # Pygame renderer
      env.py             # Gymnasium sim environments
    og/
      pipe_client.py     # Named pipe client for Haggle bridge
      env.py             # Gymnasium OG game environment
    agents/
      agents.py          # SB3 wrappers (PPO, SAC, TD3, DQN) + baselines
  haggle_mod/
    src/
      main.cpp           # C++ Haggle mod DLL (named pipe server)
    premake5.lua         # Build script
  scripts/
    train.py             # Training script
    eval.py              # Evaluation script
    play.py              # Interactive play / watch agent
  configs/
    ppo_continuous.yaml  # PPO continuous config
    sac_continuous.yaml  # SAC continuous config
    dqn_discrete.yaml    # DQN discrete config
    ...
  .github/
    workflows/
      bump-tag.yml          # CI: auto-bump semver tag on haggle_mod changes
      build-haggle-mod.yml  # CI: build DLL + upload to Releases (triggered by tag)
```

## Credits

This project was built with substantial assistance from **Claude** (Anthropic). The binary level parser, physics engine, game engine, Gymnasium environments, Haggle mod bridge, RL agent wrappers, and project infrastructure were all developed collaboratively with Claude.

- **Haggle SDK**: [PeggleCommunity/haggle](https://github.com/PeggleCommunity/haggle) -- the modding SDK that makes the OG game bridge possible
- **PeggleEdit**: [IntelOrca/PeggleEdit](https://github.com/IntelOrca/PeggleEdit) -- C# source used as reference for the binary `.dat` parser
- **Stable-Baselines3**: RL algorithm implementations

## License

MIT
