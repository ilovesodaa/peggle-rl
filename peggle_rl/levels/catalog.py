"""
Level catalog for Peggle Deluxe adventure mode.

All 55 levels organized by stage, with character/power mappings.
Provides functions to load parsed level data from .dat files.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from peggle_rl.levels.parser import Level, parse_level_file, parse_all_levels


# ---------------------------------------------------------------------------
# Stage / level ordering (from stages.cfg)
# ---------------------------------------------------------------------------

# 11 stages, 5 levels each = 55 levels total
STAGE_LEVELS: list[list[str]] = [
    # Stage 1: Bjorn (SuperGuide)
    ["level1", "level2", "level3", "level4", "level5"],
    # Stage 2: Jimmy Lightning (Multiball)
    ["plinko", "ram", "threevalleys", "spiral", "owl"],
    # Stage 3: Kat Tut (Pyramid)
    ["totempole", "infinity", "eye", "fourdoors", "fever"],
    # Stage 4: Splork (SpaceBlast)
    ["theamoeban", "3d", "bug", "spaceballs", "sunflower"],
    # Stage 5: Claude (Flippers)
    ["threehills", "toggle", "sine", "infinitejest", "sideaction"],
    # Stage 6: Renfield (SpookyBall)
    ["waves", "spiderweb", "blockers", "baseball", "vermin"],
    # Stage 7: Tula (FlowerPower)
    ["windmills", "tulips", "tubing", "car", "sunny"],
    # Stage 8: Warren (LuckySpin)
    ["monkey", "pinball", "cyclops", "dice", "cards"],
    # Stage 9: Lord Cinderbottom (Fireball)
    ["paperclips", "gateway", "blocks", "hollywoodcircles", "bunches"],
    # Stage 10: Master Hu (ZenBall)
    ["spincycle", "vortex", "holes", "yinyang", "doublehelix"],
    # Stage 11: Bjorn (Master Levels)
    ["whirlpool", "crisscross", "blindfold", "aim", "dna"],
]

STAGE_CHARACTERS: list[str] = [
    "Bjorn Unicorn",
    "Jimmy Lightning",
    "Kat Tut",
    "Splork",
    "Claude",
    "Renfield",
    "Tula",
    "Warren",
    "Lord Cinderbottom",
    "Master Hu",
    "Bjorn Unicorn",  # Master levels
]

STAGE_POWERS: list[str] = [
    "SuperGuide",
    "Multiball",
    "Pyramid",
    "SpaceBlast",
    "Flippers",
    "SpookyBall",
    "FlowerPower",
    "LuckySpin",
    "Fireball",
    "ZenBall",
    "SuperGuide",  # Master levels
]

STAGE_NAMES: list[str] = [
    "Peggle Basics",
    "Jimmy Lightning",
    "Kat Tut",
    "Splork",
    "Claude",
    "Renfield",
    "Tula",
    "Warren",
    "Lord Cinderbottom",
    "Master Hu",
    "Master Levels",
]

# Flat list of all 55 level names in adventure order
LEVEL_ORDER: list[str] = [
    name for stage in STAGE_LEVELS for name in stage
]

# Map level name -> (stage_index, level_in_stage_index)
LEVEL_INDEX: dict[str, tuple[int, int]] = {
    name: (si, li)
    for si, stage in enumerate(STAGE_LEVELS)
    for li, name in enumerate(stage)
}


# ---------------------------------------------------------------------------
# Level data loading
# ---------------------------------------------------------------------------

# Default path to extracted level .dat files
_DEFAULT_LEVELS_DIR: Path | None = None


def set_levels_dir(path: str | Path) -> None:
    """Set the directory containing extracted .dat level files."""
    global _DEFAULT_LEVELS_DIR
    _DEFAULT_LEVELS_DIR = Path(path)


def _find_levels_dir() -> Path:
    """Try to find the levels directory automatically."""
    if _DEFAULT_LEVELS_DIR is not None:
        return _DEFAULT_LEVELS_DIR

    # Check common locations
    candidates = [
        Path.home() / "Documents" / "peggle_extracted" / "levels",
        Path.home() / "peggle_extracted" / "levels",
        Path(__file__).parent.parent.parent / "data" / "levels",
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.dat")):
            return p

    raise FileNotFoundError(
        "Cannot find extracted Peggle level .dat files. "
        "Call set_levels_dir() or place them in ~/Documents/peggle_extracted/levels/"
    )


@lru_cache(maxsize=64)
def get_level_data(name: str) -> Level:
    """Load and parse a single level by name (cached)."""
    name = name.lower()
    levels_dir = _find_levels_dir()
    dat_path = levels_dir / f"{name}.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"Level file not found: {dat_path}")
    return parse_level_file(dat_path)


def get_all_level_data() -> dict[str, Level]:
    """Load all 55 adventure level files."""
    levels_dir = _find_levels_dir()
    return parse_all_levels(levels_dir)


def get_stage_levels(stage: int) -> list[Level]:
    """Load all 5 levels for a given stage (0-indexed)."""
    return [get_level_data(name) for name in STAGE_LEVELS[stage]]


def level_info(name: str) -> dict:
    """Get metadata for a level (stage, character, power)."""
    name = name.lower()
    si, li = LEVEL_INDEX[name]
    return {
        "name": name,
        "stage": si + 1,
        "stage_name": STAGE_NAMES[si],
        "level_in_stage": li + 1,
        "character": STAGE_CHARACTERS[si],
        "power": STAGE_POWERS[si],
        "adventure_index": si * 5 + li + 1,  # 1-55
    }
