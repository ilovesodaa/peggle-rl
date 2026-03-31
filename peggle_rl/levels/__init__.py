"""Level data: parser + level definitions for all 55 adventure levels."""

from peggle_rl.levels.parser import (
    Level,
    CircleEntry,
    BrickEntry,
    RodEntry,
    PolygonEntry,
    TeleportEntry,
    PegInfo,
    PegType,
    BrickType,
    Movement,
    MovementShape,
    parse_level,
    parse_level_file,
    parse_all_levels,
)
from peggle_rl.levels.catalog import (
    LEVEL_ORDER,
    STAGE_NAMES,
    STAGE_CHARACTERS,
    STAGE_POWERS,
    get_level_data,
    get_all_level_data,
)

__all__ = [
    "Level", "CircleEntry", "BrickEntry", "RodEntry", "PolygonEntry",
    "TeleportEntry", "PegInfo", "PegType", "BrickType", "Movement",
    "MovementShape", "parse_level", "parse_level_file", "parse_all_levels",
    "LEVEL_ORDER", "STAGE_NAMES", "STAGE_CHARACTERS", "STAGE_POWERS",
    "get_level_data", "get_all_level_data",
]
