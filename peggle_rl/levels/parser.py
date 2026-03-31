"""
Peggle Deluxe .dat level file parser.

Faithfully ported from the PeggleEdit C# source (IntelOrca/PeggleEdit).
Every field ordering, flag bit, and data type matches the original C# code.

All coordinates are in Peggle's native space:
  - Board width: 800 pixels (logical)
  - Board height: ~580 pixels (playable area, full frame is 600)
  - Origin (0,0) at top-left
  - Ball launcher at top center (~400, 0)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


# ---------------------------------------------------------------------------
# Enums (matching C# LevelEntryTypes and other enums)
# ---------------------------------------------------------------------------

class EntryType(IntEnum):
    ROD = 2
    POLYGON = 3
    CIRCLE = 5
    BRICK = 6
    TELEPORT = 8
    EMITTER = 9


class PegType(IntEnum):
    NONE = 0
    NORMAL = 1      # Blue
    GOAL = 2        # Orange
    SCORE = 3       # Purple (bonus score)
    POWERUP = 4     # Green


class MovementShape(IntEnum):
    NO_MOVEMENT = 0
    VERTICAL_CYCLE = 1
    HORIZONTAL_CYCLE = 2
    CIRCLE = 3
    HORIZONTAL_INFINITY = 4
    VERTICAL_INFINITY = 5
    HORIZONTAL_ARC = 6
    VERTICAL_ARC = 7
    ROTATE = 8
    ROTATE_BACK_AND_FORTH = 9
    UNUSED = 10
    VERTICAL_WRAP = 11
    HORIZONTAL_WRAP = 12
    ROTATE_AROUND_CIRCLE = 13
    RETRACE_CIRCLE = 14
    WEIRD_SHAPE = 15


class BrickType(IntEnum):
    NORMAL_CURVED = 0
    OUTER_CURVED = 1
    OUTER_CURVED_90 = 2
    INNER_CURVED = 3
    INNER_CURVED_90 = 4
    STRAIGHT = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PegInfo:
    """Peg type info (from PegInfo.cs ReadData)."""
    peg_type: PegType = PegType.NONE
    variable: bool = False      # Can be orange (randomly assigned)
    crumble: bool = False       # Quick-disappear peg


@dataclass
class Movement:
    """Movement path data (from Movement.cs ReadData)."""
    shape: MovementShape = MovementShape.NO_MOVEMENT
    reversed: bool = False
    anchor_x: float = 0.0
    anchor_y: float = 0.0
    time_period: int = 0        # deciseconds (100 = 1 second)
    offset: int = 0             # int16
    radius1: int = 0            # int16
    radius2: int = 0            # int16
    start_phase: float = 0.0
    move_rotation_deg: float = 0.0  # stored as radians, converted to degrees
    pause1: int = 0             # int16, deciseconds
    pause2: int = 0             # int16, deciseconds
    phase1: int = 0             # byte (percentage 0-100)
    phase2: int = 0             # byte (percentage 0-100)
    post_delay_phase: float = 0.0
    max_angle: float = 0.0
    rotation_deg: float = 0.0   # fA[14], stored as radians, converted to degrees
    # Sub-movement (fA[12])
    sub_offset_x: float = 0.0
    sub_offset_y: float = 0.0
    sub_movement: Movement | None = None
    # Object position (fA[13])
    object_x: float = 0.0
    object_y: float = 0.0


@dataclass
class CircleEntry:
    """A circular peg (from Circle.cs ReadData)."""
    x: float = 0.0
    y: float = 0.0
    radius: float = 10.0
    # Generic fields
    peg_info: PegInfo | None = None
    movement: Movement | None = None
    rolly: float | None = None
    bouncy: float | None = None
    can_move: bool = True
    visible: bool = True


@dataclass
class BrickEntry:
    """A brick / wall segment (from Brick.cs ReadData)."""
    x: float = 0.0
    y: float = 0.0
    length: float = 50.0
    angle: float = 0.0          # radians
    width: float = 20.0
    brick_type: BrickType = BrickType.STRAIGHT
    curve_points: int = 0
    left_angle: float = 0.0
    right_angle: float = 0.0
    sector_angle: float = 0.0
    texture_flip: bool = False
    # Generic fields
    peg_info: PegInfo | None = None
    movement: Movement | None = None
    rolly: float | None = None
    bouncy: float | None = None
    can_move: bool = True
    visible: bool = True


@dataclass
class RodEntry:
    """A rod / line segment (from Rod.cs ReadData)."""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    # Generic fields
    peg_info: PegInfo | None = None
    movement: Movement | None = None
    rolly: float | None = None
    bouncy: float | None = None


@dataclass
class PolygonEntry:
    """A polygon shape (from Polygon.cs ReadData)."""
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0
    normal_dir: int = 0
    points: list[tuple[float, float]] = field(default_factory=list)
    # Generic fields
    peg_info: PegInfo | None = None
    movement: Movement | None = None
    rolly: float | None = None
    bouncy: float | None = None


@dataclass
class TeleportEntry:
    """A teleport region (from Teleport.cs ReadData)."""
    x: float = 0.0
    y: float = 0.0
    width: int = 0              # int32
    height: int = 0             # int32
    # Generic fields
    peg_info: PegInfo | None = None
    movement: Movement | None = None
    rolly: float | None = None
    bouncy: float | None = None


@dataclass
class Level:
    """A parsed Peggle level."""
    version: int = 0
    circles: list[CircleEntry] = field(default_factory=list)
    bricks: list[BrickEntry] = field(default_factory=list)
    rods: list[RodEntry] = field(default_factory=list)
    polygons: list[PolygonEntry] = field(default_factory=list)
    teleports: list[TeleportEntry] = field(default_factory=list)

    @property
    def pegs(self) -> list[CircleEntry]:
        """All circle pegs that have peg info."""
        return [c for c in self.circles if c.peg_info is not None]

    @property
    def orange_pegs(self) -> list[CircleEntry]:
        return [c for c in self.circles
                if c.peg_info and c.peg_info.peg_type == PegType.GOAL]

    @property
    def blue_pegs(self) -> list[CircleEntry]:
        return [c for c in self.circles
                if c.peg_info and c.peg_info.peg_type == PegType.NORMAL]

    @property
    def green_pegs(self) -> list[CircleEntry]:
        return [c for c in self.circles
                if c.peg_info and c.peg_info.peg_type == PegType.POWERUP]

    @property
    def purple_pegs(self) -> list[CircleEntry]:
        return [c for c in self.circles
                if c.peg_info and c.peg_info.peg_type == PegType.SCORE]

    @property
    def variable_pegs(self) -> list[CircleEntry]:
        """Circle pegs that can be randomly assigned as orange."""
        return [c for c in self.circles
                if c.peg_info and c.peg_info.variable]

    @property
    def variable_bricks(self) -> list[BrickEntry]:
        """Bricks that can be randomly assigned as orange."""
        return [b for b in self.bricks
                if b.peg_info and b.peg_info.variable]

    @property
    def total_variable(self) -> int:
        """Total variable entries (circles + bricks)."""
        return len(self.variable_pegs) + len(self.variable_bricks)

    @property
    def orange_bricks(self) -> list[BrickEntry]:
        return [b for b in self.bricks
                if b.peg_info and b.peg_info.peg_type == PegType.GOAL]

    @property
    def total_orange(self) -> int:
        return len(self.orange_pegs) + len(self.orange_bricks)

    @property
    def all_entries(self):
        return (self.circles + self.bricks + self.rods
                + self.polygons + self.teleports)


# ---------------------------------------------------------------------------
# Binary reader helper
# ---------------------------------------------------------------------------

class BinaryReader:
    """Little-endian binary stream reader matching C# BinaryReader semantics."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def read_bytes(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise EOFError(
                f"Cannot read {n} bytes at offset 0x{self._pos:X}, "
                f"only {self.remaining} remaining")
        result = self._data[self._pos:self._pos + n]
        self._pos += n
        return result

    def read_byte(self) -> int:
        return self.read_bytes(1)[0]

    def read_sbyte(self) -> int:
        return struct.unpack_from('<b', self.read_bytes(1))[0]

    def read_int16(self) -> int:
        return struct.unpack_from('<h', self.read_bytes(2))[0]

    def read_uint16(self) -> int:
        return struct.unpack_from('<H', self.read_bytes(2))[0]

    def read_int32(self) -> int:
        return struct.unpack_from('<i', self.read_bytes(4))[0]

    def read_uint32(self) -> int:
        return struct.unpack_from('<I', self.read_bytes(4))[0]

    def read_single(self) -> float:
        """ReadSingle - IEEE 754 single-precision float."""
        return struct.unpack_from('<f', self.read_bytes(4))[0]

    def read_uint24(self) -> int:
        """Read 3-byte unsigned integer (little-endian)."""
        b = self.read_bytes(3)
        return b[0] | (b[1] << 8) | (b[2] << 16)

    def read_popcap_string(self) -> str:
        """Read a PopCap length-prefixed string (int16 length + UTF-8 bytes)."""
        length = self.read_int16()
        if length <= 0:
            return ""
        return self.read_bytes(length).decode('utf-8', errors='replace')


# ---------------------------------------------------------------------------
# FlagGroup helper (mirrors C# FlagGroup)
# ---------------------------------------------------------------------------

def flag_get(flags: int, bit: int) -> bool:
    """Check if bit `bit` is set in flags (FlagGroup[i] = flags & (1 << i))."""
    return bool(flags & (1 << bit))


# ---------------------------------------------------------------------------
# PegInfo reader (from PegInfo.cs ReadData)
# ---------------------------------------------------------------------------

def _read_peg_info(br: BinaryReader, version: int) -> PegInfo:
    """
    C# PegInfo.ReadData:
        PegType = (PegType)br.ReadByte()
        fA = FlagGroup(br.ReadByte())
        variable = fA[1]          // bit 1 = 0x02 (CanBeOrange)
        if fA[2]: ReadInt32()     // bit 2 = 0x04
        crumble = fA[3]           // bit 3 = 0x08 (QuickDisappear)
        if fA[4]: ReadInt32()     // bit 4 = 0x10
        if fA[5]: ReadByte()      // bit 5 = 0x20
        if fA[7]: ReadByte()      // bit 7 = 0x80
    """
    info = PegInfo()
    info.peg_type = PegType(br.read_byte())

    flags = br.read_byte()

    # bit 1: variable (CanBeOrange) - flag only, no data
    info.variable = flag_get(flags, 1)

    # bit 2: unknown int32
    if flag_get(flags, 2):
        br.read_int32()

    # bit 3: crumble (QuickDisappear) - flag only, no data
    info.crumble = flag_get(flags, 3)

    # bit 4: unknown int32
    if flag_get(flags, 4):
        br.read_int32()

    # bit 5: unknown byte
    if flag_get(flags, 5):
        br.read_byte()

    # bit 7: unknown byte
    if flag_get(flags, 7):
        br.read_byte()

    return info


# ---------------------------------------------------------------------------
# Movement reader (from Movement.cs + MovementLink.cs ReadData)
# ---------------------------------------------------------------------------

def _read_movement_link(br: BinaryReader, version: int) -> Movement | None:
    """
    C# MovementLink.ReadData:
        InternalLinkId = br.ReadInt32()
        if InternalLinkId == 1:
            InternalMovement = new Movement()
            InternalMovement.ReadData(br, version)
    """
    link_id = br.read_int32()

    if link_id == 1:
        return _read_movement_data(br, version)
    else:
        # link_id == 0 means no movement
        # link_id > 1 means reference to another movement (by ID)
        # For the simulator we don't resolve cross-references
        return None


def _read_movement_data(br: BinaryReader, version: int) -> Movement:
    """
    C# Movement.ReadData - faithfully ported field by field.
    Note the exact data types: offset/radius/pause are int16, phase are byte.
    """
    mov = Movement()

    # movementShape: sbyte. Negative means reversed.
    shape_byte = br.read_sbyte()
    if shape_byte < 0:
        mov.reversed = True
        mov.shape = MovementShape(abs(shape_byte))
    else:
        mov.shape = MovementShape(shape_byte)

    mov.anchor_x = br.read_single()
    mov.anchor_y = br.read_single()
    mov.time_period = br.read_int16()

    fA = br.read_int16()

    if flag_get(fA, 0):
        mov.offset = br.read_int16()
    if flag_get(fA, 1):
        mov.radius1 = br.read_int16()
    if flag_get(fA, 2):
        mov.start_phase = br.read_single()
    if flag_get(fA, 3):
        mov.move_rotation_deg = math.degrees(br.read_single())
    if flag_get(fA, 4):
        mov.radius2 = br.read_int16()
    if flag_get(fA, 5):
        mov.pause1 = br.read_int16()
    if flag_get(fA, 6):
        mov.pause2 = br.read_int16()
    if flag_get(fA, 7):
        mov.phase1 = br.read_byte()
    if flag_get(fA, 8):
        mov.phase2 = br.read_byte()
    if flag_get(fA, 9):
        mov.post_delay_phase = br.read_single()
    if flag_get(fA, 10):
        mov.max_angle = br.read_single()
    if flag_get(fA, 11):
        br.read_single()  # mUnknown8
    # NOTE: fA[14] is read BEFORE fA[12] and fA[13] in the C# source
    if flag_get(fA, 14):
        mov.rotation_deg = math.degrees(br.read_single())
    if flag_get(fA, 12):
        mov.sub_offset_x = br.read_single()
        mov.sub_offset_y = br.read_single()
        # Nested MovementLink
        mov.sub_movement = _read_movement_link(br, version)
    if flag_get(fA, 13):
        mov.object_x = br.read_single()
        mov.object_y = br.read_single()

    return mov


# ---------------------------------------------------------------------------
# ReadGenericData (from LevelEntry.cs ReadGenericData)
#
# CRITICAL: PegInfo (bit 2) and MovementLink (bit 3) are read LAST,
# after ALL other conditional fields. This is the key ordering bug
# that was in the previous parser.
# ---------------------------------------------------------------------------

@dataclass
class _GenericData:
    """Container for data read from ReadGenericData."""
    flags: int = 0
    rolly: float | None = None
    bouncy: float | None = None
    peg_info: PegInfo | None = None
    movement: Movement | None = None
    can_move: bool = True
    visible: bool = True


def _read_generic_data(br: BinaryReader, version: int) -> _GenericData:
    """
    C# LevelEntry.ReadGenericData - exact port.

    The read order in C# is:
        flags (int32 or uint24)
        if f[0]: rolly = ReadSingle()
        if f[1]: bouncy = ReadSingle()
        // f[2] PegInfo and f[3] MovementLink are NOT read here
        if f[4]: ReadInt32()  // unknown
        // f[5] collision - flag only
        // f[6] visible - flag only
        // f[7] canMove - flag only
        if f[8]: solidColour = ReadInt32()
        if f[9]: outlineColour = ReadInt32()
        if f[10]: imageFilename = ReadPopcapString()
        if f[11]: imageDX = ReadSingle()
        if f[12]: imageDY = ReadSingle()
        if f[13]: imageRotation = ReadSingle()
        // f[14] background - flag only
        // f[15] baseObject - flag only
        if f[16]: ReadInt32()
        if f[17]: ID = ReadPopcapString()
        if f[18]: ReadInt32()
        if f[19]: sound = ReadByte()
        // f[20] ballStopReset - flag only
        if f[21]: logic = ReadPopcapString()
        // f[22] foreground - flag only
        if f[23]: maxBounceVelocity = ReadSingle()
        // f[24] drawSort - flag only
        // f[25] foreground2 - flag only
        if f[26]: subID = ReadInt32()
        if f[27]: flipperFlags = ReadByte()
        // f[28] drawFloat - flag only
        if version >= 0x50 && f[30]: shadow - flag only
        // NOW read PegInfo and MovementLink LAST:
        if f[2]: PegInfo.ReadData()
        if f[3]: MovementLink.ReadData()
    """
    g = _GenericData()

    if version >= 0x0F:
        g.flags = br.read_int32()
    else:
        g.flags = br.read_uint24()

    f = g.flags

    # --- Fields read in order (matching C# exactly) ---

    if flag_get(f, 0):
        g.rolly = br.read_single()
    if flag_get(f, 1):
        g.bouncy = br.read_single()

    # bits 2, 3 are DEFERRED to end

    if flag_get(f, 4):
        br.read_int32()     # unknown

    # bit 5: collision - flag only
    # bit 6: visible flag
    if flag_get(f, 6):
        g.visible = False
    # bit 7: canMove flag
    if flag_get(f, 7):
        g.can_move = False

    if flag_get(f, 8):
        br.read_int32()     # solidColour
    if flag_get(f, 9):
        br.read_int32()     # outlineColour
    if flag_get(f, 10):
        br.read_popcap_string()  # imageFilename
    if flag_get(f, 11):
        br.read_single()    # imageDX
    if flag_get(f, 12):
        br.read_single()    # imageDY
    if flag_get(f, 13):
        br.read_single()    # imageRotation

    # bit 14: background - flag only
    # bit 15: baseObject - flag only

    if flag_get(f, 16):
        br.read_int32()     # unknown
    if flag_get(f, 17):
        br.read_popcap_string()  # ID
    if flag_get(f, 18):
        br.read_int32()     # unknown
    if flag_get(f, 19):
        br.read_byte()      # sound

    # bit 20: ballStopReset - flag only

    if flag_get(f, 21):
        br.read_popcap_string()  # logic

    # bit 22: foreground - flag only

    if flag_get(f, 23):
        br.read_single()    # maxBounceVelocity

    # bit 24: drawSort - flag only
    # bit 25: foreground2 - flag only

    if flag_get(f, 26):
        br.read_int32()     # subID
    if flag_get(f, 27):
        br.read_byte()      # flipperFlags

    # bit 28: drawFloat - flag only

    if version >= 0x50 and flag_get(f, 30):
        pass  # shadow - flag only

    # --- DEFERRED: PegInfo and MovementLink read LAST ---

    if flag_get(f, 2):
        g.peg_info = _read_peg_info(br, version)
    if flag_get(f, 3):
        g.movement = _read_movement_link(br, version)

    return g


# ---------------------------------------------------------------------------
# Circle reader (from Circle.cs ReadData)
# ---------------------------------------------------------------------------

def _read_circle(br: BinaryReader, version: int, g: _GenericData) -> CircleEntry:
    """
    C# Circle.ReadData:
        fA = FlagGroup(br.ReadByte())
        if version >= 0x52: fB = FlagGroup(br.ReadByte())  // Peggle Nights
        if fA[1]: X = ReadSingle(); Y = ReadSingle()       // bit 1 = 0x02
        radius = ReadSingle()   // always
    Note: fA[0] is a flag ("make it bounce") with no associated data read.
    """
    entry = CircleEntry(
        rolly=g.rolly,
        bouncy=g.bouncy,
        peg_info=g.peg_info,
        movement=g.movement,
        can_move=g.can_move,
        visible=g.visible,
    )

    fA = br.read_byte()
    if version >= 0x52:
        _fB = br.read_byte()  # Peggle Nights only

    # bit 1 (0x02): position
    if flag_get(fA, 1):
        entry.x = br.read_single()
        entry.y = br.read_single()

    # radius always present
    entry.radius = br.read_single()

    return entry


# ---------------------------------------------------------------------------
# Brick reader (from Brick.cs ReadData)
# ---------------------------------------------------------------------------

def _read_brick(br: BinaryReader, version: int, g: _GenericData) -> BrickEntry:
    """
    C# Brick.ReadData - exact port with correct flag ordering.

    fA = FlagGroup(ReadByte())
    if version >= 0x23: fB = FlagGroup(ReadByte())

    NOTE: fA field read order is NOT sequential!
    if fA[2]: ReadSingle()       // unknown
    if fA[3]: ReadSingle()       // unknown
    if fA[5]: ReadSingle()       // unknown
    if fA[1]: ReadByte()         // unknown
    if fA[4]: X = ReadSingle(); Y = ReadSingle()

    if fB[0]: ReadByte()
    if fB[1]: ReadInt32()
    if fB[2]: ReadInt16()

    fC = FlagGroup(ReadInt16())
    NOTE: fC field read order is also NOT sequential!
    if fC[8]: ReadSingle()       // unknown
    if fC[9]: ReadSingle()       // unknown
    if fC[2]: type = ReadByte(); if type==5 -> straight
    if fC[3]: curvePoints = ReadByte() + 2
    if fC[5]: leftAngle = ReadSingle()
    if fC[6]: rightAngle = ReadSingle(); ReadSingle()  // extra float!
    if fC[4]: sectorAngle = ReadSingle()
    if fC[7]: width = ReadSingle()
    textureFlip = fC[10]         // flag only
    length = ReadSingle()        // always
    angle = ReadSingle()         // always
    trailer = ReadUInt32()       // always
    """
    entry = BrickEntry(
        rolly=g.rolly,
        bouncy=g.bouncy,
        peg_info=g.peg_info,
        movement=g.movement,
        can_move=g.can_move,
        visible=g.visible,
    )

    fA = br.read_byte()
    fB = 0
    if version >= 0x23:
        fB = br.read_byte()

    # fA conditionals (non-sequential order!)
    if flag_get(fA, 2):
        br.read_single()        # unknown
    if flag_get(fA, 3):
        br.read_single()        # unknown
    if flag_get(fA, 5):
        br.read_single()        # unknown
    if flag_get(fA, 1):
        br.read_byte()          # unknown
    if flag_get(fA, 4):
        entry.x = br.read_single()
        entry.y = br.read_single()

    # fB conditionals
    if flag_get(fB, 0):
        br.read_byte()          # unknown
    if flag_get(fB, 1):
        br.read_int32()         # unknown
    if flag_get(fB, 2):
        br.read_int16()         # unknown

    # fC flags
    fC = br.read_int16()

    # fC conditionals (non-sequential order!)
    if flag_get(fC, 8):
        br.read_single()        # unknown
    if flag_get(fC, 9):
        br.read_single()        # unknown
    if flag_get(fC, 2):
        btype = br.read_byte()
        try:
            entry.brick_type = BrickType(btype)
        except ValueError:
            entry.brick_type = BrickType.STRAIGHT
    if flag_get(fC, 3):
        entry.curve_points = br.read_byte() + 2
    if flag_get(fC, 5):
        entry.left_angle = br.read_single()
    if flag_get(fC, 6):
        entry.right_angle = br.read_single()
        br.read_single()        # extra float (always accompanies right_angle)
    if flag_get(fC, 4):
        entry.sector_angle = br.read_single()
    if flag_get(fC, 7):
        entry.width = br.read_single()

    entry.texture_flip = flag_get(fC, 10)

    # Always present
    entry.length = br.read_single()
    entry.angle = br.read_single()
    _trailer = br.read_uint32()

    return entry


# ---------------------------------------------------------------------------
# Rod reader (from Rod.cs ReadData)
# ---------------------------------------------------------------------------

def _read_rod(br: BinaryReader, version: int, g: _GenericData) -> RodEntry:
    """
    C# Rod.ReadData:
        fA = FlagGroup(ReadByte())
        pointA = (ReadSingle(), ReadSingle())  // always
        pointB = (ReadSingle(), ReadSingle())  // always
        if fA[0]: ReadSingle()  // unknown
        if fA[1]: ReadSingle()  // unknown
    """
    entry = RodEntry(
        rolly=g.rolly,
        bouncy=g.bouncy,
        peg_info=g.peg_info,
        movement=g.movement,
    )

    fA = br.read_byte()

    # Points always present
    entry.x1 = br.read_single()
    entry.y1 = br.read_single()
    entry.x2 = br.read_single()
    entry.y2 = br.read_single()

    # Conditional fields
    if flag_get(fA, 0):
        br.read_single()        # unknown
    if flag_get(fA, 1):
        br.read_single()        # unknown

    return entry


# ---------------------------------------------------------------------------
# Polygon reader (from Polygon.cs ReadData)
# ---------------------------------------------------------------------------

def _read_polygon(br: BinaryReader, version: int, g: _GenericData) -> PolygonEntry:
    """
    C# Polygon.ReadData:
        fA = FlagGroup(ReadByte())
        if version >= 0x23: fB = FlagGroup(ReadByte())

        NOTE: fA field read order is NOT sequential (same pattern as Brick):
        if fA[2]: rotation = ReadSingle()
        if fA[3]: ReadSingle()          // unknown
        if fA[5]: scale = ReadSingle()
        if fA[1]: normalDir = ReadByte()
        if fA[4]: X = ReadSingle(); Y = ReadSingle()

        numPoints = ReadInt32()
        for each point: (ReadSingle(), ReadSingle())

        if fB[0]: ReadByte()            // unknown
        if fB[1]: growType = ReadInt32()
    """
    entry = PolygonEntry(
        rolly=g.rolly,
        bouncy=g.bouncy,
        peg_info=g.peg_info,
        movement=g.movement,
    )

    fA = br.read_byte()
    fB = 0
    if version >= 0x23:
        fB = br.read_byte()

    # fA conditionals (non-sequential!)
    if flag_get(fA, 2):
        entry.rotation = br.read_single()
    if flag_get(fA, 3):
        br.read_single()        # unknown
    if flag_get(fA, 5):
        entry.scale = br.read_single()
    if flag_get(fA, 1):
        entry.normal_dir = br.read_byte()
    if flag_get(fA, 4):
        entry.x = br.read_single()
        entry.y = br.read_single()

    # Points
    num_points = br.read_int32()
    for _ in range(num_points):
        px = br.read_single()
        py = br.read_single()
        entry.points.append((px, py))

    # fB conditionals
    if flag_get(fB, 0):
        br.read_byte()          # unknown
    if flag_get(fB, 1):
        br.read_int32()         # growType

    return entry


# ---------------------------------------------------------------------------
# Teleport reader (from Teleport.cs ReadData)
# ---------------------------------------------------------------------------

def _read_teleport(br: BinaryReader, version: int, g: _GenericData) -> TeleportEntry:
    """
    C# Teleport.ReadData:
        fA = FlagGroup(ReadByte())
        width = ReadInt32()     // NOTE: int32, not float!
        height = ReadInt32()    // NOTE: int32, not float!
        if fA[1]: ReadInt16()
        if fA[3]: ReadInt32()
        if fA[5]: ReadInt32()
        if fA[4]: entry = LevelEntryFactory.CreateLevelEntry(br, version)
        if fA[2]: X = ReadSingle(); Y = ReadSingle()
        if fA[6]: ReadSingle(); ReadSingle()
    """
    entry = TeleportEntry(
        peg_info=g.peg_info,
        movement=g.movement,
        rolly=g.rolly,
        bouncy=g.bouncy,
    )

    fA = br.read_byte()

    entry.width = br.read_int32()
    entry.height = br.read_int32()

    if flag_get(fA, 1):
        br.read_int16()
    if flag_get(fA, 3):
        br.read_int32()
    if flag_get(fA, 5):
        br.read_int32()
    if flag_get(fA, 4):
        # Nested entry (destination) - read and discard
        _read_nested_entry(br, version)
    if flag_get(fA, 2):
        entry.x = br.read_single()
        entry.y = br.read_single()
    if flag_get(fA, 6):
        br.read_single()
        br.read_single()

    return entry


# ---------------------------------------------------------------------------
# VariableFloat reader (from VariableFloat.cs ReadData)
# ---------------------------------------------------------------------------

def _read_variable_float(br: BinaryReader) -> float:
    """
    C# VariableFloat.ReadData:
        var1 = ReadByte()
        if var1 > 0: value = ReadSingle()
        else: variableValue = ReadString()  (int16 length + UTF8)
    Returns the float value (or 0.0 for variable expressions).
    """
    var1 = br.read_byte()
    if var1 > 0:
        return br.read_single()
    else:
        # Variable string expression - read and discard
        br.read_popcap_string()
        return 0.0


# ---------------------------------------------------------------------------
# Emitter reader (from Emitter.cs ReadData)
# ---------------------------------------------------------------------------

def _read_emitter(br: BinaryReader, version: int, g: _GenericData) -> None:
    """
    C# Emitter.ReadData - read and discard (not gameplay-relevant).
    This must consume the exact right number of bytes to keep the
    stream aligned for subsequent entries.
    """
    main_var = br.read_int32()
    fA = br.read_int16()

    # Flags (no data): fA[2]=transparency, fA[4]=randomStartPos,
    # fA[6]=changeUnknown, fA[7]=changeScale, fA[9]=changeOpacity,
    # fA[10]=changeVelocity, fA[11]=changeDirection, fA[12]=changeRotation

    _image = br.read_popcap_string()
    _width = br.read_int32()
    _height = br.read_int32()

    if main_var == 2:
        br.read_int32()          # mainVar0
        br.read_single()         # mainVar1
        br.read_popcap_string()  # mainVar2
        br.read_byte()           # mainVar3

        if flag_get(fA, 13):
            _read_variable_float(br)  # unknown0
            _read_variable_float(br)  # unknown1

    if flag_get(fA, 5):
        br.read_single()        # X
        br.read_single()        # Y

    br.read_popcap_string()     # emitImage
    br.read_single()            # unknownEmitRate
    br.read_single()            # unknown2
    br.read_single()            # rotation
    br.read_int32()             # maxQuantity

    br.read_single()            # timeBeforeFadeOut
    br.read_single()            # fadeInTime
    br.read_single()            # lifeDuration

    _read_variable_float(br)    # emitRate
    _read_variable_float(br)    # emitAreaMultiplier

    if flag_get(fA, 12):        # changeRotation
        _read_variable_float(br)  # initialRotation
        _read_variable_float(br)  # rotationVelocity
        br.read_single()          # rotationUnknown

    if flag_get(fA, 7):         # changeScale
        _read_variable_float(br)  # minScale
        _read_variable_float(br)  # scaleVelocity
        br.read_single()          # maxRandScale

    if flag_get(fA, 8):         # changeColour
        _read_variable_float(br)  # red
        _read_variable_float(br)  # green
        _read_variable_float(br)  # blue

    if flag_get(fA, 9):         # changeOpacity
        _read_variable_float(br)  # opacity

    if flag_get(fA, 10):        # changeVelocity
        _read_variable_float(br)  # minVelocityX
        _read_variable_float(br)  # minVelocityY
        br.read_single()          # maxVelocityX
        br.read_single()          # maxVelocityY
        br.read_single()          # accelerationX
        br.read_single()          # accelerationY

    if flag_get(fA, 11):        # changeDirection
        br.read_single()          # directionSpeed
        br.read_single()          # directionRandomSpeed
        br.read_single()          # directionAcceleration
        br.read_single()          # directionAngle
        br.read_single()          # directionRandomAngle

    if flag_get(fA, 6):         # changeUnknown
        br.read_single()          # unknownA
        br.read_single()          # unknownB


# ---------------------------------------------------------------------------
# Nested entry reader (for Teleport destinations)
# ---------------------------------------------------------------------------

def _read_nested_entry(br: BinaryReader, version: int):
    """
    C# LevelEntryFactory.CreateLevelEntry(BinaryReader, int):
        magic = ReadInt32()
        if magic == 0 || magic != 1: return null
        type = ReadInt32()
        entry.ReadGenericData(br, version)
        entry.ReadData(br, version)
    """
    magic = br.read_int32()
    if magic == 0 or magic != 1:
        return

    etype = br.read_int32()
    gen = _read_generic_data(br, version)

    if etype == EntryType.CIRCLE:
        _read_circle(br, version, gen)
    elif etype == EntryType.BRICK:
        _read_brick(br, version, gen)
    elif etype == EntryType.ROD:
        _read_rod(br, version, gen)
    elif etype == EntryType.POLYGON:
        _read_polygon(br, version, gen)
    elif etype == EntryType.TELEPORT:
        _read_teleport(br, version, gen)
    elif etype == EntryType.EMITTER:
        _read_emitter(br, version, gen)


# ---------------------------------------------------------------------------
# Main level parser
# ---------------------------------------------------------------------------

def parse_level(data: bytes) -> Level:
    """Parse a Peggle .dat level file from raw bytes."""
    br = BinaryReader(data)
    level = Level()

    level.version = br.read_int32()
    _always_one = br.read_byte()
    num_entries = br.read_int32()

    for i in range(num_entries):
        if br.remaining < 4:
            break

        magic = br.read_int32()
        if magic == 0:
            break  # End marker
        if magic != 1:
            break  # Invalid

        entry_type = br.read_int32()

        try:
            generic = _read_generic_data(br, level.version)
        except (EOFError, struct.error) as e:
            break

        try:
            if entry_type == EntryType.CIRCLE:
                entry = _read_circle(br, level.version, generic)
                level.circles.append(entry)
            elif entry_type == EntryType.BRICK:
                entry = _read_brick(br, level.version, generic)
                level.bricks.append(entry)
            elif entry_type == EntryType.ROD:
                entry = _read_rod(br, level.version, generic)
                level.rods.append(entry)
            elif entry_type == EntryType.POLYGON:
                entry = _read_polygon(br, level.version, generic)
                level.polygons.append(entry)
            elif entry_type == EntryType.TELEPORT:
                entry = _read_teleport(br, level.version, generic)
                level.teleports.append(entry)
            elif entry_type == EntryType.EMITTER:
                _read_emitter(br, level.version, generic)
                # Emitters are particle effects, skip
            else:
                # Unknown entry type - stop to avoid corruption
                break

        except (EOFError, struct.error):
            break

    return level


def parse_level_file(path: str | Path) -> Level:
    """Parse a Peggle .dat level file from a file path."""
    path = Path(path)
    data = path.read_bytes()
    return parse_level(data)


# ---------------------------------------------------------------------------
# Batch parsing
# ---------------------------------------------------------------------------

def parse_all_levels(levels_dir: str | Path) -> dict[str, Level]:
    """Parse all .dat files in a directory, returning {name: Level}."""
    levels_dir = Path(levels_dir)
    results = {}
    for dat_file in sorted(levels_dir.glob("*.dat")):
        if dat_file.name.startswith("_"):
            continue  # Skip _base.dat
        name = dat_file.stem.lower()
        try:
            level = parse_level_file(dat_file)
            results[name] = level
        except Exception as e:
            print(f"WARNING: Failed to parse {dat_file.name}: {e}")
    return results


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <levels_dir_or_file>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        levels = parse_all_levels(target)
        print(f"\nParsed {len(levels)} levels:\n")
        total_circles = 0
        total_bricks = 0
        total_orange = 0
        total_variable = 0
        all_good = True
        for name, level in sorted(levels.items()):
            nc = len(level.circles)
            nb = len(level.bricks)
            no = level.total_orange
            nv = level.total_variable
            nr = len(level.rods)
            np_ = len(level.polygons)
            nt = len(level.teleports)
            total_circles += nc
            total_bricks += nb
            total_orange += no
            total_variable += nv

            moving = sum(1 for c in level.circles if c.movement)
            moving += sum(1 for b in level.bricks if b.movement)

            status = "OK" if (nc + nb) > 0 else "EMPTY"
            if no == 0 and nv == 0:
                status = "NO_ORANGE"
                all_good = False

            print(f"  {name:25s}  circles={nc:3d}  bricks={nb:3d}  "
                  f"orange={no:2d}  variable={nv:2d}  rods={nr}  "
                  f"poly={np_}  tele={nt}  moving={moving}  [{status}]")

        print(f"\nTotals: {total_circles} circles, {total_bricks} bricks, "
              f"{total_orange} fixed orange, {total_variable} variable")
        print(f"Parsed {len(levels)}/55 levels "
              f"{'ALL OK' if all_good and len(levels) == 55 else 'ISSUES FOUND'}")
    else:
        level = parse_level_file(target)
        print(f"Version: 0x{level.version:02X}")
        print(f"Circles: {len(level.circles)}")
        print(f"  Blue:   {len(level.blue_pegs)}")
        print(f"  Orange: {len(level.orange_pegs)}")
        print(f"  Green:  {len(level.green_pegs)}")
        print(f"  Purple: {len(level.purple_pegs)}")
        print(f"  Variable (can-be-orange): {len(level.variable_pegs)}")
        print(f"Bricks: {len(level.bricks)}")
        print(f"  Orange bricks: {len(level.orange_bricks)}")
        print(f"Rods: {len(level.rods)}")
        print(f"Polygons: {len(level.polygons)}")
        print(f"Teleports: {len(level.teleports)}")

        if level.circles:
            print(f"\nSample pegs (first 10):")
            for c in level.circles[:10]:
                ptype = c.peg_info.peg_type.name if c.peg_info else "NONE"
                var = " VAR" if (c.peg_info and c.peg_info.variable) else ""
                mov = " MOV" if c.movement else ""
                print(f"  ({c.x:7.1f}, {c.y:7.1f}) r={c.radius:5.1f} "
                      f"type={ptype}{var}{mov}")
