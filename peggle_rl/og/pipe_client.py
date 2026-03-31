"""
Named-pipe client for the peggle-rl-bridge Haggle mod.

Communicates with the DLL running inside Peggle Deluxe via
the named pipe \\.\pipe\peggle_rl_bridge.

All communication is binary, little-endian.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import win32file  # type: ignore[import-untyped]
import win32pipe  # type: ignore[import-untyped]
import pywintypes  # type: ignore[import-untyped]


PIPE_NAME = r"\\.\pipe\peggle_rl_bridge"
CONNECT_TIMEOUT_S = 30.0
CONNECT_RETRY_S = 0.5


# ---------------------------------------------------------------------------
# Protocol constants (must match main.cpp)
# ---------------------------------------------------------------------------

class Cmd(IntEnum):
    GET_STATE       = 0x01
    SET_ANGLE       = 0x02
    SHOOT           = 0x03
    GET_PEGS        = 0x04
    ACTIVATE_POWER  = 0x05
    WAIT_STATE      = 0x06
    RESET_LEVEL     = 0x07
    GET_SCORE       = 0x08
    PING            = 0x09
    GET_SHOT_INFO   = 0x0A
    QUIT            = 0xFF


class Status(IntEnum):
    OK          = 0x00
    ERR_BAD_CMD = 0x01
    ERR_STATE   = 0x02
    ERR_TIMEOUT = 0x03


class GameState(IntEnum):
    NONE            = 0
    PRE_SHOT        = 1
    SHOT            = 2
    POST_SHOT       = 3
    TOTAL_MISS      = 4
    LEVEL_DONE      = 5
    POST_POST_SHOT  = 6
    SHOT_EXTENDER   = 7
    INIT_LEVEL      = 8
    CHAR_DIALOG     = 9
    ZEN_BALL        = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PegInfo:
    """A peg as reported by the bridge mod."""
    x: float
    y: float
    peg_type: int  # 0=blue, 1=orange, 2=green, 3=purple
    hit: bool


@dataclass
class StateInfo:
    """Current game state snapshot."""
    state: GameState
    gun_angle: float
    pegs_hit_this_shot: int


@dataclass
class ShotInfo:
    """Cumulative shot info."""
    pegs_hit_this_shot: int
    total_pegs_hit: int
    orange_hit: int


# ---------------------------------------------------------------------------
# Pipe client
# ---------------------------------------------------------------------------

class PeggleBridgeError(Exception):
    """Error communicating with the Peggle bridge mod."""


class PeggleBridgeClient:
    """
    Client for the peggle-rl-bridge named pipe.

    Usage::

        client = PeggleBridgeClient()
        client.connect()
        state = client.get_state()
        client.set_angle(15.0)
        client.shoot()
        client.wait_state(GameState.PRE_SHOT, timeout_ms=10000)
        client.disconnect()
    """

    def __init__(self, pipe_name: str = PIPE_NAME):
        self._pipe_name = pipe_name
        self._handle: Optional[int] = None

    @property
    def connected(self) -> bool:
        return self._handle is not None

    def connect(self, timeout: float = CONNECT_TIMEOUT_S) -> None:
        """Connect to the named pipe (blocks until available or timeout)."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                self._handle = win32file.CreateFile(
                    self._pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )
                # Set pipe to byte mode
                win32pipe.SetNamedPipeHandleState(
                    self._handle,
                    win32pipe.PIPE_READMODE_BYTE,
                    None,
                    None,
                )
                return
            except pywintypes.error:
                time.sleep(CONNECT_RETRY_S)

        raise PeggleBridgeError(
            f"Timed out connecting to {self._pipe_name} after {timeout}s. "
            "Is Peggle running with the bridge mod loaded?"
        )

    def disconnect(self) -> None:
        """Disconnect from the pipe."""
        if self._handle is not None:
            try:
                self._write(struct.pack("<B", Cmd.QUIT))
            except Exception:
                pass
            try:
                win32file.CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # -----------------------------------------------------------------------
    # Raw I/O
    # -----------------------------------------------------------------------

    def _write(self, data: bytes) -> None:
        if self._handle is None:
            raise PeggleBridgeError("Not connected")
        win32file.WriteFile(self._handle, data)

    def _read(self, n: int) -> bytes:
        if self._handle is None:
            raise PeggleBridgeError("Not connected")
        buf = b""
        while len(buf) < n:
            _, chunk = win32file.ReadFile(self._handle, n - len(buf))
            if not chunk:
                raise PeggleBridgeError("Pipe closed unexpectedly")
            buf += chunk
        return buf

    def _read_u8(self) -> int:
        return struct.unpack("<B", self._read(1))[0]

    def _read_i32(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def _read_u16(self) -> int:
        return struct.unpack("<H", self._read(2))[0]

    def _read_f32(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def _check_status(self) -> None:
        status = self._read_u8()
        if status == Status.ERR_BAD_CMD:
            raise PeggleBridgeError("Bridge returned ERR_BAD_CMD")
        if status == Status.ERR_STATE:
            raise PeggleBridgeError("Bridge returned ERR_STATE (wrong game state)")
        if status == Status.ERR_TIMEOUT:
            raise PeggleBridgeError("Bridge returned ERR_TIMEOUT")

    # -----------------------------------------------------------------------
    # Commands
    # -----------------------------------------------------------------------

    def ping(self) -> bool:
        """Test connectivity. Returns True if bridge responds with PONG."""
        self._write(struct.pack("<B", Cmd.PING))
        resp = self._read(4)
        return resp == b"PONG"

    def get_state(self) -> StateInfo:
        """Get the current game state."""
        self._write(struct.pack("<B", Cmd.GET_STATE))
        self._check_status()
        state = self._read_u8()
        angle = self._read_f32()
        extra = self._read_u8()
        return StateInfo(
            state=GameState(state),
            gun_angle=angle,
            pegs_hit_this_shot=extra,
        )

    def set_angle(self, degrees: float) -> None:
        """Set the gun angle (-97 to +97 degrees)."""
        self._write(struct.pack("<Bf", Cmd.SET_ANGLE, degrees))
        self._check_status()

    def shoot(self) -> None:
        """Fire a shot. Must be in PRE_SHOT state."""
        self._write(struct.pack("<B", Cmd.SHOOT))
        self._check_status()

    def get_pegs(self) -> list[PegInfo]:
        """Get all pegs with their positions, types, and hit status."""
        self._write(struct.pack("<B", Cmd.GET_PEGS))
        self._check_status()
        count = self._read_u16()
        pegs = []
        for _ in range(count):
            x = self._read_f32()
            y = self._read_f32()
            info = self._read_u8()
            peg_type = info & 0x7F
            hit = bool(info & 0x80)
            pegs.append(PegInfo(x=x, y=y, peg_type=peg_type, hit=hit))
        return pegs

    def activate_power(self, power: int, sub: int = 0) -> None:
        """Activate a powerup."""
        self._write(struct.pack("<Bii", Cmd.ACTIVATE_POWER, power, sub))
        self._check_status()

    def wait_state(
        self,
        target: GameState,
        timeout_ms: int = 10000,
    ) -> GameState:
        """
        Block until the game reaches the target state.
        Returns the actual state reached.
        """
        self._write(struct.pack(
            "<BBI", Cmd.WAIT_STATE, int(target), timeout_ms
        ))
        status = self._read_u8()
        actual = GameState(self._read_u8())
        if status == Status.ERR_TIMEOUT:
            raise PeggleBridgeError(
                f"Timed out waiting for state {target.name}; "
                f"current state is {actual.name}"
            )
        return actual

    def reset_level(self) -> None:
        """Reload the current level."""
        self._write(struct.pack("<B", Cmd.RESET_LEVEL))
        self._check_status()

    def get_score(self) -> int:
        """
        Get the current score (may be 0 if not available via SDK).
        The Python side should supplement with OCR if needed.
        """
        self._write(struct.pack("<B", Cmd.GET_SCORE))
        self._check_status()
        return self._read_i32()

    def get_shot_info(self) -> ShotInfo:
        """Get cumulative peg hit counters."""
        self._write(struct.pack("<B", Cmd.GET_SHOT_INFO))
        self._check_status()
        pegs_this = self._read_i32()
        total = self._read_i32()
        orange = self._read_i32()
        return ShotInfo(
            pegs_hit_this_shot=pegs_this,
            total_pegs_hit=total,
            orange_hit=orange,
        )

    def shoot_and_wait(
        self,
        angle: float,
        timeout_ms: int = 15000,
    ) -> tuple[GameState, ShotInfo]:
        """
        Convenience: set angle, shoot, wait for pre-shot or terminal state.
        Returns (state, shot_info).
        """
        self.set_angle(angle)
        self.shoot()

        # Wait for the shot to resolve (ball leaves play)
        # Poll for either PRE_SHOT (next turn) or LEVEL_DONE
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            info = self.get_state()
            if info.state in (
                GameState.PRE_SHOT,
                GameState.LEVEL_DONE,
                GameState.INIT_LEVEL,
            ):
                shot_info = self.get_shot_info()
                return info.state, shot_info
            time.sleep(0.05)

        # Timed out
        shot_info = self.get_shot_info()
        return self.get_state().state, shot_info
