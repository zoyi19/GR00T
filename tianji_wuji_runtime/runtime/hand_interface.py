"""Dexterous hand SDK boundary for Tianji Wuji runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema


class HandError(RuntimeError):
    """Raised by hand SDK adapters."""


@dataclass
class HandConnectionConfig:
    side: str
    ip: str | None = None
    port: int | None = None
    unit: str = schema.STATE_UNIT


class HandInterface:
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass

    def get_joint_state(self) -> np.ndarray:
        """Return 20-dim joint positions."""
        raise NotImplementedError

    def send_joint_position(self, q: np.ndarray) -> None:
        raise NotImplementedError

    def hold_position(self) -> None:
        raise NotImplementedError

    def go_home(self) -> None:
        raise NotImplementedError


class FakeHandInterface(HandInterface):
    """In-memory hand used for dry-run and integration tests."""

    def __init__(self, config: HandConnectionConfig) -> None:
        self.config = config
        self.connected = False
        self.q = np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_state(self) -> np.ndarray:
        return self.q.astype(np.float32, copy=True)

    def send_joint_position(self, q: np.ndarray) -> None:
        arr = np.asarray(q, dtype=np.float32)
        if arr.shape != (schema.LEFT_HAND_DOF,):
            raise HandError(f"{self.config.side} hand command must have shape (20,), got {arr.shape}")
        self.q = arr.copy()

    def hold_position(self) -> None:
        return None

    def go_home(self) -> None:
        self.q[:] = 0.0

