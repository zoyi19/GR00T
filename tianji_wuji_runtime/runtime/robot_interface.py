"""Unified dual-arm dual-hand robot interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import schema
from .action_adapter import DualArmHandAction
from .arm_interface import ArmConnectionConfig, ArmInterface, FakeArmInterface
from .hand_interface import FakeHandInterface, HandConnectionConfig, HandInterface


class RobotError(RuntimeError):
    """Unified runtime robot error."""


@dataclass
class RobotConnectionConfig:
    backend: str = "fake"
    left_arm_ip: str | None = None
    left_arm_port: int | None = None
    right_arm_ip: str | None = None
    right_arm_port: int | None = None
    left_hand_ip: str | None = None
    left_hand_port: int | None = None
    right_hand_ip: str | None = None
    right_hand_port: int | None = None


class DualArmHandRobot:
    """Composes four SDK interfaces behind a single safe contract."""

    def __init__(
        self,
        left_arm: ArmInterface,
        left_hand: HandInterface,
        right_arm: ArmInterface,
        right_hand: HandInterface,
    ) -> None:
        self.left_arm = left_arm
        self.left_hand = left_hand
        self.right_arm = right_arm
        self.right_hand = right_hand
        self._connected = False

    def connect(self) -> None:
        try:
            self.left_arm.connect()
            self.left_hand.connect()
            self.right_arm.connect()
            self.right_hand.connect()
            self._connected = True
        except Exception as exc:  # noqa: BLE001 - SDKs throw mixed exception types.
            self.hold_position()
            raise RobotError(f"failed to connect robot interfaces: {exc}") from exc

    def disconnect(self) -> None:
        for part in (self.left_arm, self.left_hand, self.right_arm, self.right_hand):
            try:
                part.disconnect()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_state(self) -> np.ndarray:
        try:
            state = np.concatenate(
                [
                    self.left_arm.get_joint_state(),
                    self.right_arm.get_joint_state(),
                    self.left_hand.get_joint_state(),
                    self.right_hand.get_joint_state(),
                ],
                axis=0,
            ).astype(np.float32)
            return schema.validate_flat_vector(state, dim=schema.STATE_DIM, name="robot_state")
        except Exception as exc:  # noqa: BLE001
            self.hold_position()
            raise RobotError(f"failed to read robot state: {exc}") from exc

    def send_action(self, action: DualArmHandAction) -> None:
        if not isinstance(action, DualArmHandAction):
            raise RobotError(f"send_action expects DualArmHandAction, got {type(action)!r}")
        try:
            self.left_arm.send_joint_position(action.left_arm_q)
            self.left_hand.send_joint_position(action.left_hand_q)
            self.right_arm.send_joint_position(action.right_arm_q)
            self.right_hand.send_joint_position(action.right_hand_q)
        except Exception as exc:  # noqa: BLE001
            self.hold_position()
            raise RobotError(f"failed to send robot action: {exc}") from exc

    def hold_position(self) -> None:
        for part in (self.left_arm, self.left_hand, self.right_arm, self.right_hand):
            try:
                part.hold_position()
            except Exception:
                pass

    def go_home(self) -> None:
        try:
            self.left_arm.go_home()
            self.left_hand.go_home()
            self.right_arm.go_home()
            self.right_hand.go_home()
        except Exception as exc:  # noqa: BLE001
            self.hold_position()
            raise RobotError(f"failed to go home: {exc}") from exc


def make_robot(config: RobotConnectionConfig) -> DualArmHandRobot:
    if config.backend != "fake":
        raise RobotError(
            f"robot backend {config.backend!r} is not implemented in this local runtime. "
            "Wire the vendor SDK by implementing ArmInterface/HandInterface."
        )
    return DualArmHandRobot(
        left_arm=FakeArmInterface(
            ArmConnectionConfig("left", ip=config.left_arm_ip, port=config.left_arm_port)
        ),
        left_hand=FakeHandInterface(
            HandConnectionConfig("left", ip=config.left_hand_ip, port=config.left_hand_port)
        ),
        right_arm=FakeArmInterface(
            ArmConnectionConfig("right", ip=config.right_arm_ip, port=config.right_arm_port)
        ),
        right_hand=FakeHandInterface(
            HandConnectionConfig("right", ip=config.right_hand_ip, port=config.right_hand_port)
        ),
    )
