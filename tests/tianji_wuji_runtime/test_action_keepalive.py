from __future__ import annotations

import threading
import time

import numpy as np
from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter, DualArmHandAction
from tianji_wuji_runtime.runtime.action_keepalive import ActionKeepalive
from tianji_wuji_runtime.runtime.executor import ActionExecutor
from tianji_wuji_runtime.runtime.robot_state import DualArmHandState


def _action(value: float) -> DualArmHandAction:
    return DualArmHandAction(
        left_arm_q=np.full(schema.LEFT_ARM_DOF, value, dtype=np.float32),
        left_hand_q=np.full(schema.LEFT_HAND_DOF, value, dtype=np.float32),
        right_arm_q=np.full(schema.RIGHT_ARM_DOF, value, dtype=np.float32),
        right_hand_q=np.full(schema.RIGHT_HAND_DOF, value, dtype=np.float32),
    )


class RecordingRobot:
    def __init__(self) -> None:
        self.actions: list[DualArmHandAction] = []
        self.sent_twice = threading.Event()

    def send_action(self, action: DualArmHandAction) -> None:
        self.actions.append(action.copy())
        if len(self.actions) >= 2:
            self.sent_twice.set()

    def get_state(self) -> DualArmHandState:
        return DualArmHandState(
            left_arm_q=np.zeros(schema.LEFT_ARM_DOF, dtype=np.float32),
            left_hand_q=np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32),
            right_arm_q=np.zeros(schema.RIGHT_ARM_DOF, dtype=np.float32),
            right_hand_q=np.zeros(schema.RIGHT_HAND_DOF, dtype=np.float32),
        )

    def hold_position(self) -> None:
        return None


def test_action_keepalive_repeats_last_action_until_stopped() -> None:
    robot = RecordingRobot()
    keepalive = ActionKeepalive(robot)  # type: ignore[arg-type]

    keepalive.start(_action(3.0), interval_sec=0.01)
    assert robot.sent_twice.wait(timeout=1.0)
    keepalive.stop()

    sent_count = len(robot.actions)
    time.sleep(0.05)

    assert sent_count >= 2
    assert len(robot.actions) == sent_count
    assert np.allclose(robot.actions[-1].right_arm_q, np.full(schema.RIGHT_ARM_DOF, 3.0))


def test_executor_returns_last_executed_action() -> None:
    robot = RecordingRobot()
    executor = ActionExecutor(robot, adapter=ActionAdapter())  # type: ignore[arg-type]
    actions = [_action(1.0), _action(2.0)]

    last_action = executor.execute_chunk(actions, dt=0.0)

    assert last_action is not None
    assert len(robot.actions) == 2
    assert np.allclose(last_action.right_arm_q, np.full(schema.RIGHT_ARM_DOF, 2.0))


def test_executor_returns_none_in_dry_run() -> None:
    robot = RecordingRobot()
    executor = ActionExecutor(robot, adapter=ActionAdapter())  # type: ignore[arg-type]

    last_action = executor.execute_chunk([_action(1.0)], dt=0.0, dry_run=True)

    assert last_action is None
    assert robot.actions == []


def test_executor_cubic_interpolates_arm_commands() -> None:
    robot = RecordingRobot()
    executor = ActionExecutor(
        robot,
        adapter=ActionAdapter(),
        arm_interpolation_hz=200.0,
        arm_interpolation_mode="cubic",
    )  # type: ignore[arg-type]

    last_action = executor.execute_chunk([_action(10.0)], dt=0.05)

    assert last_action is not None
    assert len(robot.actions) == 10
    assert np.allclose(robot.actions[0].right_arm_q, np.full(schema.RIGHT_ARM_DOF, 0.28))
    assert np.allclose(robot.actions[-1].right_arm_q, np.full(schema.RIGHT_ARM_DOF, 10.0))
    assert np.allclose(robot.actions[0].right_hand_q, np.full(schema.RIGHT_HAND_DOF, 10.0))
    assert np.all(np.diff([action.right_arm_q[0] for action in robot.actions]) > 0.0)


def test_executor_interpolation_chains_from_last_sent_target() -> None:
    robot = RecordingRobot()
    executor = ActionExecutor(
        robot,
        adapter=ActionAdapter(),
        arm_interpolation_hz=200.0,
        arm_interpolation_mode="cubic",
    )  # type: ignore[arg-type]

    executor.execute_chunk([_action(10.0), _action(20.0)], dt=0.05)

    assert len(robot.actions) == 20
    assert np.allclose(robot.actions[9].right_arm_q, np.full(schema.RIGHT_ARM_DOF, 10.0))
    assert np.allclose(robot.actions[10].right_arm_q, np.full(schema.RIGHT_ARM_DOF, 10.28))
    assert np.allclose(robot.actions[-1].right_arm_q, np.full(schema.RIGHT_ARM_DOF, 20.0))
