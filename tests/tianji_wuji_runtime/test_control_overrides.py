import numpy as np
import pytest

from tianji_wuji_runtime.runtime.action_adapter import DualArmHandAction
from tianji_wuji_runtime.runtime.control_overrides import (
    ControlOverrideError,
    LeftSideFreezeTarget,
    apply_left_side_freeze,
)
from tianji_wuji_runtime.runtime.robot_state import DualArmHandState


def _action(offset: float = 0.0) -> DualArmHandAction:
    return DualArmHandAction(
        left_arm_q=np.arange(7, dtype=np.float32) + offset,
        left_hand_q=np.arange(20, dtype=np.float32) + 10.0 + offset,
        right_arm_q=np.arange(7, dtype=np.float32) + 100.0 + offset,
        right_hand_q=np.arange(20, dtype=np.float32) + 200.0 + offset,
    )


def test_left_freeze_preserves_right_side_policy_actions() -> None:
    target = LeftSideFreezeTarget(
        left_arm_q=np.full(7, 1.5, dtype=np.float32),
        left_hand_q=np.full(20, -2.0, dtype=np.float32),
        source="test",
    )
    actions = [_action(0.0), _action(1.0)]

    frozen, events = apply_left_side_freeze(actions, target)

    assert len(frozen) == 2
    assert len(events) == 2
    assert events[0]["type"] == "control_override"
    for original, overridden in zip(actions, frozen):
        np.testing.assert_allclose(overridden.left_arm_q, target.left_arm_q)
        np.testing.assert_allclose(overridden.left_hand_q, target.left_hand_q)
        np.testing.assert_allclose(overridden.right_arm_q, original.right_arm_q)
        np.testing.assert_allclose(overridden.right_hand_q, original.right_hand_q)


def test_left_freeze_target_can_be_captured_from_state() -> None:
    state = DualArmHandState(
        left_arm_q=np.arange(7, dtype=np.float32),
        right_arm_q=np.zeros(7, dtype=np.float32),
        left_hand_q=np.arange(20, dtype=np.float32),
        right_hand_q=np.zeros(20, dtype=np.float32),
    )

    target = LeftSideFreezeTarget.from_state(state)

    np.testing.assert_allclose(target.left_arm_q, state.left_arm_q)
    np.testing.assert_allclose(target.left_hand_q, state.left_hand_q)
    assert target.source == "current_state"


def test_left_freeze_rejects_bad_shape() -> None:
    with pytest.raises(ControlOverrideError, match="left_freeze_arm"):
        LeftSideFreezeTarget(
            left_arm_q=np.zeros(6, dtype=np.float32),
            left_hand_q=np.zeros(20, dtype=np.float32),
            source="test",
        )
