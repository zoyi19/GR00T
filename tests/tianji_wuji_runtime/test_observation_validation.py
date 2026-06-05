import numpy as np
import pytest

from tianji_wuji_runtime.runtime.observation_builder import (
    ObservationError,
    validate_policy_inputs,
)
from tianji_wuji_runtime.runtime.robot_state import DualArmHandState


def _state() -> DualArmHandState:
    return DualArmHandState(
        left_arm_q=np.zeros(7, dtype=np.float32),
        right_arm_q=np.zeros(7, dtype=np.float32),
        left_hand_q=np.zeros(20, dtype=np.float32),
        right_hand_q=np.zeros(20, dtype=np.float32),
    )


def _images() -> dict[str, np.ndarray]:
    return {
        "head": np.zeros((240, 424, 3), dtype=np.uint8),
        "left_wrist": np.zeros((240, 424, 3), dtype=np.uint8),
        "right_wrist": np.zeros((240, 424, 3), dtype=np.uint8),
    }


def test_validate_policy_inputs_accepts_complete_observation() -> None:
    checked = validate_policy_inputs(
        _state(),
        _images(),
        required_camera_keys=["head", "left_wrist", "right_wrist"],
    )

    assert checked.left_arm_q.shape == (7,)


def test_validate_policy_inputs_rejects_missing_camera() -> None:
    images = _images()
    images.pop("left_wrist")

    with pytest.raises(ObservationError, match="missing camera"):
        validate_policy_inputs(
            _state(),
            images,
            required_camera_keys=["head", "left_wrist", "right_wrist"],
        )


def test_validate_policy_inputs_rejects_nonfinite_state() -> None:
    state = _state()
    state.right_hand_q[3] = np.nan

    with pytest.raises(ObservationError, match="right_hand contains NaN or Inf"):
        validate_policy_inputs(
            state,
            _images(),
            required_camera_keys=["head", "left_wrist", "right_wrist"],
        )


def test_validate_policy_inputs_rejects_bad_rgb_shape() -> None:
    images = _images()
    images["head"] = np.zeros((240, 424), dtype=np.uint8)

    with pytest.raises(ObservationError, match="image head must be HxWx3"):
        validate_policy_inputs(
            _state(),
            images,
            required_camera_keys=["head", "left_wrist", "right_wrist"],
        )
