from pathlib import Path

import numpy as np

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter, DualArmHandAction
from tianji_wuji_runtime.runtime.robot_state import DualArmHandState
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyLayer


def test_safety_config_loads_robot_limits_yaml() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "tianji_wuji_runtime"
        / "configs"
        / "robot_limits.yaml"
    )

    config = SafetyConfig.from_yaml(config_path)

    assert config.left_arm_joint_min.shape == (schema.LEFT_ARM_DOF,)
    assert config.left_hand_joint_min.shape == (schema.LEFT_HAND_DOF,)
    assert np.allclose(
        config.left_arm_joint_min,
        np.array([-178.0, -120.0, -178.0, -145.0, -178.0, -60.0, -90.0], dtype=np.float32),
    )
    assert config.enable_arm_joint_limit is True
    assert config.enable_hand_joint_limit is False
    assert config.enable_arm_delta_clip is True
    assert config.enable_hand_delta_clip is False
    assert config.enable_arm_velocity_limit is True
    assert config.enable_hand_velocity_limit is False
    assert config.arm_max_velocity is None
    assert config.hand_max_velocity is None


def test_safety_config_yaml_scalar_velocity_expands(tmp_path: Path) -> None:
    config_path = tmp_path / "robot_limits.yaml"
    config_path.write_text(
        "\n".join(
            [
                "left_arm_joint_min: [0, 0, 0, 0, 0, 0, 0]",
                "left_arm_joint_max: [1, 1, 1, 1, 1, 1, 1]",
                "right_arm_joint_min: [0, 0, 0, 0, 0, 0, 0]",
                "right_arm_joint_max: [1, 1, 1, 1, 1, 1, 1]",
                "left_hand_joint_min: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "left_hand_joint_max: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
                "right_hand_joint_min: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "right_hand_joint_max: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]",
                "arm_max_velocity: 2.5",
                "hand_max_velocity: 3.5",
            ]
        ),
        encoding="utf-8",
    )

    config = SafetyConfig.from_yaml(config_path)

    assert np.allclose(config.arm_max_velocity, np.full(schema.LEFT_ARM_DOF, 2.5, dtype=np.float32))
    assert np.allclose(
        config.hand_max_velocity,
        np.full(schema.LEFT_HAND_DOF, 3.5, dtype=np.float32),
    )


def test_default_safety_clips_arm_but_not_hand() -> None:
    adapter = ActionAdapter()
    config = SafetyConfig.permissive(arm_max_step=3.0, hand_max_step=4.5)
    safety = SafetyLayer(config, adapter)
    current = DualArmHandState(
        left_arm_q=np.zeros(schema.LEFT_ARM_DOF, dtype=np.float32),
        right_arm_q=np.zeros(schema.RIGHT_ARM_DOF, dtype=np.float32),
        left_hand_q=np.zeros(schema.LEFT_HAND_DOF, dtype=np.float32),
        right_hand_q=np.zeros(schema.RIGHT_HAND_DOF, dtype=np.float32),
    )
    action = DualArmHandAction(
        left_arm_q=np.full(schema.LEFT_ARM_DOF, 20.0, dtype=np.float32),
        right_arm_q=np.full(schema.RIGHT_ARM_DOF, -20.0, dtype=np.float32),
        left_hand_q=np.full(schema.LEFT_HAND_DOF, 999.0, dtype=np.float32),
        right_hand_q=np.full(schema.RIGHT_HAND_DOF, -999.0, dtype=np.float32),
    )

    processed, events = safety.process_chunk(current, [action], dt=0.05)

    assert np.allclose(processed[0].left_arm_q, np.full(schema.LEFT_ARM_DOF, 3.0, dtype=np.float32))
    assert np.allclose(processed[0].right_arm_q, np.full(schema.RIGHT_ARM_DOF, -3.0, dtype=np.float32))
    assert np.allclose(processed[0].left_hand_q, np.full(schema.LEFT_HAND_DOF, 999.0, dtype=np.float32))
    assert np.allclose(processed[0].right_hand_q, np.full(schema.RIGHT_HAND_DOF, -999.0, dtype=np.float32))
    assert {event["segment"] for event in events} == {"left_arm", "right_arm"}
