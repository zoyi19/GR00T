import numpy as np

from tianji_wuji_runtime.runtime.tianji_arm_system import TianjiDualArmSystem, TianjiHostConfig


class _FakeRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    def clear_set(self) -> None:
        self.calls.append(("clear_set", None))

    def clear_error(self, arm: str) -> None:
        self.calls.append(("clear_error", arm))

    def send_cmd(self) -> None:
        self.calls.append(("send_cmd", None))


class _FakeController:
    instances: list["_FakeController"] = []

    def __init__(self, **_: object) -> None:
        self.robot = _FakeRobot()
        self.set_impedance_mode_calls: list[str] = []
        self.move_to_joints_direct_calls: list[tuple[list[float] | None, list[float] | None]] = []
        self.disable_called = False
        self._joints = (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
            np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0], dtype=np.float32),
        )
        _FakeController.instances.append(self)

    def set_impedance_mode(self, mode: str = "joint") -> None:
        self.set_impedance_mode_calls.append(mode)

    def get_current_joints(self) -> tuple[np.ndarray, np.ndarray]:
        return self._joints[0].copy(), self._joints[1].copy()

    def move_to_joints_direct(
        self,
        left_joints: list[float] | None = None,
        right_joints: list[float] | None = None,
    ) -> None:
        self.move_to_joints_direct_calls.append((left_joints, right_joints))
        if left_joints is not None:
            self._joints = (
                np.asarray(left_joints, dtype=np.float32),
                self._joints[1],
            )
        if right_joints is not None:
            self._joints = (
                self._joints[0],
                np.asarray(right_joints, dtype=np.float32),
            )

    def disable_and_release(self) -> None:
        self.disable_called = True


def test_connect_prepares_joint_control_and_caches_hold(monkeypatch, tmp_path) -> None:
    sdk_root = tmp_path / "sdk"
    sdk_root.mkdir()
    config_path = tmp_path / "robot.MvKDCfg"
    config_path.write_text("stub", encoding="utf-8")
    _FakeController.instances.clear()
    monkeypatch.setattr(
        "tianji_wuji_runtime.runtime.tianji_arm_system._load_tianji_controller",
        lambda _sdk_root: _FakeController,
    )

    system = TianjiDualArmSystem(
        TianjiHostConfig(
            robot_ip="127.0.0.1",
            sdk_root=str(sdk_root),
            config_path=str(config_path),
        )
    )

    system.connect()

    controller = _FakeController.instances[-1]
    assert controller.robot.calls == [
        ("clear_set", None),
        ("clear_error", "A"),
        ("clear_error", "B"),
        ("send_cmd", None),
    ]
    assert controller.set_impedance_mode_calls == ["joint"]
    assert np.allclose(
        system.get_joint_state("left"),
        np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
    )


def test_flush_uses_cached_other_side_when_only_one_arm_staged(monkeypatch, tmp_path) -> None:
    sdk_root = tmp_path / "sdk"
    sdk_root.mkdir()
    config_path = tmp_path / "robot.MvKDCfg"
    config_path.write_text("stub", encoding="utf-8")
    _FakeController.instances.clear()
    monkeypatch.setattr(
        "tianji_wuji_runtime.runtime.tianji_arm_system._load_tianji_controller",
        lambda _sdk_root: _FakeController,
    )

    system = TianjiDualArmSystem(
        TianjiHostConfig(
            robot_ip="127.0.0.1",
            sdk_root=str(sdk_root),
            config_path=str(config_path),
        )
    )
    system.connect()

    left_target = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float32)
    system.stage_joint_position("left", left_target)
    system.flush_staged_commands()

    controller = _FakeController.instances[-1]
    sent_left, sent_right = controller.move_to_joints_direct_calls[-1]
    assert sent_left == left_target.tolist()
    assert sent_right == [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
