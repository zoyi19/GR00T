import numpy as np

from tianji_wuji_runtime.runtime.hand_interface import HandConnectionConfig, WujiDirectHandInterface


class _FakeLowPass:
    def __init__(self, cutoff_freq: float) -> None:
        self.cutoff_freq = cutoff_freq


class _FakeFilterNamespace:
    LowPass = _FakeLowPass


class _FakeRealtimeController:
    def __init__(self, actual_position: np.ndarray) -> None:
        self.actual_position = np.asarray(actual_position, dtype=np.float32)
        self.targets: list[np.ndarray] = []

    def get_joint_actual_position(self) -> np.ndarray:
        return self.actual_position.copy()

    def set_joint_target_position(self, q: np.ndarray) -> None:
        target = np.asarray(q, dtype=np.float32)
        self.targets.append(target.copy())
        self.actual_position = target.reshape(-1).copy()


class _FakeHand:
    def __init__(self, actual_position: np.ndarray) -> None:
        self.actual_position = np.asarray(actual_position, dtype=np.float32)
        self.enabled_commands: list[bool] = []
        self.disable_check_called = False
        self.controller = _FakeRealtimeController(self.actual_position)

    def disable_thread_safe_check(self) -> None:
        self.disable_check_called = True

    def write_joint_enabled(self, enabled: bool) -> None:
        self.enabled_commands.append(enabled)

    def realtime_controller(self, *, enable_upstream: bool, filter) -> _FakeRealtimeController:
        assert enable_upstream is False
        assert isinstance(filter, _FakeLowPass)
        return self.controller


class _FakeWujiModule:
    filter = _FakeFilterNamespace()

    def __init__(self, actual_position: np.ndarray) -> None:
        self._actual_position = actual_position
        self.hands: list[_FakeHand] = []

    def Hand(self, serial_number=None) -> _FakeHand:  # noqa: N802 - match SDK API
        hand = _FakeHand(self._actual_position)
        self.hands.append(hand)
        return hand


def test_connect_primes_realtime_controller_with_current_pose(monkeypatch) -> None:
    initial = np.arange(20, dtype=np.float32)
    fake_module = _FakeWujiModule(initial)
    monkeypatch.setattr(
        "tianji_wuji_runtime.runtime.hand_interface._load_wujihandpy",
        lambda: fake_module,
    )

    interface = WujiDirectHandInterface(
        HandConnectionConfig(
            side="left",
            serial_number="L123",
            lowpass_cutoff_hz=7.5,
        )
    )
    interface.connect()

    hand = fake_module.hands[-1]
    assert hand.disable_check_called is True
    assert hand.enabled_commands == [True]
    assert len(hand.controller.targets) == 1
    assert np.allclose(hand.controller.targets[0].reshape(-1), initial)


def test_go_home_uses_configured_pose(monkeypatch) -> None:
    initial = np.arange(20, dtype=np.float32)
    home = np.linspace(10.0, 29.0, 20, dtype=np.float32)
    fake_module = _FakeWujiModule(initial)
    monkeypatch.setattr(
        "tianji_wuji_runtime.runtime.hand_interface._load_wujihandpy",
        lambda: fake_module,
    )

    interface = WujiDirectHandInterface(
        HandConnectionConfig(
            side="right",
            serial_number="R123",
            home_position=home,
        )
    )
    interface.connect()
    interface.go_home()

    hand = fake_module.hands[-1]
    assert len(hand.controller.targets) == 2
    assert np.allclose(hand.controller.targets[-1].reshape(-1), home)
