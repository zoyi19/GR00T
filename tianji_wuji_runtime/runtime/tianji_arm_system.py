"""Host-side Tianji dual-arm integration using the legacy Marvin SDK wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any

import numpy as np

from . import schema


DEFAULT_TIANJI_SDK_ROOT = Path(
    "/home/user/workspace/DexProj_back_up_0602/"
    "wuji-hand-teleop/src/output_devices/tianji_output"
)
DEFAULT_TIANJI_CONFIG_PATH = DEFAULT_TIANJI_SDK_ROOT / "tianji_output/config/ccs_m6.MvKDCfg"


@dataclass
class TianjiHostConfig:
    robot_ip: str
    sdk_root: str | None = None
    config_path: str | None = None
    unit: str = schema.STATE_UNIT


class TianjiHostError(RuntimeError):
    """Raised when the Tianji host-side integration cannot proceed safely."""


class TianjiDualArmSystem:
    """Single shared controller for the Tianji left/right arm pair."""

    def __init__(self, config: TianjiHostConfig) -> None:
        self.config = config
        self._controller: Any | None = None
        self._connected = False
        self._resource_id = object()
        self._staged_left: np.ndarray | None = None
        self._staged_right: np.ndarray | None = None
        self._hold_left: np.ndarray | None = None
        self._hold_right: np.ndarray | None = None
        self._command_lock = threading.RLock()

    def shared_resource_id(self) -> object:
        return self._resource_id

    def connect(self) -> None:
        if self._connected:
            return
        controller_cls = _load_tianji_controller(self._resolve_sdk_root())
        config_path = self._resolve_config_path()
        self._controller = controller_cls(
            robot_ip=self.config.robot_ip,
            config_path=str(config_path),
            dry_run=False,
            read_only=False,
            feedback_handshake=False,
            prefer_last_ik_reference=False,
            ik_subprocess_isolate=False,
        )
        self._prepare_joint_control()
        self._connected = True

    def disconnect(self) -> None:
        controller = self._controller
        self._controller = None
        self._connected = False
        self._clear_staged_commands()
        self._hold_left = None
        self._hold_right = None
        if controller is None:
            return
        try:
            controller.disable_and_release()
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to release Tianji controller: {exc}") from exc

    def get_joint_state(self, side: str) -> np.ndarray:
        left, right = self.get_current_joints()
        values = left if side == "left" else right
        return _as_arm_array(values, name=f"{side}_arm_state")

    def stage_joint_position(self, side: str, q: np.ndarray) -> None:
        arr = _as_arm_array(q, name=f"{side}_arm_command")
        with self._command_lock:
            if side == "left":
                self._staged_left = arr
            else:
                self._staged_right = arr

    def flush_staged_commands(self) -> None:
        controller = self._require_controller()
        with self._command_lock:
            left_cmd = self._staged_left
            right_cmd = self._staged_right
            hold_left = None if self._hold_left is None else self._hold_left.copy()
            hold_right = None if self._hold_right is None else self._hold_right.copy()
            self._staged_left = None
            self._staged_right = None

        if left_cmd is None and right_cmd is None:
            return

        if left_cmd is None or right_cmd is None:
            if hold_left is None or hold_right is None:
                hold_left, hold_right = self.get_current_joints()
            if left_cmd is None:
                left_cmd = hold_left
            if right_cmd is None:
                right_cmd = hold_right

        try:
            controller.move_to_joints_direct(
                left_joints=left_cmd.tolist(),
                right_joints=right_cmd.tolist(),
            )
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to send Tianji joint command: {exc}") from exc
        self._update_hold_targets(left_cmd, right_cmd)

    def hold_position(self) -> None:
        self._clear_staged_commands()
        try:
            left, right = self.get_current_joints()
        except TianjiHostError:
            if self._hold_left is None or self._hold_right is None:
                raise
            left = self._hold_left.copy()
            right = self._hold_right.copy()
        try:
            self._require_controller().move_to_joints_direct(
                left_joints=left.tolist(),
                right_joints=right.tolist(),
            )
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to hold Tianji arms: {exc}") from exc
        self._update_hold_targets(left, right)

    def go_home(self) -> None:
        controller = self._require_controller()
        self._clear_staged_commands()
        try:
            controller.move_to_init(wait=False, duration=3.0, dt=0.01, sides="both")
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to send Tianji home trajectory: {exc}") from exc

    def get_current_joints(self) -> tuple[np.ndarray, np.ndarray]:
        controller = self._require_controller()
        try:
            left, right = controller.get_current_joints()
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(f"failed to read Tianji joint feedback: {exc}") from exc
        left_arr = _as_arm_array(left, name="left_arm_state")
        right_arr = _as_arm_array(
            right,
            name="right_arm_state",
        )
        self._update_hold_targets(left_arr, right_arr)
        return left_arr, right_arr

    def _prepare_joint_control(self) -> None:
        controller = self._require_controller()
        try:
            self._clear_robot_errors_and_prime_feedback(controller)
            controller.set_impedance_mode(mode="joint")
            left, right = controller.get_current_joints()
        except Exception as exc:  # noqa: BLE001
            raise TianjiHostError(
                "Tianji controller connected but joint-control preparation failed "
                f"(set mode / clear error / feedback verify path): {exc}"
            ) from exc
        self._update_hold_targets(
            _as_arm_array(left, name="left_arm_feedback_verify"),
            _as_arm_array(right, name="right_arm_feedback_verify"),
        )

    def _require_controller(self) -> Any:
        if self._controller is None:
            raise TianjiHostError("Tianji controller is not connected")
        return self._controller

    def _clear_staged_commands(self) -> None:
        with self._command_lock:
            self._staged_left = None
            self._staged_right = None

    def _clear_robot_errors_and_prime_feedback(self, controller: Any) -> None:
        robot = getattr(controller, "robot", None)
        if robot is None:
            return
        clear_set = getattr(robot, "clear_set", None)
        clear_error = getattr(robot, "clear_error", None)
        send_cmd = getattr(robot, "send_cmd", None)
        if not callable(clear_set) or not callable(clear_error) or not callable(send_cmd):
            return
        clear_set()
        clear_error("A")
        clear_error("B")
        send_cmd()

    def _update_hold_targets(self, left: np.ndarray, right: np.ndarray) -> None:
        with self._command_lock:
            self._hold_left = np.asarray(left, dtype=np.float32).copy()
            self._hold_right = np.asarray(right, dtype=np.float32).copy()

    def _resolve_sdk_root(self) -> Path:
        root = Path(self.config.sdk_root) if self.config.sdk_root else DEFAULT_TIANJI_SDK_ROOT
        if not root.exists():
            raise TianjiHostError(
                f"Tianji SDK root does not exist: {root}. Pass --tianji-sdk-root explicitly."
            )
        return root

    def _resolve_config_path(self) -> Path:
        if self.config.config_path is not None:
            path = Path(self.config.config_path)
        else:
            path = DEFAULT_TIANJI_CONFIG_PATH
        if not path.exists():
            raise TianjiHostError(
                f"Tianji kinematics config does not exist: {path}. "
                "Pass --tianji-config-path explicitly."
            )
        return path


def _load_tianji_controller(sdk_root: Path) -> type[Any]:
    _ensure_ament_index_stub(sdk_root)
    root_str = str(sdk_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    loaded = sys.modules.get("tianji_output")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", "") or ""
        if loaded_file and not Path(loaded_file).resolve().is_relative_to(sdk_root.resolve()):
            sys.modules.pop("tianji_output", None)
    try:
        module = import_module("tianji_output")
    except Exception as exc:  # noqa: BLE001
        raise TianjiHostError(
            f"failed to import legacy Tianji SDK package from {sdk_root}: {exc}"
        ) from exc
    controller = getattr(module, "TianjiArmController", None)
    if controller is None:
        raise TianjiHostError("legacy tianji_output package does not export TianjiArmController")
    return controller


def _ensure_ament_index_stub(sdk_root: Path) -> None:
    try:
        import_module("ament_index_python.packages")
        return
    except Exception:
        pass

    package_root = sdk_root / "tianji_output"

    def get_package_share_directory(_: str) -> str:
        return str(package_root)

    ament_module = ModuleType("ament_index_python")
    packages_module = ModuleType("ament_index_python.packages")
    packages_module.get_package_share_directory = get_package_share_directory
    ament_module.packages = packages_module
    sys.modules.setdefault("ament_index_python", ament_module)
    sys.modules["ament_index_python.packages"] = packages_module


def _as_arm_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (schema.LEFT_ARM_DOF,):
        raise TianjiHostError(f"{name} must have shape ({schema.LEFT_ARM_DOF},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise TianjiHostError(f"{name} contains NaN or Inf")
    return arr
