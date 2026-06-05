#!/usr/bin/env python3
"""Real-runtime GR00T inference entrypoint for Tianji Wuji."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
DEFAULT_LEFT_FREEZE_TARGET_PATH = RUNTIME_ROOT / "configs" / "left_freeze_defaults.json"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter, ActionAdapterError
from tianji_wuji_runtime.runtime.camera_manager import CameraError, CameraManager
from tianji_wuji_runtime.runtime.control_overrides import (
    ControlOverrideError,
    LeftSideFreezeTarget,
    apply_left_side_freeze,
)
from tianji_wuji_runtime.runtime.executor import ActionExecutor
from tianji_wuji_runtime.runtime.groot_policy_client import GrootPolicyClient, PolicyServerError
from tianji_wuji_runtime.runtime.keyboard import KeyboardController, RuntimeState, RuntimeStateMachine
from tianji_wuji_runtime.runtime.observation_builder import ObservationBuilder, ObservationError
from tianji_wuji_runtime.runtime.recorder import Recorder
from tianji_wuji_runtime.runtime.robot_interface import (
    RobotConnectionConfig,
    RobotError,
    make_robot,
)
from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyError, SafetyLayer


def _parse_optional_joint_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != schema.LEFT_HAND_DOF:
        raise ValueError(
            f"hand home pose must provide {schema.LEFT_HAND_DOF} comma-separated values, "
            f"got {values.size}"
        )
    return tuple(float(v) for v in values.tolist())


def _parse_optional_vector(raw: str | None, *, dim: int, name: str) -> np.ndarray | None:
    if raw is None:
        return None
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.size != dim:
        raise ValueError(f"{name} must provide {dim} comma-separated values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
    return values.astype(np.float32, copy=True)


def _load_left_freeze_target(path: Path) -> LeftSideFreezeTarget:
    if not path.exists():
        raise ControlOverrideError(f"left freeze target file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ControlOverrideError(f"failed to read left freeze target file {path}: {exc}") from exc
    try:
        return LeftSideFreezeTarget(
            left_arm_q=np.asarray(raw["left_arm"], dtype=np.float32),
            left_hand_q=np.asarray(raw["left_hand"], dtype=np.float32),
            source=f"file:{path}",
        )
    except KeyError as exc:
        raise ControlOverrideError(
            f"left freeze target file {path} must contain left_arm and left_hand"
        ) from exc


def _resolve_freeze_segment(
    *,
    cli_value: np.ndarray | None,
    file_value: np.ndarray | None,
    current_value: np.ndarray | None,
    name: str,
) -> np.ndarray:
    if cli_value is not None:
        return cli_value.copy()
    if file_value is not None:
        return file_value.copy()
    if current_value is not None:
        return np.asarray(current_value, dtype=np.float32).copy()
    raise ControlOverrideError(f"no freeze target available for {name}")


def _freeze_segment_source(*, cli_value: str | None, use_current: bool) -> str:
    if cli_value is not None:
        return "cli"
    if use_current:
        return "current_state"
    return "default_json"


def _freeze_target_source(args: argparse.Namespace, file_target: LeftSideFreezeTarget | None) -> str:
    if args.freeze_left_arm is not None or args.freeze_left_hand is not None:
        base = "current_state" if args.freeze_left_use_current else (file_target.source if file_target else "none")
        return f"cli_override+{base}"
    if args.freeze_left_use_current:
        return "current_state"
    if file_target is not None:
        return file_target.source
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-limits",
        default=str(RUNTIME_ROOT / "configs" / "robot_limits.yaml"),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-timeout-ms", type=int, default=15000)
    parser.add_argument("--task", required=True)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-dir", default=str(RUNTIME_ROOT / "infer_logs"))
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--image-source", default=None)
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--camera-width", type=int, default=424)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--head-stereo-crop", choices=["left", "right"], default=None)
    parser.add_argument("--max-camera-age-ms", type=float, default=150.0)
    parser.add_argument("--camera-warmup-sec", type=float, default=3.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--left-arm-ip", default=None)
    parser.add_argument("--left-arm-port", type=int, default=None)
    parser.add_argument("--right-arm-ip", default=None)
    parser.add_argument("--right-arm-port", type=int, default=None)
    parser.add_argument("--left-hand-ip", default=None)
    parser.add_argument("--left-hand-port", type=int, default=None)
    parser.add_argument("--right-hand-ip", default=None)
    parser.add_argument("--right-hand-port", type=int, default=None)
    parser.add_argument("--left-hand-serial", default=None)
    parser.add_argument("--right-hand-serial", default=None)
    parser.add_argument("--left-hand-home", default=None)
    parser.add_argument("--right-hand-home", default=None)
    parser.add_argument("--hand-lowpass-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--tianji-sdk-root", default=None)
    parser.add_argument("--tianji-config-path", default=None)
    parser.add_argument("--max-arm-joint-step", type=float, default=3.0)
    parser.add_argument("--max-hand-joint-step", type=float, default=4.5)
    parser.add_argument("--max-arm-velocity", type=float, default=None)
    parser.add_argument("--max-hand-velocity", type=float, default=None)
    parser.add_argument("--use-filter", action="store_true")
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--robot-backend", default="fake")
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--policy-unit", default="deg")
    parser.add_argument("--control-unit", default="deg")
    parser.add_argument(
        "--freeze-left-side",
        action="store_true",
        help=(
            "Right-side-only control mode: keep left arm and left hand fixed while "
            "still feeding/predicting the full 54-DoF policy contract."
        ),
    )
    parser.add_argument(
        "--freeze-left-arm",
        default=None,
        help=(
            "Optional 7 comma-separated left-arm target for --freeze-left-side. "
            "If omitted, the target comes from --freeze-left-target-path unless "
            "--freeze-left-use-current is set."
        ),
    )
    parser.add_argument(
        "--freeze-left-hand",
        default=None,
        help=(
            "Optional 20 comma-separated left-hand target for --freeze-left-side. "
            "If omitted, the target comes from --freeze-left-target-path unless "
            "--freeze-left-use-current is set."
        ),
    )
    parser.add_argument(
        "--freeze-left-target-path",
        default=str(DEFAULT_LEFT_FREEZE_TARGET_PATH),
        help="Default left-side freeze target JSON used by --freeze-left-side.",
    )
    parser.add_argument(
        "--freeze-left-use-current",
        action="store_true",
        help="Use the current left arm + left hand state as the freeze target instead of JSON defaults.",
    )
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.no_keyboard and not args.auto_start:
        raise ValueError("--no-keyboard requires --auto-start so execution is explicit")
    if not args.freeze_left_side and (
        args.freeze_left_arm is not None or args.freeze_left_hand is not None
        or args.freeze_left_use_current
    ):
        raise ValueError(
            "--freeze-left-arm/--freeze-left-hand/--freeze-left-use-current "
            "require --freeze-left-side"
        )
    if args.freeze_left_side and args.action_mode != "absolute":
        raise ValueError("--freeze-left-side expects --action-mode absolute")
    if args.safe_mode and args.execution_horizon > 2:
        print("[runtime] warning: initial safe execution should use --execution-horizon 1 or 2")
    if not args.dry_run and args.robot_backend == "fake":
        print("[runtime] warning: robot-backend=fake; commands update only the in-memory robot")

    adapter = ActionAdapter(
        policy_unit=args.policy_unit,
        control_unit=args.control_unit,
        action_mode=args.action_mode,
    )
    policy = GrootPolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    if not policy.ping():
        raise PolicyServerError(f"cannot ping policy server {args.policy_host}:{args.policy_port}")
    modality_configs = policy.get_modality_config()
    obs_builder = ObservationBuilder(modality_configs)

    cameras = CameraManager.from_cli_specs(
        args.camera,
        required_keys=obs_builder.required_camera_keys,
        image_source=args.image_source,
        allow_dummy=args.dry_run,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        head_stereo_crop=args.head_stereo_crop,
    )
    robot = make_robot(
        RobotConnectionConfig(
            backend=args.robot_backend,
            robot_ip=args.robot_ip,
            left_arm_ip=args.left_arm_ip,
            left_arm_port=args.left_arm_port,
            right_arm_ip=args.right_arm_ip,
            right_arm_port=args.right_arm_port,
            left_hand_ip=args.left_hand_ip,
            left_hand_port=args.left_hand_port,
            right_hand_ip=args.right_hand_ip,
            right_hand_port=args.right_hand_port,
            left_hand_serial=args.left_hand_serial,
            right_hand_serial=args.right_hand_serial,
            left_hand_home=_parse_optional_joint_list(args.left_hand_home),
            right_hand_home=_parse_optional_joint_list(args.right_hand_home),
            hand_lowpass_cutoff_hz=args.hand_lowpass_cutoff_hz,
            tianji_sdk_root=args.tianji_sdk_root,
            tianji_config_path=args.tianji_config_path,
        )
    )
    safety_config = SafetyConfig.from_yaml(args.robot_limits)
    safety_config.arm_max_step = args.max_arm_joint_step
    safety_config.hand_max_step = args.max_hand_joint_step
    if args.max_arm_velocity is not None:
        safety_config.arm_max_velocity = np.full(
            schema.LEFT_ARM_DOF, args.max_arm_velocity, dtype=np.float32
        )
    if args.max_hand_velocity is not None:
        safety_config.hand_max_velocity = np.full(
            schema.LEFT_HAND_DOF, args.max_hand_velocity, dtype=np.float32
        )
    safety_config.enable_filter = args.use_filter
    safety = SafetyLayer(safety_config, adapter)
    recorder = Recorder(
        args.record_dir,
        config={
            "entrypoint": "infer_tianji_wuji.py",
            "inference_mode": "sync",
            "policy_host": args.policy_host,
            "policy_port": args.policy_port,
            "task": args.task,
            "execution_horizon": args.execution_horizon,
            "duration": args.duration,
            "control_frequency_hz": 1.0 / args.duration,
            "safe_mode": args.safe_mode,
            "dry_run": args.dry_run,
            "camera": args.camera,
            "image_source": args.image_source,
            "robot_backend": args.robot_backend,
            "control_overrides": {
                "freeze_left_side": args.freeze_left_side,
                "freeze_left_arm_source": (
                    _freeze_segment_source(
                        cli_value=args.freeze_left_arm,
                        use_current=args.freeze_left_use_current,
                    )
                ),
                "freeze_left_hand_source": (
                    _freeze_segment_source(
                        cli_value=args.freeze_left_hand,
                        use_current=args.freeze_left_use_current,
                    )
                ),
                "freeze_left_target_path": args.freeze_left_target_path,
            },
            "modality": {
                name: {
                    "keys": list(cfg.modality_keys),
                    "delta_indices": list(cfg.delta_indices),
                }
                for name, cfg in modality_configs.items()
            },
        },
        adapter=adapter,
    )
    executor = ActionExecutor(robot, adapter=adapter, recorder=recorder)
    state_machine = RuntimeStateMachine(auto_start=args.auto_start, safe_mode=args.safe_mode)

    print(f"[runtime] logs: {recorder.run_dir}")
    print(f"[runtime] inference mode: sync, action step: {args.duration:.4f}s")
    print("[runtime] controls: R run, P pause, Space hold, H home, N next safe chunk, Q quit")

    chunk_count = 0
    left_freeze_target: LeftSideFreezeTarget | None = None
    try:
        robot.connect()
        if args.freeze_left_side:
            current_state = robot.get_state() if args.freeze_left_use_current else None
            file_target = (
                None
                if args.freeze_left_use_current
                else _load_left_freeze_target(Path(args.freeze_left_target_path))
            )
            freeze_arm = _parse_optional_vector(
                args.freeze_left_arm,
                dim=schema.LEFT_ARM_DOF,
                name="--freeze-left-arm",
            )
            freeze_hand = _parse_optional_vector(
                args.freeze_left_hand,
                dim=schema.LEFT_HAND_DOF,
                name="--freeze-left-hand",
            )
            left_freeze_target = LeftSideFreezeTarget(
                left_arm_q=(
                    _resolve_freeze_segment(
                        cli_value=freeze_arm,
                        file_value=None if file_target is None else file_target.left_arm_q,
                        current_value=None if current_state is None else current_state.left_arm_q,
                        name="left_arm",
                    )
                ),
                left_hand_q=(
                    _resolve_freeze_segment(
                        cli_value=freeze_hand,
                        file_value=None if file_target is None else file_target.left_hand_q,
                        current_value=None if current_state is None else current_state.left_hand_q,
                        name="left_hand",
                    )
                ),
                source=_freeze_target_source(args, file_target),
            )
            _write_json(recorder.run_dir / "left_freeze_target.json", left_freeze_target.as_dict())
            print(
                "[runtime] freeze-left-side enabled: "
                f"left_arm={left_freeze_target.left_arm_q.tolist()}, "
                f"left_hand={left_freeze_target.left_hand_q.tolist()}"
            )
        cameras.connect_all()
        cameras.start_streaming()
        cameras.wait_until_ready(timeout_sec=args.camera_warmup_sec)
        with KeyboardController(enabled=not args.no_keyboard) as keyboard:
            def stop_requested() -> bool:
                state_machine.update(keyboard.poll())
                if state_machine.home_requested:
                    robot.go_home()
                    state_machine.home_requested = False
                if state_machine.quit_requested:
                    robot.hold_position()
                    return True
                return state_machine.state != RuntimeState.RUNNING

            while True:
                """
                R      让程序跑起来，可以继续推理执行
                P      暂停，但程序不退出
                Space  停止当前运行状态，更像急停式 hold 的软件入口
                H      命令机器人回 home
                Q      退出整个 runtime，并断开资源
                N      safe-mode 下放行下一段，执行完又暂停
                """
                state_machine.update(keyboard.poll())
                if state_machine.home_requested:
                    robot.go_home()
                    state_machine.home_requested = False
                if state_machine.quit_requested:
                    robot.hold_position()
                    break
                if state_machine.state != RuntimeState.RUNNING:
                    time.sleep(0.01)
                    continue

                raw_chunk = None
                safe_actions = []
                safety_events = []
                try:
                    state_t0 = time.perf_counter()
                    robot_state = robot.get_state()
                    state_t1 = time.perf_counter()
                    reference_time = 0.5 * (state_t0 + state_t1)
                    frames = cameras.snapshot_latest(
                        reference_time=reference_time,
                        max_age_ms=args.max_camera_age_ms,
                    )
                    images = {key: frame.image for key, frame in frames.items()}
                    observation = obs_builder.build(robot_state, images, args.task)
                    raw_chunk = policy.predict_action_chunk(
                        observation,
                        min_horizon=args.execution_horizon,
                    )
                    actions = adapter.split_chunk(raw_chunk)
                    override_events: list[dict[str, object]] = []
                    if left_freeze_target is not None:
                        actions, override_events = apply_left_side_freeze(
                            actions,
                            left_freeze_target,
                        )
                    safe_actions, safety_events = safety.process_chunk(
                        robot_state,
                        actions,
                        args.duration,
                    )
                    safety_events = override_events + safety_events
                    recorder.save_chunk(
                        observation=observation,
                        raw_chunk=raw_chunk,
                        safe_actions=safe_actions,
                        safety_events=safety_events,
                        inference_latency_ms=policy.last_latency_ms,
                    )
                    executor.execute_chunk(
                        safe_actions[: args.execution_horizon],
                        args.duration,
                        dry_run=args.dry_run,
                        chunk_index=chunk_count,
                        raw_chunk=raw_chunk,
                        safety_events=safety_events,
                        stop_callback=stop_requested,
                    )
                    chunk_count += 1
                    state_machine.pause_after_safe_chunk()
                    if args.max_chunks is not None and chunk_count >= args.max_chunks:
                        break
                except (
                    ActionAdapterError,
                    CameraError,
                    ControlOverrideError,
                    ObservationError,
                    PolicyServerError,
                    RobotError,
                    SafetyError,
                ) as exc:
                    print(f"[runtime] ERROR: {exc}")
                    robot.hold_position()
                    state_machine.to_error()
                    if args.no_keyboard:
                        return 2
    except KeyboardInterrupt:
        print("\n[runtime] interrupted")
    finally:
        robot.hold_position()
        cameras.stop_streaming()
        cameras.disconnect_all()
        robot.disconnect()
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
