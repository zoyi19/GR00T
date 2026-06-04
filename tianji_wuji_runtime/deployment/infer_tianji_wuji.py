#!/usr/bin/env python3
"""Real-runtime GR00T inference entrypoint for Tianji Wuji."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT))

from tianji_wuji_runtime.runtime.action_adapter import ActionAdapter, ActionAdapterError
from tianji_wuji_runtime.runtime.camera_manager import CameraError, CameraManager
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
from tianji_wuji_runtime.runtime.safety import SafetyConfig, SafetyError, SafetyLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--left-arm-ip", default=None)
    parser.add_argument("--left-arm-port", type=int, default=None)
    parser.add_argument("--right-arm-ip", default=None)
    parser.add_argument("--right-arm-port", type=int, default=None)
    parser.add_argument("--left-hand-ip", default=None)
    parser.add_argument("--left-hand-port", type=int, default=None)
    parser.add_argument("--right-hand-ip", default=None)
    parser.add_argument("--right-hand-port", type=int, default=None)
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
    )
    robot = make_robot(
        RobotConnectionConfig(
            backend=args.robot_backend,
            left_arm_ip=args.left_arm_ip,
            left_arm_port=args.left_arm_port,
            right_arm_ip=args.right_arm_ip,
            right_arm_port=args.right_arm_port,
            left_hand_ip=args.left_hand_ip,
            left_hand_port=args.left_hand_port,
            right_hand_ip=args.right_hand_ip,
            right_hand_port=args.right_hand_port,
        )
    )
    safety_config = SafetyConfig.permissive(
        arm_max_step=args.max_arm_joint_step,
        hand_max_step=args.max_hand_joint_step,
        arm_max_velocity=args.max_arm_velocity,
        hand_max_velocity=args.max_hand_velocity,
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
    try:
        robot.connect()
        cameras.connect_all()
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
                    robot_state = robot.get_state()
                    images = cameras.read()
                    observation = obs_builder.build(robot_state, images, args.task)
                    raw_chunk = policy.predict_action_chunk(
                        observation,
                        min_horizon=args.execution_horizon,
                    )
                    actions = adapter.split_chunk(raw_chunk)
                    safe_actions, safety_events = safety.process_chunk(
                        robot_state,
                        actions,
                        args.duration,
                    )
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
        cameras.disconnect_all()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
