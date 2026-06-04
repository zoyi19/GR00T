"""Fixed-period action chunk executor."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .action_adapter import ActionAdapter, DualArmHandAction
from .recorder import Recorder
from .robot_interface import DualArmHandRobot, RobotError


class ActionExecutor:
    def __init__(
        self,
        robot: DualArmHandRobot,
        *,
        adapter: ActionAdapter,
        recorder: Recorder | None = None,
    ) -> None:
        self.robot = robot
        self.adapter = adapter
        self.recorder = recorder

    def execute_chunk(
        self,
        actions: list[DualArmHandAction],
        dt: float,
        *,
        dry_run: bool = False,
        chunk_index: int = 0,
        raw_chunk: np.ndarray | None = None,
        safety_events: list[dict[str, object]] | None = None,
        stop_callback: Callable[[], bool] | None = None,
    ) -> None:
        for step_idx, action in enumerate(actions):
            if stop_callback is not None and stop_callback():
                self.robot.hold_position()
                break
            step_start = time.perf_counter()
            state_before = None
            state_after = None
            executed = False
            try:
                state_before = self.robot.get_state()
                if not dry_run:
                    self.robot.send_action(action)
                    executed = True
                state_after = self.robot.get_state()
            except RobotError:
                self.robot.hold_position()
                raise
            finally:
                elapsed = time.perf_counter() - step_start
                if self.recorder is not None:
                    raw_action = raw_chunk[step_idx] if raw_chunk is not None else None
                    self.recorder.record_step(
                        chunk_index=chunk_index,
                        step_in_chunk=step_idx,
                        state_before=state_before,
                        raw_action=raw_action,
                        safe_action=action,
                        executed=executed,
                        state_after=state_after,
                        control_latency_ms=elapsed * 1000.0,
                        safety_events=[
                            event
                            for event in safety_events or []
                            if event.get("step_in_chunk") == step_idx
                        ],
                    )
            remaining = float(dt) - elapsed
            if remaining > 0:
                _interruptible_sleep(remaining, stop_callback)
            else:
                print(
                    f"[executor] warning: control step overran by {-remaining * 1000.0:.2f} ms"
                )


def _interruptible_sleep(
    seconds: float,
    stop_callback: Callable[[], bool] | None,
    quantum: float = 0.01,
) -> None:
    deadline = time.perf_counter() + seconds
    while True:
        if stop_callback is not None and stop_callback():
            return
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(quantum, remaining))
