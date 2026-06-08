"""Fixed-period action chunk executor."""

from __future__ import annotations

from collections.abc import Callable
import time

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
        event_logger: Callable[..., None] | None = None,
    ) -> None:
        self.robot = robot
        self.adapter = adapter
        self.recorder = recorder
        self.event_logger = event_logger

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
    ) -> DualArmHandAction | None:
        last_executed_action: DualArmHandAction | None = None
        for step_idx, action in enumerate(actions):
            if stop_callback is not None and stop_callback():
                self._log(
                    "executor_stop_requested",
                    chunk_index=chunk_index,
                    step_in_chunk=step_idx,
                )
                self.robot.hold_position()
                break
            step_start = time.perf_counter()
            state_before_t0: float | None = None
            state_before_t1: float | None = None
            send_t0: float | None = None
            send_t1: float | None = None
            state_after_t0: float | None = None
            state_after_t1: float | None = None
            state_before = None
            state_after = None
            executed = False
            error: str | None = None
            try:
                state_before_t0 = time.perf_counter()
                state_before = self.robot.get_state()
                state_before_t1 = time.perf_counter()
                if not dry_run:
                    send_t0 = time.perf_counter()
                    self.robot.send_action(action)
                    send_t1 = time.perf_counter()
                    executed = True
                    last_executed_action = action.copy()
                state_after_t0 = time.perf_counter()
                state_after = self.robot.get_state()
                state_after_t1 = time.perf_counter()
            except RobotError as exc:
                error = repr(exc)
                self.robot.hold_position()
                raise
            finally:
                elapsed = time.perf_counter() - step_start
                self._log(
                    "executor_step",
                    chunk_index=chunk_index,
                    step_in_chunk=step_idx,
                    dt_sec=float(dt),
                    step_start_monotonic=step_start,
                    step_end_monotonic=step_start + elapsed,
                    control_latency_ms=elapsed * 1000.0,
                    period_overrun_ms=max((elapsed - float(dt)) * 1000.0, 0.0),
                    executed=executed,
                    dry_run=dry_run,
                    error=error,
                    state_before_latency_ms=_elapsed_ms(state_before_t0, state_before_t1),
                    send_latency_ms=_elapsed_ms(send_t0, send_t1),
                    state_after_latency_ms=_elapsed_ms(state_after_t0, state_after_t1),
                    action=_action_debug_summary(action),
                )
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
                print(f"[executor] warning: control step overran by {-remaining * 1000.0:.2f} ms")
        return last_executed_action

    def _log(self, event: str, **payload: object) -> None:
        if self.event_logger is None:
            return
        try:
            self.event_logger(event, **payload)
        except Exception:
            # Logging must never interrupt the control path.
            return


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


def _elapsed_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _action_debug_summary(action: DualArmHandAction) -> dict[str, object]:
    right_arm = action.right_arm_q
    right_hand = action.right_hand_q
    return {
        "right_arm": right_arm.tolist(),
        "right_hand": right_hand.tolist(),
        "right_arm_l2": float((right_arm * right_arm).sum() ** 0.5),
        "right_hand_l2": float((right_hand * right_hand).sum() ** 0.5),
    }
