#!/usr/bin/env python3
"""Read the unified 54-DoF robot state once."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RUNTIME_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-backend", default="fake")
    args = parser.parse_args()
    robot = make_robot(RobotConnectionConfig(backend=args.robot_backend))
    robot.connect()
    try:
        state = robot.get_state()
        print(f"state shape: {state.shape}")
        print(state)
    finally:
        robot.hold_position()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

