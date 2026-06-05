import importlib.util
import json
from pathlib import Path

import numpy as np

from tianji_wuji_runtime.runtime import schema
from tianji_wuji_runtime.runtime.camera_manager import CameraManager, CameraSlotConfig
from tianji_wuji_runtime.runtime.robot_interface import RobotConnectionConfig, make_robot


_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "tianji_wuji_runtime"
    / "tools"
    / "capture_observation_samples.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("capture_observation_samples", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_once_writes_full_sample(tmp_path) -> None:
    tool = _load_tool()

    robot = make_robot(RobotConnectionConfig(backend="fake"))
    cameras = CameraManager(
        [
            CameraSlotConfig(key="head", source="dummy"),
            CameraSlotConfig(key="left_wrist", source="dummy"),
            CameraSlotConfig(key="right_wrist", source="dummy"),
        ]
    )
    run_dir = tmp_path / "run"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True)

    robot.connect()
    cameras.connect_all()
    cameras.start_streaming()
    try:
        # Allow the workers to publish a first frame.
        import time

        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            try:
                cameras.snapshot_latest()
                break
            except Exception:
                time.sleep(0.02)

        record = tool.capture_once(
            robot=robot,
            cameras=cameras,
            task="debug observation",
            index=0,
            samples_dir=samples_dir,
            run_dir=run_dir,
            max_camera_age_ms=None,
        )
    finally:
        cameras.stop_streaming()
        cameras.disconnect_all()
        robot.disconnect()

    sample_dir = samples_dir / "sample_000000"
    for name in ("head.png", "left_wrist.png", "right_wrist.png", "state.json", "observation.npz"):
        assert (sample_dir / name).exists()

    state = json.loads((sample_dir / "state.json").read_text())
    assert len(state["flat"]) == schema.STATE_DIM
    assert len(state["segments"]["left_arm"]) == 7
    assert len(state["segments"]["right_arm"]) == 7
    assert len(state["segments"]["left_hand"]) == 20
    assert len(state["segments"]["right_hand"]) == 20

    obs = np.load(sample_dir / "observation.npz")
    assert obs["video.head"].shape == (1, 1, 240, 424, 3)
    assert obs["video.head"].dtype == np.uint8
    assert obs["state.left_arm_joint"].shape == (1, 1, 7)
    assert obs["state.right_hand"].shape == (1, 1, 20)
    assert obs["state.left_arm_joint"].dtype == np.float32

    for key in ("head", "left_wrist", "right_wrist"):
        assert record["camera"][key]["shape"] == [240, 424, 3]
        assert "age_ms" in record["camera"][key]
