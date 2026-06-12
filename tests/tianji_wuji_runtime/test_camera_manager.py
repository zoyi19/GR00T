import time

import cv2
import numpy as np
from PIL import Image
import pytest
from tianji_wuji_runtime.runtime.camera_manager import (
    CameraError,
    CameraManager,
    CameraSlotConfig,
    LatestFrame,
    _enumerate_realsense_devices,
    _resolve_realsense_video_node,
)


def _dummy_manager() -> CameraManager:
    slots = [
        CameraSlotConfig(key="head", source="dummy", width=424, height=240),
        CameraSlotConfig(key="left_wrist", source="dummy", width=424, height=240),
        CameraSlotConfig(key="right_wrist", source="dummy", width=424, height=240),
    ]
    return CameraManager(slots)


def test_dummy_read_returns_target_shape() -> None:
    manager = _dummy_manager()
    manager.connect_all()
    images = manager.read()
    for key in ("head", "left_wrist", "right_wrist"):
        assert images[key].shape == (240, 424, 3)
        assert images[key].dtype == np.uint8


def test_image_source_is_resized_to_target(tmp_path) -> None:
    src = tmp_path / "head.png"
    Image.fromarray(np.full((480, 640, 3), 7, dtype=np.uint8)).save(src)
    manager = CameraManager([CameraSlotConfig(key="head", source=str(src))])
    manager.connect_all()
    image = manager.read()["head"]
    assert image.shape == (240, 424, 3)
    assert image.dtype == np.uint8


def test_cli_config_separates_capture_and_processing_fps() -> None:
    manager = CameraManager.from_cli_specs(
        ["head:18"],
        required_keys=["head"],
        width=424,
        height=240,
        capture_fps=60.0,
        fps=15.0,
    )

    slot = manager.slots[0]
    assert slot.capture_fps == 60.0
    assert slot.fps == 15.0


def test_live_device_uses_v4l2_and_native_capture_settings(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.values = {
                cv2.CAP_PROP_FRAME_WIDTH: 640.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
                cv2.CAP_PROP_FPS: 30.0,
            }
            self.set_calls = []
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            self.set_calls.append((prop, value))
            self.values[prop] = value
            return True

        def get(self, prop: int) -> float:
            return self.values.get(prop, 0.0)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    calls = []

    def fake_video_capture(target, backend):
        calls.append((target, backend))
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)
    manager = CameraManager(
        [
            CameraSlotConfig(
                key="head",
                source="/dev/video18",
                width=424,
                height=240,
                capture_fps=60.0,
                fps=15.0,
            )
        ]
    )

    manager.connect_all()

    assert calls == [("/dev/video18", cv2.CAP_V4L2)]
    assert (cv2.CAP_PROP_FRAME_WIDTH, 424) in capture.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 240) in capture.set_calls
    assert (cv2.CAP_PROP_FPS, 60.0) in capture.set_calls
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in capture.set_calls


def test_live_device_rejects_wrong_native_resolution(monkeypatch) -> None:
    class FakeCapture:
        released = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            values = {
                cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
                cv2.CAP_PROP_FPS: 60.0,
            }
            return values.get(prop, 0.0)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    manager = CameraManager(
        [CameraSlotConfig(key="head", source="/dev/video18", width=424, height=240)]
    )

    with pytest.raises(CameraError, match="rejected native capture settings"):
        manager.connect_all()

    assert capture.released


def test_enumerate_realsense_devices_parses_serial_number_and_port(monkeypatch) -> None:
    output = """
Device info:
    Name                          : \tIntel RealSense D405
    Serial Number                 : \t260522272600
    Physical Port                 : \t/sys/devices/.../video4linux/video12

Device info:
    Name                          : \tIntel RealSense D435I
    Serial Number                 : \t254322070127
    Physical Port                 : \t/sys/devices/.../video4linux/video5
"""

    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *args, **kwargs: output,
    )

    devices = _enumerate_realsense_devices()

    assert [device.serial_number for device in devices] == ["260522272600", "254322070127"]
    assert devices[0].physical_port.endswith("video12")


def test_resolve_realsense_video_node_prefers_yuyv_node(monkeypatch, tmp_path) -> None:
    video_dir = tmp_path / "2-10:1.0" / "video4linux"
    video_dir.mkdir(parents=True)
    for name in ("video12", "video14", "video16"):
        (video_dir / name).mkdir()

    monkeypatch.setattr(
        "tianji_wuji_runtime.runtime.camera_manager._enumerate_realsense_devices",
        lambda: [
            type(
                "_Device",
                (),
                {
                    "serial_number": "260522272600",
                    "physical_port": str(video_dir / "video12"),
                },
            )()
        ],
    )

    def fake_check_output(args, text=True, stderr=None):
        target = args[2]
        if target.endswith("video16"):
            return "ioctl\n\t[0]: 'YUYV' (YUYV 4:2:2)\n"
        if target.endswith("video14"):
            return "ioctl\n\t[0]: 'UYVY' (UYVY 4:2:2)\n"
        return "ioctl\n\t[0]: 'Z16 ' (16-bit Depth)\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    resolved = _resolve_realsense_video_node("260522272600")

    assert resolved.endswith("video16")


def test_streaming_starts_one_thread_per_slot() -> None:
    manager = _dummy_manager()
    manager.connect_all()
    manager.start_streaming()
    try:
        names = {t.name for t in manager._threads.values()}
        assert names == {"camera-head", "camera-left_wrist", "camera-right_wrist"}
        # Wait for first frames to land.
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            try:
                frames = manager.snapshot_latest()
                break
            except CameraError:
                time.sleep(0.02)
        else:
            pytest.fail("no frames produced by streaming workers")
        assert set(frames) == {"head", "left_wrist", "right_wrist"}
        for frame in frames.values():
            assert frame.image.shape == (240, 424, 3)
            assert frame.frame_id >= 1
    finally:
        manager.stop_streaming()
        manager.disconnect_all()


def test_streaming_tolerates_transient_initial_read_failure() -> None:
    class FlakyCameraManager(CameraManager):
        def __init__(self) -> None:
            super().__init__([CameraSlotConfig(key="head", source="dummy", fps=50.0)])
            self.failures_remaining = 1

        def _acquire_frame(self, slot: CameraSlotConfig) -> np.ndarray:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise CameraError("transient read failure")
            return np.zeros((slot.height, slot.width, 3), dtype=np.uint8)

    manager = FlakyCameraManager()
    manager.connect_all()
    manager.start_streaming()
    try:
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            try:
                frames = manager.snapshot_latest()
                break
            except CameraError:
                time.sleep(0.02)
        else:
            pytest.fail("streaming worker did not recover from transient failure")

        assert frames["head"].frame_id >= 1
        assert manager._thread_errors == {}
    finally:
        manager.stop_streaming()
        manager.disconnect_all()


def test_wait_until_ready_waits_for_first_frame() -> None:
    class DelayedCameraManager(CameraManager):
        def __init__(self) -> None:
            super().__init__([CameraSlotConfig(key="head", source="dummy", fps=50.0)])
            self.failures_remaining = 2

        def _acquire_frame(self, slot: CameraSlotConfig) -> np.ndarray:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise CameraError("not ready yet")
            return np.zeros((slot.height, slot.width, 3), dtype=np.uint8)

    manager = DelayedCameraManager()
    manager.connect_all()
    manager.start_streaming()
    try:
        frames = manager.wait_until_ready(timeout_sec=2.0)
        assert frames["head"].frame_id >= 1
        assert manager._thread_errors == {}
    finally:
        manager.stop_streaming()
        manager.disconnect_all()


def test_snapshot_requires_streaming() -> None:
    manager = _dummy_manager()
    manager.connect_all()
    with pytest.raises(CameraError, match="start_streaming"):
        manager.snapshot_latest()


def test_snapshot_no_frame_yet_errors() -> None:
    import threading

    manager = _dummy_manager()
    manager._streaming = True
    manager._frame_locks = {slot.key: threading.Lock() for slot in manager.slots}
    manager._latest_frames = {}
    with pytest.raises(CameraError, match="no frame yet"):
        manager.snapshot_latest()


def test_snapshot_stale_frame_errors() -> None:
    import threading

    manager = CameraManager([CameraSlotConfig(key="head", source="dummy")])
    manager._streaming = True
    manager._frame_locks = {"head": threading.Lock()}
    now = time.perf_counter()
    manager._latest_frames = {
        "head": LatestFrame(
            key="head",
            image=np.zeros((240, 424, 3), dtype=np.uint8),
            frame_id=1,
            wall_time=time.time(),
            monotonic_time=now - 1.0,  # 1000ms old
            source="dummy",
            width=424,
            height=240,
        )
    }
    with pytest.raises(CameraError, match="stale"):
        manager.snapshot_latest(reference_time=now, max_age_ms=150.0)


def test_stereo_crop_takes_requested_half(tmp_path) -> None:
    wide = np.zeros((240, 800, 3), dtype=np.uint8)
    wide[:, :400] = [255, 0, 0]  # left red
    wide[:, 400:] = [0, 0, 255]  # right blue
    src = tmp_path / "head.png"
    Image.fromarray(wide).save(src)

    manager = CameraManager(
        [CameraSlotConfig(key="head", source=str(src), width=424, height=240, stereo_crop="left")]
    )
    manager.connect_all()
    image = manager.read()["head"]
    assert image.shape == (240, 424, 3)
    # Cropped to the red left half, so red channel dominates.
    assert image[:, :, 0].mean() > 200
    assert image[:, :, 2].mean() < 60
