import time

import numpy as np
import pytest
from PIL import Image

from tianji_wuji_runtime.runtime.camera_manager import (
    CameraError,
    CameraManager,
    CameraSlotConfig,
    LatestFrame,
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
