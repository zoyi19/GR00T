"""Camera acquisition and RGB validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class CameraError(RuntimeError):
    """Raised when camera acquisition fails."""


@dataclass
class CameraSlotConfig:
    key: str
    source: str = "dummy"
    width: int = 640
    height: int = 480
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate_degrees: int = 0


class CameraManager:
    def __init__(self, slots: list[CameraSlotConfig]) -> None:
        if not slots:
            raise CameraError("at least one camera slot is required")
        self.slots = slots
        self._captures: dict[str, Any] = {}

    @classmethod
    def from_cli_specs(
        cls,
        specs: Iterable[str],
        *,
        required_keys: Iterable[str],
        image_source: str | None = None,
        allow_dummy: bool = False,
    ) -> "CameraManager":
        parsed: dict[str, CameraSlotConfig] = {}
        for spec in specs:
            slot = parse_camera_spec(spec)
            parsed[slot.key] = slot

        for key in required_keys:
            if key in parsed:
                continue
            if image_source is not None:
                parsed[key] = CameraSlotConfig(key=key, source=image_source)
            elif allow_dummy:
                parsed[key] = CameraSlotConfig(key=key, source="dummy")
            else:
                raise CameraError(f"missing camera slot {key!r}; pass --camera {key}:<source>")

        return cls([parsed[key] for key in parsed])

    def connect_all(self) -> None:
        for slot in self.slots:
            source = slot.source
            if source in {"dummy", "zeros"} or _looks_like_file(source):
                continue
            if source.isdigit():
                try:
                    import cv2  # type: ignore
                except ImportError as exc:
                    raise CameraError(
                        "opencv-python is required for live camera indices; "
                        "use --image-source for dry-run or install the runtime env"
                    ) from exc
                capture = cv2.VideoCapture(int(source))
                if not capture.isOpened():
                    raise CameraError(f"failed to open camera {slot.key}:{source}")
                self._captures[slot.key] = capture
            else:
                raise CameraError(
                    f"unsupported camera source {source!r} for {slot.key}; "
                    "use an index, image path, or dummy"
                )

    def read(self) -> dict[str, np.ndarray]:
        images = {}
        for slot in self.slots:
            images[slot.key] = self._read_slot(slot)
        return images

    def disconnect_all(self) -> None:
        for capture in self._captures.values():
            try:
                capture.release()
            except Exception:
                pass
        self._captures.clear()

    def _read_slot(self, slot: CameraSlotConfig) -> np.ndarray:
        source = slot.source
        if source in {"dummy", "zeros"}:
            image = np.zeros((slot.height, slot.width, 3), dtype=np.uint8)
        elif _looks_like_file(source):
            path = Path(source)
            if not path.exists():
                raise CameraError(f"image source does not exist for {slot.key}: {path}")
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        else:
            capture = self._captures.get(slot.key)
            if capture is None:
                raise CameraError(f"camera {slot.key} is not connected")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise CameraError(f"camera {slot.key} read timeout/failure")
            # OpenCV returns BGR.
            image = frame[..., ::-1].astype(np.uint8)

        image = _apply_orientation(image, slot)
        _validate_rgb(image, slot.key)
        return image


def parse_camera_spec(spec: str) -> CameraSlotConfig:
    if ":" not in spec:
        raise CameraError(f"camera spec must be key:source, got {spec!r}")
    key, source = spec.split(":", 1)
    key = key.strip()
    source = source.strip()
    if not key or not source:
        raise CameraError(f"camera spec must be key:source, got {spec!r}")
    return CameraSlotConfig(key=key, source=source)


def _looks_like_file(source: str) -> bool:
    suffix = Path(source).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _apply_orientation(image: np.ndarray, slot: CameraSlotConfig) -> np.ndarray:
    out = image
    if slot.flip_horizontal:
        out = np.flip(out, axis=1)
    if slot.flip_vertical:
        out = np.flip(out, axis=0)
    if slot.rotate_degrees:
        turns = (slot.rotate_degrees // 90) % 4
        out = np.rot90(out, k=turns)
    return np.ascontiguousarray(out)


def _validate_rgb(image: np.ndarray, key: str) -> None:
    if not isinstance(image, np.ndarray):
        raise CameraError(f"camera {key} did not return ndarray")
    if image.dtype != np.uint8:
        raise CameraError(f"camera {key} must return uint8 RGB, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise CameraError(f"camera {key} must return HxWx3 RGB, got {image.shape}")

