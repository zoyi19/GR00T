from pathlib import Path

import pytest
from tianji_wuji_runtime.deployment import infer_tianji_wuji


def test_parse_args_loads_yaml_config(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "\n".join(
            [
                "task: pick up the bottle",
                "robot_backend: tianji_wuji_host",
                "robot_ip: 192.168.8.166",
                "camera:",
                "  - head:realsense:254322070127",
                "  - left_wrist:realsense:260522279497",
                "camera_capture_fps: 60",
                "camera_fps: 15",
                "freeze_left_side: true",
                "freeze_left_use_current: true",
            ]
        ),
        encoding="utf-8",
    )

    args = infer_tianji_wuji.parse_args(["--config", str(config)])

    assert args.task == "pick up the bottle"
    assert args.robot_backend == "tianji_wuji_host"
    assert args.robot_ip == "192.168.8.166"
    assert args.camera == [
        "head:realsense:254322070127",
        "left_wrist:realsense:260522279497",
    ]
    assert args.camera_capture_fps == 60.0
    assert args.camera_fps == 15.0
    assert args.freeze_left_side is True
    assert args.freeze_left_use_current is True


def test_parse_args_cli_overrides_scalar_config_values(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "\n".join(
            [
                "task: pick up the bottle",
                "camera_fps: 15",
                "execution_horizon: 16",
            ]
        ),
        encoding="utf-8",
    )

    args = infer_tianji_wuji.parse_args(
        [
            "--config",
            str(config),
            "--camera-fps",
            "10",
            "--execution-horizon",
            "4",
        ]
    )

    assert args.camera_fps == 10.0
    assert args.execution_horizon == 4


def test_parse_args_rejects_unknown_config_key(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("task: pick up the bottle\nnot_a_real_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config key"):
        infer_tianji_wuji.parse_args(["--config", str(config)])


def test_parse_args_uses_default_infer_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    config.write_text(
        "\n".join(
            [
                "task: default task",
                "robot_backend: tianji_wuji_host",
                "camera:",
                "  - head:realsense:254322070127",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(infer_tianji_wuji, "DEFAULT_INFER_CONFIG_PATH", config)

    args = infer_tianji_wuji.parse_args([])

    assert args.config == str(config)
    assert args.task == "default task"
    assert args.robot_backend == "tianji_wuji_host"
    assert args.camera == ["head:realsense:254322070127"]
