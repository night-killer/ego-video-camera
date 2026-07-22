from pathlib import Path

import yaml

from ego_video_camera.config import load_config


def test_cli_overrides_environment_overrides_yaml(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"da3": {"sample_fps": 8}, "data_root": "/yaml"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("EGO_SAMPLE_FPS", "5")
    monkeypatch.setenv("EGO_DATA_ROOT", "/environment")
    config = load_config(path, {"da3": {"sample_fps": 3}})
    assert config["da3"]["sample_fps"] == 3
    assert config["data_root"] == "/environment"
