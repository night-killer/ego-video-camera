from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


ENV_OVERRIDES = {
    "EGO_DATA_ROOT": ("data_root",),
    "EGO_OUTPUT_ROOT": ("output_root",),
    "EGO_DA3_CHECKPOINT": ("da3", "checkpoint_path"),
    "EGO_DA3_SOURCE": ("da3", "source_root"),
    "EGO_SAMPLE_FPS": ("da3", "sample_fps"),
    "EGO_INPUT_RESOLUTION": ("da3", "input_resolution"),
    "EGO_WINDOW_SIZE": ("da3", "window_size"),
    "EGO_WINDOW_OVERLAP": ("da3", "window_overlap"),
    "EGO_DURATION_SEC": ("clip", "duration_sec"),
    "EGO_FFMPEG": ("runtime", "ffmpeg_path"),
    "EGO_FFPROBE": ("runtime", "ffprobe_path"),
}


def _set_nested(data: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _coerce_environment_value(raw: str, template: Any) -> Any:
    if isinstance(template, bool):
        normalized = raw.strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"Invalid boolean environment override: {raw}")
        return normalized in {"true", "1", "yes"}
    if isinstance(template, int) and not isinstance(template, bool):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for env_name, keys in ENV_OVERRIDES.items():
        if os.environ.get(env_name):
            template = _get_nested(config, keys)
            _set_nested(config, keys, _coerce_environment_value(os.environ[env_name], template))
    if cli_overrides:
        config = deep_merge(config, cli_overrides)
    config["_config_path"] = str(path)
    return config


def resolve_path(value: str | Path | None, repo_root: str | Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()
