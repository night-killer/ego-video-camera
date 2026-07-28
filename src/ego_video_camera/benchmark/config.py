from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .schema import MethodSpec


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark config: {config_path}")
    config = copy.deepcopy(payload)
    root_value = config.get("repo_root", "..")
    root = _resolve(config_path.parent, root_value)
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(root)
    config["benchmark"]["output_root"] = str(
        _resolve(root, config["benchmark"]["output_root"])
    )
    for source in config["datasets"]["sources"]:
        source["root"] = str(_resolve(root, source["root"]))
    for method in config["methods"]:
        method["repo"] = str(_resolve(root, method["repo"]))
        method["checkpoints"] = [
            str(_resolve(root, item)) for item in method.get("checkpoints", [])
        ]
    return config


def method_specs(config: dict[str, Any]) -> list[MethodSpec]:
    result = []
    default_timeout = int(config["benchmark"].get("timeout_sec", 7200))
    for item in config["methods"]:
        subset = item.get("subset", [])
        subset_from = item.get("subset_from")
        if subset_from:
            if subset:
                raise ValueError(f"Method {item['id']} defines both subset and subset_from")
            subset = config.get(str(subset_from), [])
        result.append(
            MethodSpec(
                method_id=str(item["id"]),
                family=str(item.get("family", item["id"])),
                display_name=str(item.get("display_name", item["id"])),
                adapter=str(item["adapter"]),
                conda_env=str(item["conda_env"]),
                repo=Path(item["repo"]),
                checkpoint_paths=tuple(Path(path) for path in item.get("checkpoints", [])),
                seeds=tuple(int(seed) for seed in item.get("seeds", [0])),
                input_intrinsics=str(item.get("input_intrinsics", "unknown")),
                causal=bool(item.get("causal", False)),
                metric_scale=bool(item.get("metric_scale", False)),
                canonical=bool(item.get("canonical", True)),
                subset=tuple(str(value) for value in subset),
                timeout_sec=int(item.get("timeout_sec", default_timeout)),
                parameters=dict(item.get("parameters", {})),
            )
        )
    ids = [method.method_id for method in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark method ids must be unique")
    return result
