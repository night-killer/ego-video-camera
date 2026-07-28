from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .schema import MethodSpec, SequenceRecord


def _conda_environments() -> tuple[set[str], str | None]:
    conda = shutil.which("conda")
    if conda is None:
        return set(), "conda executable not found"
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        paths = json.loads(result.stdout).get("envs", [])
        names = {Path(path).name for path in paths}
        return names, None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return set(), f"cannot query Conda environments: {error}"


def preflight_report(
    config: dict[str, Any],
    methods: Iterable[MethodSpec],
    sequences: Iterable[SequenceRecord],
    *,
    check_environments: bool = True,
) -> dict[str, Any]:
    method_list = list(methods)
    sequence_list = list(sequences)
    environments, conda_error = _conda_environments() if check_environments else (set(), None)
    ffmpeg_name = str(config["benchmark"].get("ffmpeg", "ffmpeg"))
    checks: list[dict[str, Any]] = []

    def add(kind: str, target: str | Path, ok: bool, detail: str = "") -> None:
        checks.append({"kind": kind, "target": str(target), "ok": bool(ok), "detail": detail})

    video_count = sum(sequence.input_kind == "video" for sequence in sequence_list)
    if video_count:
        add(
            "tool",
            ffmpeg_name,
            shutil.which(ffmpeg_name) is not None,
            f"required by {video_count} selected MP4 inputs",
        )
    if check_environments and conda_error:
        add("tool", "conda", False, conda_error)
    for method in method_list:
        add("repository", method.repo, method.repo.is_dir(), method.method_id)
        for checkpoint in method.checkpoint_paths:
            add("checkpoint", checkpoint, checkpoint.exists(), method.method_id)
        for index in method.parameters.get("torchhub_checkpoint_indices", []):
            try:
                repository = method.checkpoint_paths[int(index)]
            except (IndexError, TypeError, ValueError):
                add(
                    "torchhub_repository",
                    f"checkpoint[{index}]",
                    False,
                    f"{method.method_id}: invalid checkpoint index",
                )
            else:
                hubconf = repository / "hubconf.py"
                add(
                    "torchhub_repository",
                    hubconf,
                    repository.is_dir() and hubconf.is_file(),
                    method.method_id,
                )
        if check_environments:
            add(
                "conda_environment",
                method.conda_env,
                method.conda_env in environments,
                method.method_id,
            )
        executable = method.parameters.get("executable")
        if executable:
            path = Path(config["_repo_root"]) / str(executable)
            executable_ok = path.is_file() and path.stat().st_mode & 0o111 != 0
            add("executable", path, executable_ok, method.method_id)
    for sequence in sequence_list:
        add("input", sequence.input_path, sequence.input_path.exists(), sequence.key)
        add("metadata", sequence.clip_json, sequence.clip_json.is_file(), sequence.key)
    failed = [check for check in checks if not check["ok"]]
    return {
        "schema_version": 1,
        "status": "ok" if not failed else "failed",
        "checked_method_count": len(method_list),
        "checked_sequence_count": len(sequence_list),
        "failed_count": len(failed),
        "checks": checks,
    }
