from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..serialization import read_json, write_json
from .registry import load_frames, write_worker_manifest
from .schema import RunSpec, RunStatus
from .telemetry import run_monitored_command, utc_now


OOM_MARKERS = (
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "hip out of memory",
    "outofmemoryerror",
    "std::bad_alloc",
)

ATTEMPT_OUTPUTS = (
    "evaluation.json",
    "prediction.json",
    "prediction.npz",
    "stderr.log",
    "stdout.log",
    "telemetry.json",
    "worker_events.jsonl",
    "worker_result.json",
    "worker_summary.json",
)


def worker_command(
    run: RunSpec,
    manifest_path: Path,
    *,
    conda_executable: str = "conda",
    conda_prefix: Path | None = None,
) -> list[str]:
    python_command = (
        [str(conda_prefix / "bin" / "python")]
        if conda_prefix is not None
        else [
            conda_executable,
            "run",
            "-n",
            run.method.conda_env,
            "--no-capture-output",
            "python",
        ]
    )
    command = [
        *python_command,
        "-m",
        "ego_video_camera.benchmark.worker",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(run.output_dir),
        "--adapter",
        run.method.adapter,
        "--repo",
        str(run.method.repo),
    ]
    for checkpoint in run.method.checkpoint_paths:
        command.extend(("--checkpoint", str(checkpoint)))
    return command


def _conda_environment_prefix(
    environment: Mapping[str, str], conda_env: str
) -> Path | None:
    roots = environment.get("CONDA_ENVS_PATH", "")
    return next(
        (
            Path(root).expanduser().resolve() / conda_env
            for root in roots.split(os.pathsep)
            if root
            and (Path(root).expanduser() / conda_env / "conda-meta").is_dir()
        ),
        None,
    )


def _read_optional(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _tail(path: Path, maximum_bytes: int = 1_000_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def classify_failure(
    *,
    returncode: int,
    timed_out: bool,
    stderr: str,
    worker_result: dict[str, Any] | None,
    prediction_exists: bool,
) -> RunStatus:
    if timed_out:
        return RunStatus.TIMEOUT
    combined = "\n".join(
        (
            stderr,
            str((worker_result or {}).get("error_type", "")),
            str((worker_result or {}).get("message", "")),
        )
    ).lower()
    if any(marker in combined for marker in OOM_MARKERS):
        return RunStatus.OOM
    if returncode == 0:
        if prediction_exists and (worker_result or {}).get("status") == "success":
            return RunStatus.SUCCESS
        return RunStatus.INVALID_OUTPUT
    invalid_markers = (
        "prediction frame_id",
        "prediction timestamps",
        "prediction contains",
        "did not return posetrajectory",
        "invalid homogeneous",
        "rotations are not orthonormal",
    )
    if any(marker in combined for marker in invalid_markers):
        return RunStatus.INVALID_OUTPUT
    return RunStatus.METHOD_FAILED


def _base_state(run: RunSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "method_id": run.method.method_id,
        "dataset_id": run.sequence.dataset_id,
        "sequence_id": run.sequence.sequence_id,
        "reference_grade": run.sequence.reference_grade,
        "seed": run.seed,
    }


def _successful_existing_run(run: RunSpec, state: dict[str, Any] | None) -> bool:
    inference_status = (state or {}).get("inference_status", (state or {}).get("status"))
    return bool(
        state
        and inference_status == RunStatus.SUCCESS.value
        and (run.output_dir / "prediction.npz").is_file()
        and (_read_optional(run.output_dir / "worker_result.json") or {}).get("status")
        == "success"
    )


def _clear_attempt_outputs(run: RunSpec) -> None:
    for name in ATTEMPT_OUTPUTS:
        path = run.output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()
    work_dir = run.output_dir / "work"
    if work_dir.is_symlink() or work_dir.is_file():
        work_dir.unlink()
    elif work_dir.is_dir():
        shutil.rmtree(work_dir)


def _prepend_conda_runtime_libraries(
    environment: dict[str, str], conda_env: str
) -> None:
    prefix = _conda_environment_prefix(environment, conda_env)
    if prefix is None:
        return

    library_paths: list[str] = []
    torch_libraries = sorted(
        prefix.glob("lib/python*/site-packages/torch/lib"),
        key=lambda path: str(path),
    )
    for torch_library in torch_libraries:
        if (torch_library / "libtorch_python.so").is_file():
            library_paths.append(str(torch_library.resolve()))
            break
    library_paths.append(str((prefix / "lib").resolve()))
    library_paths.extend(
        path
        for path in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(library_paths))


def _environment(config: dict[str, Any], run: RunSpec) -> dict[str, str]:
    environment = os.environ.copy()
    repo_root = Path(config["_repo_root"])
    source = str(repo_root / "src")
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = source
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(config["benchmark"].get("gpu", 0)),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    prefix = _conda_environment_prefix(environment, run.method.conda_env)
    if prefix is not None:
        environment["CONDA_PREFIX"] = str(prefix)
        environment["CONDA_DEFAULT_ENV"] = run.method.conda_env
        environment["PATH"] = os.pathsep.join(
            (str(prefix / "bin"), environment.get("PATH", ""))
        )
    _prepend_conda_runtime_libraries(environment, run.method.conda_env)
    temporary = run.output_dir / "work" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    return environment


CommandBuilder = Callable[[RunSpec, Path], Sequence[str]]


def execution_has_failures(summary: dict[str, Any]) -> bool:
    counts = summary.get("status_counts", {})
    successful = int(counts.get("success", 0)) + int(counts.get("skipped", 0))
    return bool(sum(int(value) for value in counts.values()) > successful)


def execute_runs(
    config: dict[str, Any],
    runs: Iterable[RunSpec],
    *,
    resume: bool = False,
    force: bool = False,
    dry_run: bool = False,
    command_builder: CommandBuilder | None = None,
) -> dict[str, Any]:
    run_list = list(runs)
    output_root = Path(config["benchmark"]["output_root"])
    frame_cache_dir = output_root / "cache" / "frames"
    ffmpeg = str(config["benchmark"].get("ffmpeg", "ffmpeg"))
    conda = shutil.which("conda") or "conda"
    if command_builder is None:
        command_builder = lambda run, manifest: worker_command(
            run,
            manifest,
            conda_executable=conda,
            conda_prefix=_conda_environment_prefix(os.environ, run.method.conda_env),
        )
    if dry_run:
        commands = []
        for run in run_list:
            manifest = run.output_dir / "worker_manifest.json"
            commands.append(
                {
                    **_base_state(run),
                    "status": RunStatus.PENDING.value,
                    "command": list(command_builder(run, manifest)),
                }
            )
        return {
            "schema_version": 1,
            "dry_run": True,
            "run_count": len(run_list),
            "commands": commands,
        }

    counts: Counter[str] = Counter()
    records = []
    frame_cache: dict[str, Any] = {}
    for index, run in enumerate(run_list):
        run.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = run.output_dir / "run.json"
        old_state = _read_optional(state_path)
        if _successful_existing_run(run, old_state) and not force:
            counts["skipped"] += 1
            records.append({**_base_state(run), "status": "skipped", "reason": "already_success"})
            continue
        if old_state and not (resume or force):
            counts["skipped_existing"] += 1
            records.append(
                {
                    **_base_state(run),
                    "status": "skipped_existing",
                    "reason": "existing_run_use_resume_or_force",
                }
            )
            continue

        base = {
            **_base_state(run),
            "matrix_index": index,
            "matrix_size": len(run_list),
            "status": RunStatus.RUNNING.value,
            "started_at": utc_now(),
            "attempt": int((old_state or {}).get("attempt", 0)) + 1,
        }
        write_json(state_path, base)
        manifest_path = run.output_dir / "worker_manifest.json"
        try:
            _clear_attempt_outputs(run)
            frames = frame_cache.get(run.sequence.key)
            if frames is None:
                frames = load_frames(
                    run.sequence,
                    cache_dir=frame_cache_dir,
                    ffmpeg=ffmpeg,
                    materialize=True,
                )
                frame_cache[run.sequence.key] = frames
            write_worker_manifest(
                manifest_path,
                run.run_id,
                run.method,
                run.sequence,
                frames,
                seed=run.seed,
            )
            command = list(command_builder(run, manifest_path))
            stdout_path = run.output_dir / "stdout.log"
            stderr_path = run.output_dir / "stderr.log"
            result = run_monitored_command(
                command,
                cwd=Path(config["_repo_root"]),
                env=_environment(config, run),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_dir=run.output_dir,
                timeout_sec=run.method.timeout_sec,
            )
            write_json(run.output_dir / "telemetry.json", result.telemetry)
            worker_result = _read_optional(run.output_dir / "worker_result.json")
            status = classify_failure(
                returncode=result.returncode,
                timed_out=result.timed_out,
                stderr=_tail(stderr_path),
                worker_result=worker_result,
                prediction_exists=(run.output_dir / "prediction.npz").is_file(),
            )
            final_state = {
                **base,
                "status": status.value,
                "ended_at": result.ended_at,
                "worker_started_at": result.started_at,
                "command": command,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "worker_result": worker_result,
                "telemetry": result.telemetry,
            }
        except (OSError, ValueError, RuntimeError) as error:
            status = RunStatus.INPUT_ERROR
            final_state = {
                **base,
                "status": status.value,
                "ended_at": utc_now(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        write_json(state_path, final_state)
        counts[status.value] += 1
        records.append({**_base_state(run), "status": status.value})

    return {
        "schema_version": 1,
        "dry_run": False,
        "run_count": len(run_list),
        "status_counts": dict(sorted(counts.items())),
        "runs": records,
    }
