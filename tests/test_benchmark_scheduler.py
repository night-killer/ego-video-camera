import json
import os
import sys
from pathlib import Path

import numpy as np

from ego_video_camera.benchmark.scheduler import (
    _conda_environment_prefix,
    _environment,
    classify_failure,
    execute_runs,
    execution_has_failures,
    worker_command,
)
from ego_video_camera.benchmark.schema import MethodSpec, RunSpec, RunStatus, SequenceRecord
from ego_video_camera.benchmark.telemetry import (
    run_monitored_command,
    worker_event_timings,
)
from ego_video_camera.benchmark.workers.common import poses_from_t_q


def _run_spec(tmp_path: Path, timeout_sec: float = 2.0) -> RunSpec:
    clip_dir = tmp_path / "clip"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "000000.jpg").touch()
    (clip_dir / "frames.csv").write_text(
        "output_index,filename,clip_time_s\n0,000000.jpg,0.0\n", encoding="utf-8"
    )
    sequence = SequenceRecord(
        dataset_id="dataset",
        sequence_id="sequence",
        clip_dir=clip_dir,
        clip_json=clip_dir / "clip.json",
        input_path=frames_dir,
        duration_sec=0.1,
        target_fps=10.0,
        reference_grade="A",
        reference_type="test",
        stratum="test",
        start_sec=0.0,
        frame_count=1,
        input_kind="frames",
    )
    method = MethodSpec(
        method_id="mock",
        family="mock",
        display_name="Mock",
        adapter="ego_video_camera.benchmark.workers.mock:run",
        conda_env="mock",
        repo=tmp_path,
        checkpoint_paths=(),
        seeds=(0,),
        input_intrinsics="not_used",
        causal=False,
        metric_scale=True,
        timeout_sec=timeout_sec,
    )
    return RunSpec(
        run_id="mock/dataset/sequence/seed_0",
        method=method,
        sequence=sequence,
        seed=0,
        output_dir=tmp_path / "output" / "runs" / "mock" / "dataset" / "sequence" / "seed_0",
    )


def test_worker_command_uses_conda_run_contract(tmp_path: Path):
    run = _run_spec(tmp_path)
    command = worker_command(run, tmp_path / "manifest.json", conda_executable="conda")
    assert command[:6] == [
        "conda",
        "run",
        "-n",
        "mock",
        "--no-capture-output",
        "python",
    ]


def test_worker_command_prefers_exact_conda_prefix(tmp_path: Path):
    run = _run_spec(tmp_path)
    prefix = tmp_path / "envs" / "mock"
    command = worker_command(
        run,
        tmp_path / "manifest.json",
        conda_executable="conda",
        conda_prefix=prefix,
    )
    assert command[:3] == [
        str(prefix / "bin" / "python"),
        "-m",
        "ego_video_camera.benchmark.worker",
    ]


def test_conda_environment_prefix_uses_configured_root(tmp_path: Path):
    prefix = tmp_path / "envs" / "mock"
    (prefix / "conda-meta").mkdir(parents=True)

    assert _conda_environment_prefix(
        {"CONDA_ENVS_PATH": str(tmp_path / "envs")}, "mock"
    ) == prefix.resolve()


def test_scheduler_dry_run_uses_configured_conda_prefix(
    tmp_path: Path, monkeypatch
):
    run = _run_spec(tmp_path)
    prefix = tmp_path / "envs" / "mock"
    (prefix / "conda-meta").mkdir(parents=True)
    monkeypatch.setenv("CONDA_ENVS_PATH", str(tmp_path / "envs"))
    config = {
        "_repo_root": str(tmp_path),
        "benchmark": {"output_root": str(tmp_path / "output")},
    }

    result = execute_runs(config, [run], dry_run=True)

    command = result["commands"][0]["command"]
    assert command[:3] == [
        str(prefix.resolve() / "bin" / "python"),
        "-m",
        "ego_video_camera.benchmark.worker",
    ]


def test_worker_environment_prioritizes_target_torch_libraries(
    tmp_path: Path, monkeypatch
):
    run = _run_spec(tmp_path)
    env_root = tmp_path / "envs"
    env_prefix = env_root / run.method.conda_env
    torch_library = (
        env_prefix / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
    )
    torch_library.mkdir(parents=True)
    (torch_library / "libtorch_python.so").touch()
    (env_prefix / "conda-meta").mkdir()
    foreign_library = "/foreign/torch/lib"
    driver_library = "/usr/local/nvidia/lib64"
    monkeypatch.setenv("CONDA_ENVS_PATH", str(env_root))
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", os.pathsep.join((foreign_library, driver_library))
    )
    monkeypatch.setenv("PYTHONHOME", "/foreign/python")
    monkeypatch.setenv("PYTHONPATH", "/foreign/python/site-packages")
    config = {
        "_repo_root": str(tmp_path),
        "benchmark": {"gpu": 0},
    }

    environment = _environment(config, run)

    library_paths = environment["LD_LIBRARY_PATH"].split(os.pathsep)
    assert library_paths == [
        str(torch_library.resolve()),
        str((env_prefix / "lib").resolve()),
        foreign_library,
        driver_library,
    ]
    assert "PYTHONHOME" not in environment
    assert environment["PYTHONPATH"] == str(tmp_path / "src")
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["CONDA_PREFIX"] == str(env_prefix.resolve())
    assert environment["CONDA_DEFAULT_ENV"] == "mock"
    assert environment["PATH"].split(os.pathsep)[0] == str(env_prefix.resolve() / "bin")


def test_pose_vectors_convert_xyzw_quaternions_to_c2w_matrices():
    half_sqrt = np.sqrt(0.5)
    vectors = np.asarray([[1.0, 2.0, 3.0, 0.0, 0.0, half_sqrt, half_sqrt]])
    expected = np.asarray(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(poses_from_t_q(vectors)[0], expected)


def test_scheduler_resume_skips_successful_run(tmp_path: Path):
    run = _run_spec(tmp_path)
    config = {
        "_repo_root": str(Path(__file__).resolve().parents[1]),
        "benchmark": {
            "output_root": str(tmp_path / "output"),
            "ffmpeg": "ffmpeg",
            "gpu": 0,
        },
    }

    def command_builder(current: RunSpec, manifest: Path):
        script = (
            "from pathlib import Path; import json; "
            f"root=Path({str(current.output_dir)!r}); "
            "(root/'prediction.npz').write_bytes(b'placeholder'); "
            "(root/'worker_result.json').write_text(json.dumps({'status':'success'}))"
        )
        return [sys.executable, "-c", script]

    first = execute_runs(config, [run], command_builder=command_builder)
    second = execute_runs(
        config, [run], resume=True, command_builder=command_builder
    )

    assert first["status_counts"] == {"success": 1}
    assert second["status_counts"] == {"skipped": 1}
    state = json.loads((run.output_dir / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "success"
    assert state["attempt"] == 1


def test_scheduler_flags_existing_failed_run_without_resume(tmp_path: Path):
    run = _run_spec(tmp_path)
    run.output_dir.mkdir(parents=True)
    (run.output_dir / "run.json").write_text(
        json.dumps({"status": "method_failed", "attempt": 1}), encoding="utf-8"
    )
    config = {
        "_repo_root": str(Path(__file__).resolve().parents[1]),
        "benchmark": {"output_root": str(tmp_path / "output")},
    }

    result = execute_runs(
        config,
        [run],
        command_builder=lambda _run, _manifest: (_ for _ in ()).throw(
            AssertionError("worker must not start")
        ),
    )

    assert result["status_counts"] == {"skipped_existing": 1}
    assert execution_has_failures(result)


def test_force_rerun_does_not_reuse_stale_worker_outputs(tmp_path: Path):
    run = _run_spec(tmp_path)
    config = {
        "_repo_root": str(Path(__file__).resolve().parents[1]),
        "benchmark": {
            "output_root": str(tmp_path / "output"),
            "ffmpeg": "ffmpeg",
            "gpu": 0,
        },
    }

    def successful_builder(current: RunSpec, _manifest: Path):
        script = (
            "from pathlib import Path; import json; "
            f"root=Path({str(current.output_dir)!r}); "
            "(root/'prediction.npz').write_bytes(b'placeholder'); "
            "(root/'worker_result.json').write_text(json.dumps({'status':'success'}))"
        )
        return [sys.executable, "-c", script]

    first = execute_runs(config, [run], command_builder=successful_builder)
    (run.output_dir / "evaluation.json").write_text("{}", encoding="utf-8")
    second = execute_runs(
        config,
        [run],
        force=True,
        command_builder=lambda _run, _manifest: [sys.executable, "-c", "pass"],
    )

    assert first["status_counts"] == {"success": 1}
    assert second["status_counts"] == {"invalid_output": 1}
    assert not (run.output_dir / "prediction.npz").exists()
    assert not (run.output_dir / "evaluation.json").exists()


def test_timeout_terminates_process_group_and_records_telemetry(tmp_path: Path):
    result = run_monitored_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        env=dict(__import__("os").environ),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        output_dir=tmp_path,
        timeout_sec=0.05,
        sample_interval_sec=0.02,
    )
    assert result.timed_out
    assert result.telemetry["wall_time_sec"] < 3.0
    assert result.telemetry["sample_count"] >= 1


def test_worker_event_timing_uses_command_start_clock(tmp_path: Path):
    events = tmp_path / "worker_events.jsonl"
    events.write_text(
        "\n".join(
            (
                json.dumps(
                    {"event": "model_ready", "monotonic_sec": 103.0, "elapsed_sec": 1.0}
                ),
                json.dumps(
                    {
                        "event": "first_prediction",
                        "monotonic_sec": 107.5,
                        "elapsed_sec": 5.5,
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    timings = worker_event_timings(events, command_started_monotonic=100.0)

    assert timings["model_ready_sec"] == 3.0
    assert timings["time_to_first_prediction_sec"] == 7.5


def test_failure_classifier_distinguishes_timeout_oom_and_invalid_output():
    assert classify_failure(
        returncode=-9,
        timed_out=True,
        stderr="",
        worker_result=None,
        prediction_exists=False,
    ) == RunStatus.TIMEOUT
    assert classify_failure(
        returncode=1,
        timed_out=False,
        stderr="torch.cuda.OutOfMemoryError: CUDA out of memory",
        worker_result=None,
        prediction_exists=False,
    ) == RunStatus.OOM
    assert classify_failure(
        returncode=0,
        timed_out=False,
        stderr="",
        worker_result={"status": "success"},
        prediction_exists=False,
    ) == RunStatus.INVALID_OUTPUT
