import argparse
import json
from pathlib import Path

import numpy as np

import ego_video_camera.benchmark.evaluation as evaluation_module
from ego_video_camera.benchmark.evaluation import evaluate_runs, evaluate_trajectory
from ego_video_camera.benchmark.report import bootstrap_mean_ci, generate_report
from ego_video_camera.benchmark.schema import MethodSpec, PoseTrajectory, RunSpec, SequenceRecord
from ego_video_camera.benchmark.trajectory_io import read_trajectory
from ego_video_camera.benchmark.worker import run_worker
from ego_video_camera.serialization import write_json


def _method(tmp_path: Path) -> MethodSpec:
    return MethodSpec(
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
    )


def _sequence(tmp_path: Path, sequence_id: str, count: int) -> SequenceRecord:
    return SequenceRecord(
        dataset_id="dataset",
        sequence_id=sequence_id,
        clip_dir=tmp_path,
        clip_json=tmp_path / "clip.json",
        input_path=tmp_path,
        duration_sec=count / 10.0,
        target_fps=10.0,
        reference_grade="A-external",
        reference_type="synthetic",
        stratum="test",
        start_sec=0.0,
        frame_count=count,
        input_kind="frames",
    )


def _config(tmp_path: Path) -> dict:
    return {
        "benchmark": {
            "name": "test",
            "output_root": str(tmp_path / "benchmark"),
            "bootstrap_seed": 20260724,
            "bootstrap_samples": 200,
        },
        "datasets": {"grade_aliases": {"A-external": "A"}},
    }


def test_bootstrap_is_deterministic_and_sequence_based():
    first = bootstrap_mean_ci(
        [1.0, 3.0, 8.0], samples=500, rng=np.random.default_rng(123)
    )
    second = bootstrap_mean_ci(
        [1.0, 3.0, 8.0], samples=500, rng=np.random.default_rng(123)
    )
    assert first == second
    assert first[0] == 4.0
    assert first[1] <= first[0] <= first[2]


def test_mock_worker_evaluation_and_report_end_to_end(tmp_path: Path):
    frame_count = 20
    method = _method(tmp_path)
    sequence = _sequence(tmp_path, "sequence", frame_count)
    output_dir = tmp_path / "benchmark" / "runs" / "mock" / "dataset" / "sequence" / "seed_0"
    output_dir.mkdir(parents=True)
    frames = []
    for index in range(frame_count):
        image = tmp_path / f"{index:06d}.jpg"
        image.touch()
        frames.append(
            {
                "frame_id": index,
                "timestamp_ns": index * 100_000_000,
                "image_path": str(image),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": "mock/dataset/sequence/seed_0",
        "method_id": "mock",
        "adapter": method.adapter,
        "seed": 0,
        "dataset_id": "dataset",
        "sequence_id": "sequence",
        "duration_sec": 2.0,
        "target_fps": 10.0,
        "input_intrinsics": "not_used",
        "frames": frames,
        "parameters": {},
    }
    manifest_path = output_dir / "worker_manifest.json"
    write_json(manifest_path, manifest)
    exit_code = run_worker(
        argparse.Namespace(
            manifest=str(manifest_path),
            output_dir=str(output_dir),
            adapter=method.adapter,
            repo=str(tmp_path),
            checkpoint=[],
        )
    )
    assert exit_code == 0

    prediction = read_trajectory(output_dir / "prediction.npz")
    reference = PoseTrajectory(
        timestamp_ns=prediction.timestamp_ns.copy(),
        frame_id=prediction.frame_id.copy(),
        c2w=prediction.c2w.copy(),
        valid=np.ones(frame_count, dtype=bool),
        confidence=np.ones(frame_count),
    )
    evaluation = evaluate_trajectory(prediction, reference, metric_scale=True)
    write_json(output_dir / "evaluation.json", evaluation)
    write_json(
        output_dir / "run.json",
        {"status": "success", "evaluation": {"status": "success"}},
    )
    write_json(
        output_dir / "telemetry.json",
        {"wall_time_sec": 1.0, "peak_gpu_vram_mb": 0.0},
    )
    run = RunSpec(
        run_id=manifest["run_id"],
        method=method,
        sequence=sequence,
        seed=0,
        output_dir=output_dir,
    )
    report = generate_report(_config(tmp_path), [run])

    assert report["evaluated_run_count"] == 1
    assert report["leaderboard"][0]["primary_ate_m_rmse"] == 0.0
    for path in report["artifacts"].values():
        assert Path(path).is_file()
    markdown = Path(report["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert "A 榜单" in markdown


def test_report_marks_missing_results_pending(tmp_path: Path):
    method = _method(tmp_path)
    sequence = _sequence(tmp_path, "pending", 1)
    run = RunSpec(
        run_id="mock/dataset/pending/seed_0",
        method=method,
        sequence=sequence,
        seed=0,
        output_dir=tmp_path / "missing",
    )
    report = generate_report(_config(tmp_path), [run])
    assert report["evaluated_run_count"] == 0
    assert report["leaderboard"][0]["status"] == "pending"
    markdown = Path(report["artifacts"]["markdown"]).read_text(encoding="utf-8")
    assert "pending" in markdown
    assert Path(report["artifacts"]["png"]).read_bytes().startswith(b"\x89PNG")


def test_report_ignores_stale_evaluation_after_failure(tmp_path: Path):
    method = _method(tmp_path)
    sequence = _sequence(tmp_path, "failed", 1)
    output_dir = tmp_path / "failed_output"
    output_dir.mkdir()
    write_json(
        output_dir / "evaluation.json",
        {
            "primary_protocol": "initial_se3",
            "protocols": {
                "initial_se3": {"metrics": {"ate_m_rmse": 0.0}}
            },
        },
    )
    write_json(
        output_dir / "run.json",
        {
            "status": "evaluation_failed",
            "inference_status": "success",
            "evaluation": {"status": "failed"},
        },
    )
    run = RunSpec(
        run_id="mock/dataset/failed/seed_0",
        method=method,
        sequence=sequence,
        seed=0,
        output_dir=output_dir,
    )

    report = generate_report(_config(tmp_path), [run])

    assert report["evaluated_run_count"] == 0
    assert report["run_metrics"][0]["evaluation_status"] == "failed"
    assert report["leaderboard"][0].get("primary_ate_m_rmse") is None
    assert report["leaderboard"][0]["rank"] is None


def test_evaluation_resume_retries_only_failed_evaluation(tmp_path: Path):
    method = _method(tmp_path)
    sequence = _sequence(tmp_path, "retry", 1)
    output_dir = tmp_path / "retry_output"
    output_dir.mkdir()
    write_json(output_dir / "worker_manifest.json", {"frames": []})
    write_json(output_dir / "evaluation.json", {"stale": True})
    write_json(
        output_dir / "run.json",
        {
            "status": "evaluation_failed",
            "inference_status": "success",
            "evaluation": {"status": "failed"},
        },
    )
    run = RunSpec(
        run_id="mock/dataset/retry/seed_0",
        method=method,
        sequence=sequence,
        seed=0,
        output_dir=output_dir,
    )
    config = _config(tmp_path)

    skipped = evaluate_runs(config, [run])
    assert skipped["status_counts"] == {"skipped_existing_failed": 1}

    originals = {
        name: getattr(evaluation_module, name)
        for name in (
            "load_frames",
            "load_reference",
            "read_trajectory",
            "validate_prediction",
            "evaluate_trajectory",
        )
    }
    evaluation_module.load_frames = lambda *args, **kwargs: []
    evaluation_module.load_reference = lambda *args, **kwargs: object()
    evaluation_module.read_trajectory = lambda *args, **kwargs: object()
    evaluation_module.validate_prediction = lambda *args, **kwargs: None
    evaluation_module.evaluate_trajectory = lambda *args, **kwargs: {
        "primary_protocol": "initial_se3",
        "protocols": {"initial_se3": {"status": "ok", "metrics": {}}},
    }
    try:
        resumed = evaluate_runs(config, [run], resume=True)
    finally:
        for name, value in originals.items():
            setattr(evaluation_module, name, value)

    state = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    result = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert resumed["status_counts"] == {"success": 1}
    assert state["status"] == "success"
    assert state["evaluation"]["status"] == "success"
    assert "inference_status" not in state
    assert result["run_id"] == run.run_id
