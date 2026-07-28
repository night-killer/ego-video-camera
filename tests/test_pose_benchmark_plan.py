import json
from pathlib import Path

import numpy as np

from ego_video_camera.benchmark.config import load_benchmark_config, method_specs
from ego_video_camera.benchmark.plan import build_run_matrix, matrix_summary, validate_inventory
from ego_video_camera.benchmark.registry import discover_sequences, write_worker_manifest
from ego_video_camera.benchmark.schema import FrameRecord, MethodSpec, SequenceRecord
from ego_video_camera.benchmark.windowing import local_window_trajectory, stitch_pose_windows


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ego_pose_benchmark.yaml"


def test_full_benchmark_inventory_and_matrix_are_fixed():
    config = load_benchmark_config(CONFIG)
    methods = method_specs(config)
    sequences = discover_sequences(config)
    validate_inventory(config, sequences)
    runs = build_run_matrix(methods, sequences, config["benchmark"]["output_root"])
    summary = matrix_summary(runs)

    assert len(methods) == 14
    assert sum(method.canonical for method in methods) == 12
    assert len(sequences) == 70
    assert len({sequence.key for sequence in sequences}) == 70
    assert summary["run_count"] == 1856
    assert summary["canonical_run_count"] == 1820
    assert summary["ablation_run_count"] == 36


def test_worker_manifest_has_no_reference_or_gt_fields(tmp_path: Path):
    image = tmp_path / "000000.jpg"
    image.touch()
    method = MethodSpec(
        method_id="test",
        family="test",
        display_name="Test",
        adapter="example:run",
        conda_env="test",
        repo=tmp_path,
        checkpoint_paths=(),
        seeds=(0,),
        input_intrinsics="provided",
        causal=False,
        metric_scale=False,
    )
    sequence = SequenceRecord(
        dataset_id="dataset",
        sequence_id="sequence",
        clip_dir=tmp_path,
        clip_json=tmp_path / "clip.json",
        input_path=tmp_path,
        duration_sec=0.1,
        target_fps=10.0,
        reference_grade="A",
        reference_type="hidden",
        stratum="test",
        start_sec=0.0,
        frame_count=1,
        input_kind="frames",
    )
    frame = FrameRecord(0, 0, image, intrinsic=np.eye(3))
    path = tmp_path / "manifest.json"
    payload = write_worker_manifest(
        path, "test/dataset/sequence/seed_0", method, sequence, [frame], seed=0
    )

    serialized = json.dumps(payload).lower()
    for forbidden in ("reference", "groundtruth", "ground_truth", "gt_path", "clip_json"):
        assert forbidden not in serialized
    assert np.array_equal(payload["frames"][0]["intrinsic"], np.eye(3))


def _poses(x_values):
    poses = np.repeat(np.eye(4)[None], len(x_values), axis=0)
    poses[:, 0, 3] = x_values
    return poses


def test_window_stitching_uses_predicted_overlap_and_preserves_strings():
    all_ids = np.arange(5)
    all_times = all_ids * 100_000_000
    first_rows = [
        {"frame_id": int(index), "timestamp_ns": int(all_times[index])}
        for index in (0, 1, 2, 3)
    ]
    second_rows = [
        {"frame_id": int(index), "timestamp_ns": int(all_times[index])}
        for index in (1, 2, 3, 4)
    ]
    first = local_window_trajectory(first_rows, _poses([0.0, 1.0, 2.0, 3.0]))
    second = local_window_trajectory(second_rows, _poses([0.0, 2.0, 4.0, 6.0]))

    result = stitch_pose_windows(
        [first, second], timestamp_ns=all_times, frame_id=all_ids
    )

    assert result.valid.all()
    assert np.allclose(result.c2w[:, 0, 3], np.arange(5))
    assert result.metadata["stitching"] == "prediction_only"
    assert result.metadata["stitch_events"][1]["status"] == "sim3"
    assert set(result.tracking_state) == {"tracking"}
