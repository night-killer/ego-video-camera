from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ego_video_camera.clip_pipeline as clip_pipeline
from ego_video_camera.clip_pipeline import (
    _da3_resume_matches,
    _load_and_document_da3_poses,
    run_clip,
)
from ego_video_camera.da3_adapter import (
    DA3_STREAMING_TO_EGOBODY_PV_CAMERA,
    EXPECTED_DA3_COMMIT,
)
from ego_video_camera.serialization import write_json


def _config(checkpoint, *, resolution=504, chunk=60, overlap=30):
    return {
        "da3": {
            "checkpoint_path": str(checkpoint),
            "input_resolution": resolution,
            "window_size": chunk,
            "window_overlap": overlap,
            "confidence_threshold": 1.5,
        }
    }


def test_da3_resume_requires_matching_inputs_and_inference_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"weight")
    output = tmp_path / "da3"
    output.mkdir()
    records = [SimpleNamespace(frame_id=10, timestamp=100), SimpleNamespace(frame_id=20, timestamp=200)]
    np.savez(
        output / "da3_poses_raw.npz",
        c2w=np.repeat(np.eye(4)[None], 2, axis=0),
        confidence=[2.0, 2.0],
        frame_ids=[10, 20],
        timestamps=[100, 200],
    )
    records = [
        SimpleNamespace(frame_id=10, timestamp=100, image_path=tmp_path / "a.jpg"),
        SimpleNamespace(frame_id=20, timestamp=200, image_path=tmp_path / "b.jpg"),
    ]
    write_json(
        output / "da3_poses_raw.json",
        {"records": [{"source_image": str(record.image_path)} for record in records]},
    )
    write_json(
        output / "da3_resolved_config.json",
        {
            "source_commit": EXPECTED_DA3_COMMIT,
            "checkpoint_status": "user_validated_local",
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_weight_size": weight.stat().st_size,
            "checkpoint_weight_mtime_ns": weight.stat().st_mtime_ns,
            "use_ray_pose": True,
            "process_res": 504,
            "chunk_size": 60,
            "requested_overlap": 30,
            "effective_overlap": 0,
            "loop_closure": False,
            "confidence_threshold": 1.5,
            "output_pose_basis": "egobody_pv",
            "input_count": 2,
        },
    )
    assert _da3_resume_matches(output, records, _config(checkpoint))
    assert not _da3_resume_matches(output, records, _config(checkpoint, resolution=392))
    assert not _da3_resume_matches(
        output,
        [SimpleNamespace(frame_id=10, timestamp=100, image_path=tmp_path / "a.jpg")],
        _config(checkpoint),
    )


def test_da3_resume_rejects_changed_input_image_or_pose_basis(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"weight")
    output = tmp_path / "da3"
    output.mkdir()
    record = SimpleNamespace(frame_id=10, timestamp=100, image_path=tmp_path / "a.jpg")
    np.savez(
        output / "da3_poses_raw.npz",
        c2w=np.eye(4)[None],
        confidence=[2.0],
        frame_ids=[10],
        timestamps=[100],
    )
    write_json(output / "da3_poses_raw.json", {"records": [{"source_image": str(record.image_path)}]})
    resolved = {
        "source_commit": EXPECTED_DA3_COMMIT,
        "checkpoint_status": "user_validated_local",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_weight_size": weight.stat().st_size,
        "checkpoint_weight_mtime_ns": weight.stat().st_mtime_ns,
        "use_ray_pose": True,
        "process_res": 504,
        "chunk_size": 60,
        "requested_overlap": 30,
        "effective_overlap": 0,
        "loop_closure": False,
        "confidence_threshold": 1.5,
        "output_pose_basis": "opencv",
        "input_count": 1,
    }
    write_json(output / "da3_resolved_config.json", resolved)
    assert not _da3_resume_matches(output, [record], _config(checkpoint))
    resolved["output_pose_basis"] = "egobody_pv"
    write_json(output / "da3_resolved_config.json", resolved)
    changed = SimpleNamespace(frame_id=10, timestamp=100, image_path=tmp_path / "other.jpg")
    assert not _da3_resume_matches(output, [changed], _config(checkpoint))


def test_legacy_da3_archive_is_basis_converted_without_rerunning_inference(tmp_path):
    output = tmp_path / "da3"
    output.mkdir()
    official = np.eye(4)[None]
    official[0, :3, 3] = [1.0, 2.0, 3.0]
    np.savez(
        output / "da3_poses_raw.npz",
        c2w=official,
        confidence=[2.0],
        frame_ids=[10],
        timestamps=[100],
    )
    write_json(
        output / "da3_poses_raw.json",
        {"records": [{"frame_id": 10, "stitched_c2w": official[0]}]},
    )
    loaded = _load_and_document_da3_poses(output)
    np.testing.assert_allclose(
        loaded["egobody_pv_c2w"][0],
        official[0] @ DA3_STREAMING_TO_EGOBODY_PV_CAMERA,
    )
    assert (output / "pose_basis_interpretation.json").is_file()


def _prepared_clip(tmp_path):
    pose = np.eye(4)[None]
    record = SimpleNamespace(frame_id=10, timestamp=100, image_path=tmp_path / "ego.jpg")
    return {
        "records": [record],
        "mappings": [SimpleNamespace(exo_image=tmp_path / "exo.jpg")],
        "gt": {
            "T_W_E": pose,
            "T_K_W": np.eye(4),
            "T_K_E": pose,
            "display_gt": pose,
            "timestamps_sec": np.asarray([0.0]),
            "head_mode": "camera_center_proxy",
            "frame_kind": "camera",
        },
        "camera": SimpleNamespace(),
        "report": {"frame_count": 1},
    }


@pytest.mark.parametrize("render,evaluate", [(True, False), (False, True), (True, True)])
def test_no_run_da3_requires_matching_cache_for_postprocessing(
    tmp_path, render, evaluate
):
    with pytest.raises(RuntimeError, match="matching DA3 resume cache"):
        run_clip(
            repo_root=tmp_path,
            data_root=tmp_path,
            output_dir=tmp_path / "output",
            clip={},
            config=_config(tmp_path / "missing-checkpoint"),
            run_da3=False,
            render_comparison=render,
            evaluate=evaluate,
            prepared_clip=_prepared_clip(tmp_path),
        )


def test_no_run_da3_allows_gt_only_without_reading_cache(monkeypatch, tmp_path):
    da3_dir = tmp_path / "output" / "da3"
    da3_dir.mkdir(parents=True)
    (da3_dir / "da3_poses_raw.npz").touch()
    monkeypatch.setattr(
        clip_pipeline,
        "_load_and_document_da3_poses",
        lambda path: pytest.fail("GT-only mode must not read a DA3 cache"),
    )
    prepared = _prepared_clip(tmp_path)
    result = run_clip(
        repo_root=tmp_path,
        data_root=tmp_path,
        output_dir=tmp_path / "output",
        clip={},
        config=_config(tmp_path / "missing-checkpoint"),
        run_da3=False,
        render_comparison=False,
        evaluate=False,
        prepared_clip=prepared,
    )

    assert result == {"gt": prepared["report"], "da3": "not_run_on_cpu"}


def test_no_run_da3_uses_matching_cache_for_requested_postprocessing(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    da3_dir = output / "da3"
    da3_dir.mkdir(parents=True)
    (da3_dir / "da3_poses_raw.npz").touch()
    prepared = _prepared_clip(tmp_path)
    pose = np.eye(4)[None]
    alignment = SimpleNamespace(status="ok", reason=None, transform=None)
    aligned = {
        name: {"alignment": alignment, "T_K_E": pose.copy()}
        for name in ("sim3_full", "sim3_prefix", "se3_full")
    }

    class Camera:
        width = 1920
        height = 1080

        def project(self, points):
            return np.zeros((len(points), 2)), np.ones(len(points), dtype=bool)

        def inside(self, pixels, width, height):
            return np.ones(len(pixels), dtype=bool)

    prepared["camera"] = Camera()
    monkeypatch.setattr(clip_pipeline, "_da3_resume_matches", lambda *args: True)
    monkeypatch.setattr(
        clip_pipeline,
        "_load_and_document_da3_poses",
        lambda path: {
            "egobody_pv_c2w": pose,
            "confidence": np.asarray([2.0]),
            "frame_ids": np.asarray([10]),
            "timestamps": np.asarray([100]),
        },
    )
    monkeypatch.setattr(clip_pipeline, "_align_da3", lambda *args: aligned)
    monkeypatch.setattr(clip_pipeline, "trajectory_metrics", lambda *args: {})
    monkeypatch.setattr(clip_pipeline, "_plot_trajectory", lambda *args: None)
    monkeypatch.setattr(clip_pipeline, "write_json", lambda *args: None)
    monkeypatch.setattr(
        clip_pipeline,
        "read_json",
        lambda path: [{"da3_pose_valid": False}] if Path(path).name == "frame_mapping.json" else {},
    )

    result = run_clip(
        repo_root=tmp_path,
        data_root=tmp_path,
        output_dir=output,
        clip={},
        config=_config(tmp_path / "missing-checkpoint"),
        run_da3=False,
        render_comparison=False,
        evaluate=True,
        prepared_clip=prepared,
    )

    assert set(result["metrics"]) == {"sim3_full", "sim3_prefix", "se3_full"}
