from types import SimpleNamespace

import numpy as np

from ego_video_camera.clip_pipeline import (
    _da3_resume_matches,
    _load_and_document_da3_poses,
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
    np.savez(output / "da3_poses_raw.npz", frame_ids=[10, 20], timestamps=[100, 200])
    write_json(output / "da3_poses_raw.json", {"complete": True})
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
            "input_count": 2,
        },
    )
    assert _da3_resume_matches(output, records, _config(checkpoint))
    assert not _da3_resume_matches(output, records, _config(checkpoint, resolution=392))
    assert not _da3_resume_matches(
        output,
        [SimpleNamespace(frame_id=10, timestamp=100)],
        _config(checkpoint),
    )


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
