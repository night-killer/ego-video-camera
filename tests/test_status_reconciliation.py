import json

from ego_video_camera.cli import _reconcile_real_artifact_status
from ego_video_camera.serialization import write_json


def test_status_reconciliation_uses_complete_gt_artifacts(tmp_path):
    status_path = tmp_path / "execution_status.json"
    write_json(
        status_path,
        {
            "real_clip_selection": "pending",
            "gt_only_validation": "pending",
            "gt_only_all_selected": "pending",
        },
    )
    selected = {
        "clips": {
            difficulty: {"recording_name": f"recording_{difficulty}"}
            for difficulty in ("easy", "medium", "hard")
        }
    }
    for clip in selected["clips"].values():
        directory = tmp_path / clip["recording_name"]
        directory.mkdir()
        write_json(directory / "gt_validation.json", {"ffprobe": {"streams": [{}]}})
        (directory / "gt_only_overlay.mp4").write_bytes(b"video")
        write_json(directory / "frame_mapping.json", [])

    _reconcile_real_artifact_status(status_path, tmp_path, selected)
    status = json.loads(status_path.read_text())
    assert status["real_clip_selection"] == "complete"
    assert status["gt_only_validation"] == "complete"
    assert status["gt_only_all_selected"] == "complete"
