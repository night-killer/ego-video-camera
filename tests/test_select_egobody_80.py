import csv
import json

from scripts.select_egobody_80 import (
    _window_pv_metrics,
    build_manifest,
    write_outputs,
)


def _write_actions(path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "recording_name",
                "frame_interval_start",
                "frame_interval_end",
                "frame_interval",
                "body_idx_0",
                "body_0_des",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "": "0",
                "recording_name": "desktop_recording",
                "frame_interval_start": "0",
                "frame_interval_end": "600",
                "frame_interval": "600",
                "body_idx_0": "1 female",
                "body_0_des": "draw_with_hands_clip1",
            }
        )
        writer.writerow(
            {
                "": "1",
                "recording_name": "walking_recording",
                "frame_interval_start": "0",
                "frame_interval_end": "180",
                "frame_interval": "180",
                "body_idx_0": "0 male",
                "body_0_des": "walk_clip1",
            }
        )


def test_manifest_preserves_inclusive_motion_x_endpoints_and_writes_both_views(tmp_path):
    action_csv = tmp_path / "actions.csv"
    _write_actions(action_csv)

    payload = build_manifest(tmp_path, action_csv, desktop_count=1, walking_count=1)
    desktop = payload["categories"]["desktop_head_motion"]["clips"][0]
    walking = payload["categories"]["walking_person"]["clips"][0]

    assert desktop["frame_count"] == 600
    assert desktop["frame_end_inclusive"] - desktop["frame_start_inclusive"] + 1 == 600
    assert desktop["frame_end_exclusive"] == desktop["frame_end_inclusive"] + 1
    assert walking["frame_count"] == 180
    assert payload["validation"]["duplicate_rgb_time_keys"] == []

    json_path, csv_path = write_outputs(payload, tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8"))["validation"]["primary_count"] == 2
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 3


def test_pv_rotation_metric_status_is_explicit_and_qualifies_turning_window():
    angles = [0.0, 30.0, 60.0]
    rotations = []
    import numpy as np

    for angle in angles:
        radians = np.deg2rad(angle)
        rotations.append(
            np.asarray(
                [
                    [np.cos(radians), -np.sin(radians), 0.0],
                    [np.sin(radians), np.cos(radians), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
    metrics = _window_pv_metrics(
        {"status": "computed_pv_camera_rotation_proxy_row_index", "frame": np.array([0.0, 1.0, 2.0]), "time": np.array([0.0, 10_000_000.0, 20_000_000.0]), "rotation": rotations},
        0,
        3,
    )
    assert metrics["head_motion_metric_status"].startswith("computed_pv_camera_rotation_proxy")
    assert np.isclose(metrics["head_turn_excursion_deg"], 60.0)
    assert metrics["head_motion_qualified"] is True


def test_desktop_selection_uses_qualified_candidates_when_available(tmp_path):
    action_csv = tmp_path / "actions.csv"
    with action_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["", "recording_name", "frame_interval_start", "frame_interval_end", "body_idx_0", "body_0_des"],
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow({"": index, "recording_name": f"desk{index}", "frame_interval_start": 0, "frame_interval_end": 600, "body_idx_0": "1", "body_0_des": "draw"})
    payload = build_manifest(tmp_path, action_csv, desktop_count=1, walking_count=0)
    assert payload["categories"]["desktop_head_motion"]["clips"][0]["head_motion_qualified"] is False
