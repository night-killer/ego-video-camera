import hashlib
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ego_video_camera.robot_exo as robot_exo
from ego_video_camera.eval_dataset_download import DOWNLOADERS, execute_download, load_plan
from ego_video_camera.robot_exo import (
    _prepare_rh20t_exo,
    _stage_rh20t_exo_members,
    droid_pose_matrix,
    exo_quality_passes,
    nearest_timestamp_indices,
    projection_quality,
    rh20t_aligned_extrinsics,
    scale_camera_intrinsics,
    select_droid_exo_candidate,
    prepare_robot_exo,
)
from ego_video_camera.robot_io import robot_demo_selection


ROOT = Path(__file__).resolve().parents[1]


def test_droid_euler_xyz_pose_is_camera_to_reference():
    pose = droid_pose_matrix([1, 2, 3, 0, 0, np.pi / 2])
    np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(
        pose[:3, :3],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=1e-12,
    )
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1])


def test_nearest_exo_timestamp_sync_reports_signed_delta():
    indices, deltas = nearest_timestamp_indices([0, 100, 200], [49, 50, 151])
    np.testing.assert_array_equal(indices, [0, 1, 2])
    np.testing.assert_array_equal(deltas, [-49, 50, 49])


def test_projection_quality_and_formal_gate_use_all_output_frames():
    ego = np.repeat(np.eye(4)[None], 4, axis=0)
    exo = np.repeat(np.eye(4)[None], 4, axis=0)
    ego[:, 2, 3] = 2.0
    ego[-1, 0, 3] = 100.0
    quality = projection_quality(
        ego,
        exo,
        np.asarray([[100, 0, 50], [0, 100, 50], [0, 0, 1.0]]),
        100,
        100,
    )
    assert quality["inside_ratio"] == pytest.approx(0.75)
    assert exo_quality_passes(quality, 0.95, 0.70)
    assert not exo_quality_passes(quality, 0.95, 0.80)


def test_droid_candidate_score_prefers_inside_ratio_then_margin():
    candidates = [
        {
            "serial": "a",
            "quality": {"inside_ratio": 0.9, "median_border_margin_px": 200},
        },
        {
            "serial": "b",
            "quality": {"inside_ratio": 1.0, "median_border_margin_px": 10},
        },
        {
            "serial": "c",
            "quality": {"inside_ratio": 1.0, "median_border_margin_px": 20},
        },
    ]
    assert select_droid_exo_candidate(candidates)["serial"] == "c"


def test_robot_plan_excludes_real_and_selects_exactly_seven_clips():
    plan = load_plan(ROOT / "configs" / "ego_pose_eval_robot_interaction_rgb.yaml")
    selection = robot_demo_selection(plan)
    assert len(selection) == 7
    assert [item["dataset"] for item in selection] == ["droid_wrist"] * 3 + [
        "rh20t_wrist"
    ] * 4
    assert all(not item["sequence_id"].startswith("REAL+") for item in selection)
    assert [item["sequence_id"] for item in selection[3:]] == [
        "task_0012_user_0010_scene_0008_cfg_0003",
        "task_0015_user_0010_scene_0005_cfg_0003",
        "task_0016_user_0010_scene_0009_cfg_0003",
        "task_0017_user_0010_scene_0002_cfg_0003",
    ]
    real = plan["datasets"]["droid_wrist"]["demo_exo"]["sequences"][
        "REAL+abf65a9e+2023-04-06-14h-26m-59s"
    ]
    assert real["serial"] == "23960472"
    assert real["excluded"] is True


def test_intrinsics_scale_to_actual_decoded_resolution():
    source = np.asarray([[800, 4, 640], [0, 900, 360], [0, 0, 1.0]])
    scaled, scale_x, scale_y = scale_camera_intrinsics(
        source, 1280, 720, 640, 360
    )
    assert scale_x == pytest.approx(0.5)
    assert scale_y == pytest.approx(0.5)
    np.testing.assert_allclose(
        scaled, [[400, 2, 320], [0, 450, 180], [0, 0, 1]]
    )


def test_rh20t_official_extrinsic_direction_closes(tmp_path: Path):
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    wrist = np.eye(4)
    exo = np.eye(4)
    exo[:3, 3] = [0.2, -0.1, 0.4]
    np.save(calibration / "extrinsics.npy", {"wrist": wrist, "exo": exo})
    np.save(calibration / "tcp.npy", [0, 0, 0, 1, 0, 0, 0])
    camera = {
        "tcp_camera_matrix": np.eye(4).tolist(),
        "align_base_matrix": np.eye(4).tolist(),
    }
    result = rh20t_aligned_extrinsics(calibration, camera, "wrist")
    np.testing.assert_allclose(result["exo"], exo)
    np.testing.assert_allclose(result["exo"] @ np.linalg.inv(result["exo"]), np.eye(4))


def test_rh20t_stream_tar_extracts_only_selected_members(tmp_path: Path):
    archive = tmp_path / "synthetic.tar.gz"
    required = {"ROOT/scene/cam_exo/color.mp4", "ROOT/scene/cam_exo/timestamps.npy"}
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in (
            ("ROOT/scene/cam_exo/color.mp4", b"video"),
            ("ROOT/scene/cam_exo/timestamps.npy", b"timestamps"),
            ("ROOT/unrelated/large.bin", b"ignore"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    stage = tmp_path / "data" / "_cache" / "rh20t" / "extracted"
    data_root = tmp_path / "data"
    _stage_rh20t_exo_members(archive, stage, required, data_root)
    assert all((stage / name).is_file() for name in required)
    assert not (stage / "ROOT/unrelated/large.bin").exists()


def _rh20t_lifecycle_plan(archive: Path) -> dict:
    payload = archive.read_bytes()
    identity = np.eye(4).tolist()
    return {
        "profile": {"id": "synthetic"},
        "datasets": {
            "rh20t_wrist": {
                "reference_type": "robot_kinematic_hand_eye",
                "archive_name": "archive.tar.gz",
                "archive_root": "ROOT",
                "archive_bytes": len(payload),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "google_drive_id": "id",
                "mirror_url": "https://example.invalid/archive",
                "in_hand_serial": "wrist",
                "api_repo": "repo",
                "api_commit": "commit",
                "camera": {},
                "demo_exo": {
                    "serial": "exo",
                    "source_resolution": [100, 100],
                    "sync_tolerance_ms": 50,
                    "minimum_synchronized_ratio": 0.95,
                    "minimum_projection_inside_ratio": 0.70,
                },
                "clips": [{"sequence": "scene"}],
            }
        },
    }


def _prepare_lifecycle_root(root: Path) -> None:
    reference = root / "rh20t_wrist" / "clips" / "scene" / "reference"
    reference.mkdir(parents=True)
    (reference / "camera.json").write_text(
        json.dumps({"calibration_id": "calib"}), encoding="utf-8"
    )


def test_rh20t_demo_base_uses_demo_clips_without_mutating_base_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "profile": {"id": "synthetic", "target_fps": 10},
        "datasets": {
            "rh20t_wrist": {
                "clips": [{"sequence": "base", "start_s": 0, "duration_s": 15}],
                "demo_exo": {
                    "clips": [
                        {"sequence": "demo", "start_s": 1, "duration_s": 15}
                    ]
                },
            }
        },
    }
    original = deepcopy(plan)
    calls = []

    def fake_download(context):
        calls.append(context)
        return [{"sequence_id": "demo"}]

    monkeypatch.setattr(robot_exo, "download_rh20t_wrist", fake_download)
    robot_exo._ensure_rh20t_demo_base(
        plan,
        tmp_path,
        "ffmpeg",
        2,
        False,
        None,
        plan["datasets"]["rh20t_wrist"]["demo_exo"]["clips"],
    )

    assert plan == original
    assert calls[0].plan["datasets"]["rh20t_wrist"]["clips"] == [
        {"sequence": "demo", "start_s": 1, "duration_s": 15}
    ]
    assert calls[0].robot_with_exo is True


def test_rh20t_auto_archive_is_removed_but_caller_archive_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_archive = tmp_path / "caller.tar.gz"
    source_archive.write_bytes(b"archive")
    plan = _rh20t_lifecycle_plan(source_archive)

    def fake_stage(archive_path, stage_root, required, data_root):
        video = stage_root / "ROOT" / "scene" / "cam_exo" / "color.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")

    monkeypatch.setattr(robot_exo, "_stage_rh20t_exo_members", fake_stage)
    monkeypatch.setattr(
        robot_exo,
        "_rh20t_reference_poses",
        lambda clip_dir: (
            np.asarray([1000, 1100, 1200]),
            np.asarray(
                [
                    np.asarray(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]],
                        dtype=float,
                    )
                ]
                * 3
            ),
        ),
    )

    def fake_load(path, label):
        if "timestamp" in label:
            return {"color": np.asarray([1000, 1100, 1200])}
        return {"exo": np.asarray([[10, 0, 50], [0, 10, 50], [0, 0, 1]])}

    monkeypatch.setattr(robot_exo, "_rh20t_load_dict", fake_load)
    monkeypatch.setattr(
        robot_exo,
        "rh20t_aligned_extrinsics",
        lambda *args, **kwargs: {"exo": np.eye(4)},
    )
    monkeypatch.setattr(
        robot_exo,
        "_write_exo_artifacts",
        lambda **kwargs: {
            "status": "ready",
            "sequence_id": kwargs["sequence"],
        },
    )

    caller_root = tmp_path / "caller_data"
    _prepare_lifecycle_root(caller_root)
    result = _prepare_rh20t_exo(
        plan, caller_root, "ffmpeg", 1, False, source_archive
    )
    assert result[0]["status"] == "ready"
    assert source_archive.is_file()

    automatic_root = tmp_path / "automatic_data"
    _prepare_lifecycle_root(automatic_root)
    downloaded = automatic_root / "_cache" / "rh20t" / "archive.tar.gz"

    def fake_download(file_id, destination, expected_size, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_archive.read_bytes())
        return destination

    monkeypatch.setattr(robot_exo, "download_google_drive_ranges", fake_download)
    result = _prepare_rh20t_exo(plan, automatic_root, "ffmpeg", 1, False, None)
    assert result[0]["status"] == "ready"
    assert not downloaded.exists()
    assert source_archive.is_file()


def test_robot_root_manifest_merges_separate_dataset_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = {
        "profile": {"id": "merge"},
        "datasets": {
            "droid_wrist": {
                "clips": [{"sequence": "droid"}],
                "demo_exo": {"sequences": {"droid": {}}},
            },
            "rh20t_wrist": {
                "clips": [{"sequence": "rh20t"}],
                "demo_exo": {},
            },
        },
    }
    monkeypatch.setattr(
        robot_exo,
        "_prepare_droid_exo",
        lambda *args, **kwargs: [
            {"status": "ready", "sequence_id": "droid"}
        ],
    )
    monkeypatch.setattr(
        robot_exo,
        "_prepare_rh20t_exo",
        lambda *args, **kwargs: [
            {"status": "ready", "sequence_id": "rh20t"}
        ],
    )
    prepare_robot_exo(plan, tmp_path, "ffmpeg", datasets=("droid_wrist",))
    report = prepare_robot_exo(plan, tmp_path, "ffmpeg", datasets=("rh20t_wrist",))
    assert list(report["datasets"]) == ["droid_wrist", "rh20t_wrist"]
    assert [item["sequence_id"] for item in report["ready_clips"]] == [
        "droid",
        "rh20t",
    ]


def test_generic_download_invokes_robot_preparation_without_changing_clip_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []
    monkeypatch.setitem(DOWNLOADERS, "droid_wrist", lambda ctx: [{"base": "clip"}])
    monkeypatch.setattr(
        robot_exo,
        "prepare_robot_exo",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {},
    )
    context = SimpleNamespace(
        plan={"profile": {"id": "profile"}, "_plan_path": "plan.yaml"},
        data_root=tmp_path,
        target_fps=10,
        workers=2,
        robot_with_exo=True,
        ffmpeg="ffmpeg",
        keep_source=False,
        rh20t_archive=None,
    )
    manifest = execute_download(context, ("droid_wrist",))
    state = manifest["datasets"]["droid_wrist"]
    assert state["clips"] == [{"base": "clip"}]
    assert "robot_exo" not in state
    assert calls[0][1]["datasets"] == ("droid_wrist",)
