import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from ego_video_camera.eval_dataset_download import (
    _crop_egobody_gaze_member,
    _hot3d_data_groups,
    _index_egobody_color_archive,
    _normalize_incrowd_xyzw_trajectory,
    _parse_holoassist_intrinsics,
    _parse_holoassist_poses,
    _rh20t_camera_to_aligned_base,
    _rh20t_pose_matrix,
    _rh20t_stage_is_complete,
    _sample_egobody_clip,
    _sample_rh20t_indices,
    _sample_stera_indices,
    _sample_rows,
    _stera_camera_to_world,
    _parse_tum_image_list,
    _sample_droid_indices,
    _stage_rh20t_archive,
    download_google_drive_ranges,
    load_plan,
    main,
    plan_summary,
)
from ego_video_camera.http_archives import HttpObject, RemoteTar, RemoteZip


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "configs" / "ego_pose_eval_core65.yaml"
NATIVE_RGB_PLAN = REPO_ROOT / "configs" / "ego_pose_eval_native_rgb.yaml"
ROBOT_INTERACTION_PLAN = (
    REPO_ROOT / "configs" / "ego_pose_eval_robot_interaction_rgb.yaml"
)
RESOURCE_STATUS = REPO_ROOT / "configs" / "ego_pose_eval_resource_status.yaml"


class MemoryRangeClient:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.remote = HttpObject(
            "https://example.invalid/archive",
            len(payload),
            '"stable"',
            None,
        )

    def inspect(self, url):
        return self.remote

    def read(self, remote, start, end):
        assert remote == self.remote
        return self.payload[start : end + 1]

    def copy_range(self, remote, start, size, destination, chunk_size=4 * 1024**2):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload[start : start + size])
        return destination


def test_core_plan_is_exactly_112_clips_and_65_minutes():
    plan = load_plan(PLAN)
    summary = plan_summary(plan, plan["datasets"])
    assert summary["total_clips"] == 112
    assert summary["total_seconds"] == 3900
    assert summary["total_minutes"] == 65
    assert "egobody" in plan["datasets"]
    assert "kinpoly" not in plan["datasets"]
    egobody = plan["datasets"]["egobody"]
    assert egobody["clip_count"] == 20
    assert sum(clip["split"] == "val" for clip in egobody["clips"]) == 9
    assert len(
        {clip["hololens_sequence"] for clip in egobody["clips"]}
    ) == egobody["clip_count"]
    assert {
        stratum: sum(clip["stratum"] == stratum for clip in egobody["clips"])
        for stratum in {
            "low_motion",
            "moderate_motion",
            "locomotion",
            "fast_turn",
        }
    } == {
        "low_motion": 5,
        "moderate_motion": 5,
        "locomotion": 5,
        "fast_turn": 5,
    }


def test_native_rgb_plan_is_frozen_and_excludes_fisheye_inputs():
    plan = load_plan(NATIVE_RGB_PLAN)
    summary = plan_summary(plan, plan["_dataset_order"])
    assert plan["_dataset_order"] == (
        "tum_rgbd",
        "bonn_rgbd_dynamic",
        "openloris_office",
    )
    assert summary["total_clips"] == 12
    assert summary["total_seconds"] == 180
    assert summary["target_fps"] == 10
    selection = plan["datasets"]["openloris_office"]["camera_selection"]
    assert selection["camera_key"] == "d400_color_optical_frame"
    assert selection["projection_model"] == "pinhole"
    assert all("fisheye" in name for name in selection["excluded_streams"])


def test_robot_interaction_plan_is_frozen_and_uses_native_rgb_streams():
    plan = load_plan(ROBOT_INTERACTION_PLAN)
    summary = plan_summary(plan, plan["_dataset_order"])
    assert plan["_dataset_order"] == (
        "droid_wrist",
        "holoassist",
        "rh20t_wrist",
        "stera10m",
    )
    assert summary["total_clips"] == 16
    assert summary["total_seconds"] == 200
    assert summary["target_fps"] == 10
    assert plan["datasets"]["droid_wrist"]["camera"]["projection_model"] == "pinhole"
    assert all(
        clip["split"] == "test-v1_2"
        for clip in plan["datasets"]["holoassist"]["clips"]
    )
    rh20t = plan["datasets"]["rh20t_wrist"]
    assert [clip["sequence"] for clip in rh20t["clips"]] == [
        "task_0001_user_0016_scene_0001_cfg_0003",
        "task_0004_user_0016_scene_0004_cfg_0003",
        "task_0006_user_0016_scene_0007_cfg_0003",
        "task_0008_user_0016_scene_0010_cfg_0003",
    ]
    assert rh20t["archive_sha256"] == (
        "b49b297043f3ccf8386b620e11e9ccebc634ba5704e372ae7243480f6e38b6d3"
    )
    assert rh20t["camera"]["pose_direction"] == "camera_to_aligned_robot_base"
    assert rh20t["clips"][1]["start_s"] == 1
    assert "capture gap" in rh20t["clips"][1]["selection_note"]
    stera = plan["datasets"]["stera10m"]
    assert stera["revision"] == "548a1f26741647126e4a6347b29b46759e43ebb5"
    assert stera["camera"]["projection_model"] == "pinhole"
    assert stera["camera"]["source_pose_direction"] == (
        "camera_link_to_arkit_world"
    )
    assert [clip["sequence"] for clip in stera["clips"]] == [
        "session_data_20260416_084056",
        "session_data_20260425_142753",
        "session_data_20260427_182220",
        "session_data_20260414_144830",
    ]
    assert stera["clips"][1]["start_s"] == 69.5
    assert "timestamp pause" in stera["clips"][1]["selection_note"]
    assert all(
        set(clip["source_files"])
        == {
            "annotation.hdf5",
            "hierarchy.json",
            "rgb.mp4",
            "calibrations/meta.json",
            "calibrations/rgb_K.npy",
            "calibrations/rgb_D.npy",
            "calibrations/R_optical_to_link.npy",
        }
        for clip in stera["clips"]
    )


def test_resource_status_manifest_matches_frozen_resource_totals():
    status = yaml.safe_load(RESOURCE_STATUS.read_text(encoding="utf-8"))
    assert status["source_repositories"]["top_level_count"] == 13
    assert len(status["source_repositories"]["repositories"]) == 13
    groups = status["checkpoints"]["groups"]
    assert sum(group["complete"] for group in groups) == 88
    assert sum(group["gated_missing"] for group in groups) == 0
    datasets = {item["id"]: item for item in status["evaluation_data"]}
    assert datasets["ego_pose_eval_native_rgb_v1"]["frames"] == 1800
    assert datasets["ego_pose_eval_robot_interaction_rgb_v2"]["frames"] == 2000
    assert {item["status"] for item in status["restricted_resources"]} == {
        "application-pending",
    }
    assert {item["status"] for item in status["approved_resources"]} == {
        "evaluation-subset-complete",
        "complete",
    }


def test_download_dry_run_does_not_create_data_root(tmp_path: Path):
    destination = tmp_path / "must-not-exist"
    assert main(
        [
            "download",
            "--dry-run",
            "--plan",
            str(PLAN),
            "--data-root",
            str(destination),
        ]
    ) == 0
    assert not destination.exists()


def test_timestamp_sampler_returns_ten_hz_window():
    rows = [(float(index * 100_000_000), f"{index}.png") for index in range(1000)]
    sampled = _sample_rows(rows, start_s=10, duration_s=30, fps=10)
    assert len(sampled) == 300
    assert sampled[0][1] == "100.png"
    assert sampled[-1][1] == "399.png"


def test_timestamp_sampler_keeps_unix_seconds_as_seconds():
    origin = 1_548_339_819.87426
    rows = [(origin + index / 30, f"{index}.png") for index in range(900)]
    sampled = _sample_rows(rows, start_s=2, duration_s=15, fps=10)
    assert len(sampled) == 150
    assert sampled[0][1] == "60.png"
    assert sampled[-1][1] == "507.png"


def test_droid_sampler_uses_capture_clock_and_returns_source_indices():
    origin_ms = 1_696_720_326_695
    timestamps = [origin_ms + round(index * 1000 / 15) for index in range(300)]
    sampled = _sample_droid_indices(timestamps, start_s=2, duration_s=5, fps=10)
    assert len(sampled) == 50
    assert sampled[0][1] == 30
    assert sampled[-1][1] == 104
    assert sampled[-1][0] - sampled[0][0] >= 4_800


def test_rh20t_sampler_uses_millisecond_clock_and_source_indices():
    origin_ms = 1_630_000_000_000
    timestamps = [origin_ms + round(index * 1000 / 30) for index in range(600)]
    sampled = _sample_rh20t_indices(
        timestamps, start_s=2, duration_s=15, fps=10
    )
    assert len(sampled) == 150
    assert sampled[0][1] == 60
    assert sampled[-1][1] == 507


def test_rh20t_sampler_explicitly_repeats_slow_native_frames():
    origin_ms = 1_630_000_000_000
    timestamps = [origin_ms + 125 * index for index in range(100)]
    sampled = _sample_rh20t_indices(
        timestamps, start_s=1, duration_s=1, fps=10
    )
    source_indices = [source_index for _, source_index in sampled]
    assert len(sampled) == 10
    assert len(set(source_indices)) == 8
    assert source_indices == sorted(source_indices)


def test_stera_sampler_uses_constant_mp4_frame_index():
    sampled = _sample_stera_indices(
        source_frames=1424,
        start_s=69.5,
        duration_s=15,
        source_fps=15,
        target_fps=10,
    )
    assert len(sampled) == 150
    assert sampled[0][1] == 1043
    assert sampled[-1][1] == 1266
    assert len({source_index for _, source_index in sampled}) == 150


def test_stera_pose_is_converted_from_link_to_rgb_optical_frame():
    rotation_link_to_world = np.eye(3)
    translation_world = np.array([1.0, 2.0, 3.0])
    rotation_optical_to_link = np.diag([1.0, -1.0, -1.0])
    result = _stera_camera_to_world(
        rotation_link_to_world,
        translation_world,
        rotation_optical_to_link,
    )
    np.testing.assert_allclose(result[:3, :3], rotation_optical_to_link)
    np.testing.assert_allclose(result[:3, 3], translation_world)
    np.testing.assert_allclose(result[3], [0, 0, 0, 1])


def test_rh20t_hand_eye_transform_outputs_camera_to_aligned_base():
    pose = [1, 2, 3, 1, 0, 0, 0]
    assert np.allclose(
        _rh20t_pose_matrix(pose),
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 2],
                [0, 0, 1, 3],
                [0, 0, 0, 1],
            ]
        ),
    )
    camera = {
        "tcp_camera_matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        "align_tcp_matrix": np.eye(4).tolist(),
    }
    result = _rh20t_camera_to_aligned_base(pose, camera)
    assert np.allclose(result[:3, 3], [1, 2, 2])
    assert np.allclose(result[:3, :3], np.eye(3))


def test_google_drive_mirror_and_hash_inputs_are_gated(tmp_path: Path):
    with pytest.raises(Exception, match="mirror must use HTTPS"):
        download_google_drive_ranges(
            "file-id",
            tmp_path / "archive.tar.gz",
            expected_size=10,
            mirror_url="http://example.invalid/archive.tar.gz",
        )
    with pytest.raises(Exception, match="64 hex digits"):
        download_google_drive_ranges(
            "file-id",
            tmp_path / "archive.tar.gz",
            expected_size=10,
            mirror_url="https://example.invalid/archive.tar.gz",
            expected_sha256="bad",
        )
    with pytest.raises(Exception, match="64 hex digits"):
        download_google_drive_ranges(
            "file-id",
            tmp_path / "archive.tar.gz",
            expected_size=10,
            mirror_url="https://example.invalid/archive.tar.gz",
            expected_sha256="g" * 64,
        )


def test_holoassist_pose_and_intrinsics_parsers_preserve_official_semantics():
    matrix = "1 0 0 0.1 0 1 0 0.2 0 0 1 0.3 0 0 0 1"
    poses = _parse_holoassist_poses(
        (
            f"0.0 637928128674523981 {matrix}\n"
            f"0.1 637928128675523981 {matrix}\n"
            f"0.2 637928128675523981 {matrix}\n"
        ).encode()
    )
    assert poses[0][2][3:12:4] == [0.1, 0.2, 0.3]
    assert poses[1][0] == 0.1
    assert poses[2][1] == poses[1][1]
    intrinsics = _parse_holoassist_intrinsics(
        (
            "680.4 0 445.4 0 681.4 237.4 0 0 1 "
            "0 0 0 0 0 0 0 0 680.9 680.4 681.4 445.4 237.4 1 896 504\n"
        ).encode()
    )
    assert intrinsics["fx"] == 680.4
    assert intrinsics["source_width"] == 896
    assert intrinsics["closed_form_distorts"] is True


def test_tum_image_list_parser_rejects_parent_traversal():
    assert _parse_tum_image_list(
        "# timestamp filename\n1.0 rgb/1.png\n1.1 rgb/2.png\n"
    ) == [(1.0, "rgb/1.png"), (1.1, "rgb/2.png")]
    with pytest.raises(Exception, match="Unsafe image-list path"):
        _parse_tum_image_list("1.0 ../escape.png\n")


def test_egobody_color_index_and_frozen_sampler(tmp_path: Path):
    recording = "recording_test"
    hololens = "2026-01-01-000000"
    archive_path = tmp_path / "egocentric_color.zip"
    matrix = "1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1"
    pv_lines = ["(10.0, 20.0, 640, 480)"]
    with zipfile.ZipFile(archive_path, "w") as archive:
        pv_member = (
            f"egocentric_color/{recording}/{hololens}/{hololens}_pv.txt"
        )
        for index in range(11):
            timestamp = 1_000_000_000 + index * 1_000_000
            pv_lines.append(f"{timestamp},500,501,{matrix}")
            archive.writestr(
                f"egocentric_color/{recording}/{hololens}/PV/"
                f"{timestamp}_frame_{100 + index:05d}.jpg",
                b"jpeg",
            )
        archive.writestr(pv_member, "\n".join(pv_lines) + "\n")
    pv_members, images = _index_egobody_color_archive(
        archive_path, {(recording, hololens)}
    )
    assert pv_members[(recording, hololens)] == pv_member
    pv_path = tmp_path / "pv.txt"
    pv_path.write_text("\n".join(pv_lines) + "\n", encoding="utf-8")
    calibration, sampled = _sample_egobody_clip(
        pv_path,
        {
            "sequence": recording,
            "hololens_sequence": hololens,
            "start_s": 0,
            "duration_s": 1,
            "recording_start_frame": 100,
            "recording_end_frame": 110,
            "expected_start_frame": 100,
            "expected_start_timestamp": 1_000_000_000,
        },
        images,
        target_fps=10,
    )
    assert calibration.width == 640
    assert len(sampled) == 10
    assert sampled[-1].frame_id == 109


def test_egobody_gaze_is_cropped_to_requested_window(tmp_path: Path):
    archive_path = tmp_path / "gaze.zip"
    member = (
        "egocentric_gaze/recording_test/sequence/"
        "sequence_head_hand_eye.csv"
    )
    rows = b"".join(
        f"{timestamp},1,0,0\n".encode()
        for timestamp in (1_000_000, 10_000_000, 15_000_000, 30_000_000)
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, rows)
    destination = _crop_egobody_gaze_member(
        archive_path,
        member,
        tmp_path / "head.csv",
        start_timestamp=10_000_000,
        end_timestamp=15_000_000,
    )
    timestamps = [
        int(line.split(",", 1)[0])
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert timestamps == [10_000_000, 15_000_000]


def test_remote_zip_reads_only_index_and_selected_member(tmp_path: Path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sequence/mav0/cam0/data.csv", b"1,1.png\n")
        archive.writestr("sequence/mav0/cam0/data/1.png", b"fake-png")
    client = MemoryRangeClient(buffer.getvalue())
    archive = RemoteZip(
        "https://example.invalid/archive.zip", tmp_path / "cache", client=client
    )
    assert "sequence/mav0/cam0/data.csv" in archive.names
    assert archive.read("sequence/mav0/cam0/data/1.png") == b"fake-png"


def test_remote_tar_scans_pax_archive_and_extracts_member(tmp_path: Path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        payload = b"trajectory"
        info = tarfile.TarInfo("sequence.gt_trajectory.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    client = MemoryRangeClient(buffer.getvalue())
    archive = RemoteTar("https://example.invalid/shard.tar", client=client)
    entries = archive.scan(stop_suffixes=(".gt_trajectory.txt",))
    member = entries[-1]
    assert member.name == "sequence.gt_trajectory.txt"
    destination = archive.extract(member, tmp_path / "trajectory.txt")
    assert destination.read_bytes() == b"trajectory"


def test_remote_tar_indexes_concatenated_streams_and_reuses_cache(tmp_path: Path):
    parts = []
    for name, payload in (("first.txt", b"first"), ("second.txt", b"second")):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        parts.append(buffer.getvalue())
    client = MemoryRangeClient(b"".join(parts))
    archive = RemoteTar("https://example.invalid/concatenated.tar", client=client)
    entries = archive.index(tmp_path / "index", checkpoint_members=1)
    assert [entry.name for entry in entries] == ["first.txt", "second.txt"]
    assert archive.index(tmp_path / "index") == entries


def test_remote_tar_partial_index_stops_after_required_member(tmp_path: Path):
    parts = []
    for name in ("required.txt", "unrelated.txt"):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            payload = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        parts.append(buffer.getvalue())
    client = MemoryRangeClient(b"".join(parts))
    archive = RemoteTar("https://example.invalid/partial.tar", client=client)
    entries = archive.index(
        tmp_path / "partial-index", required_names=("required.txt",)
    )
    assert [entry.name for entry in entries] == ["required.txt"]
    assert archive.index(tmp_path / "partial-index")[-1].name == "unrelated.txt"


def test_rh20t_stream_stage_extracts_only_selected_scene_and_calibration(
    tmp_path: Path,
):
    dataset = {
        "archive_root": "RH20T_cfg3",
        "in_hand_serial": "camera123",
        "clips": [{"sequence": "selected_scene"}],
    }
    archive_path = tmp_path / "RH20T_cfg3.tar.gz"
    wanted = {
        "RH20T_cfg3/selected_scene/metadata.json": b'{"calib": 42}',
        "RH20T_cfg3/selected_scene/cam_camera123/color.mp4": b"video",
        "RH20T_cfg3/selected_scene/cam_camera123/timestamps.npy": b"timestamps",
        "RH20T_cfg3/selected_scene/transformed/tcp_base.npy": b"tcp",
        "RH20T_cfg3/calib/42/devices.npy": b"devices",
        "RH20T_cfg3/calib/42/extrinsics.npy": b"extrinsics",
        "RH20T_cfg3/calib/42/intrinsics.npy": b"intrinsics",
        "RH20T_cfg3/calib/42/tcp.npy": b"calibration-tcp",
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in {
            **wanted,
            "RH20T_cfg3/unselected_scene/cam_camera123/color.mp4": b"skip",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    data_root = tmp_path / "data"
    stage_root = data_root / "_cache" / "rh20t" / "extracted"
    extracted_root = _stage_rh20t_archive(
        archive_path, stage_root, dataset, data_root
    )
    assert extracted_root == stage_root / "RH20T_cfg3"
    assert _rh20t_stage_is_complete(stage_root, dataset)
    assert all(
        (stage_root / name).read_bytes() == payload
        for name, payload in wanted.items()
    )
    assert not (stage_root / "RH20T_cfg3/unselected_scene").exists()


def test_incrowd_xyzw_trajectory_is_normalized_to_wxyz(tmp_path: Path):
    source = tmp_path / "trj_gt_sec_xyzw.txt"
    source.write_text(
        "# tracking_timestamp_us tx ty tz qx qy qz qw\n"
        "762.327171 1 2 3 0.1 0.2 0.3 0.9\n",
        encoding="utf-8",
    )
    destination = _normalize_incrowd_xyzw_trajectory(
        source, tmp_path / "trj_gt_sec_wxyz.txt"
    )
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# tracking_timestamp_sec")
    assert lines[1] == "762.327171 1 2 3 0.9 0.1 0.2 0.3"


def test_hot3d_group_order_matches_official_downloader(tmp_path: Path):
    path = tmp_path / "cdn.json"
    path.write_text(
        json.dumps(
            {
                "sequence_config": {
                    "main": {"recording": "recording.vrs", "mps": "mps.zip"},
                    "data_groups": {
                        "metadata": ["metadata.json"],
                        "hand_annotations": ["hands.jsonl"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    assert _hot3d_data_groups(path) == [
        "main_vrs",
        "mps_slam_trajectories",
        "mps_slam_calibration",
        "mps_slam_points",
        "mps_slam_summary",
        "mps_eye_gaze",
        "metadata",
        "hand_annotations",
    ]
