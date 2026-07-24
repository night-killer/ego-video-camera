import io
import json
import tarfile
import zipfile
from pathlib import Path

from ego_video_camera.eval_dataset_download import (
    _crop_egobody_gaze_member,
    _hot3d_data_groups,
    _index_egobody_color_archive,
    _normalize_incrowd_xyzw_trajectory,
    _sample_egobody_clip,
    _sample_rows,
    load_plan,
    main,
    plan_summary,
)
from ego_video_camera.http_archives import HttpObject, RemoteTar, RemoteZip


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "configs" / "ego_pose_eval_core65.yaml"


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
