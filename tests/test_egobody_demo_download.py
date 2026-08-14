import json
import zipfile
from pathlib import Path

from ego_video_camera.egobody_demo_download import (
    ImageMember,
    SequenceMembers,
    _sample_images,
    download_exo_only,
    download_manifest,
    index_color_archive,
    validate_manifest,
)


def _sequence_with_ids(ids):
    return SequenceMembers(
        recording="rec",
        sequence="seq",
        pv_member="release/egocentric_color/rec/seq/seq_pv.txt",
        images=tuple(ImageMember(frame_id=value, timestamp=None, name=f"{value}.jpg") for value in ids),
    )


def test_sampling_is_monotonic_for_dense_source_ids():
    selected = _sample_images(_sequence_with_ids(range(30)), 0, 30, 8)
    ids = [item.frame_id for item in selected]
    assert ids == [0, 4, 7, 11, 15, 19, 22, 26]
    assert all(first < second for first, second in zip(ids, ids[1:]))


def test_sampling_handles_short_gaps_without_rewinding():
    selected = _sample_images(_sequence_with_ids([value for value in range(30) if value not in {7, 8}]), 0, 30, 8)
    ids = [item.frame_id for item in selected]
    assert ids == [0, 4, 6, 11, 15, 19, 22, 26]
    assert all(first < second for first, second in zip(ids, ids[1:]))


def test_sampling_rejects_long_gaps_even_when_frame_count_is_sufficient():
    sequence = _sequence_with_ids([0, 1, 2, 3, 4, *range(20, 30)])
    try:
        _sample_images(sequence, 0, 30, 8)
    except ValueError as error:
        assert "sampling gap" in str(error)
    else:
        raise AssertionError("long source gaps must not be hidden by nearest-frame reuse")


def _write_color_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        root = "release/egocentric_color/rec/2024-01-01-000000"
        archive.writestr(f"{root}/{Path(root).name}_pv.txt", "(1, 2, 3, 4)\n")
        for frame in range(0, 30):
            archive.writestr(
                f"{root}/PV/100000000000000000_frame_{frame:05d}.jpg",
                b"jpeg",
            )


def test_color_index_reads_recording_sequence_and_frames(tmp_path: Path):
    archive = tmp_path / "color.zip"
    _write_color_zip(archive)
    indexed = index_color_archive(archive)
    assert list(indexed) == ["rec"]
    sequence = indexed["rec"][0]
    assert sequence.sequence == "2024-01-01-000000"
    assert sequence.images[0].frame_id == 0
    assert sequence.images[-1].frame_id == 29


def test_manifest_validation_rejects_overlapping_windows():
    base = {
        "schema_version": "egobody_demo_selection_v1",
        "categories": {
            "desktop_head_motion": {"clips": []},
            "walking_person": {"clips": []},
        },
    }
    for category, clip_id in (("desktop_head_motion", "D"), ("walking_person", "W")):
        base["categories"][category]["clips"].append(
            {
                "clip_id": clip_id,
                "recording_name": "rec",
                "frame_start_inclusive": 0,
                "frame_end_exclusive": 10,
                "frame_count": 10,
            }
        )
    try:
        validate_manifest(base, desktop_count=1, walking_count=1)
    except ValueError as error:
        assert "Duplicate RGB time window" in str(error) or "Overlapping" in str(error)
    else:
        raise AssertionError("overlapping windows must be rejected")


def test_local_archive_download_writes_clip_outputs(tmp_path: Path):
    archive_path = tmp_path / "color.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for recording, sequence, frame_count in (("desk", "desk-seq", 600), ("walk", "walk-seq", 180)):
            prefix = f"release/egocentric_color/{recording}/{sequence}"
            archive.writestr(f"{prefix}/{sequence}_pv.txt", "944,505,1920,1080\n")
            for frame in range(frame_count):
                archive.writestr(
                    f"{prefix}/PV/100000000000000000_frame_{frame:05d}.jpg",
                    b"jpeg-payload",
                )
    manifest = {
        "schema_version": "egobody_demo_selection_v1",
        "categories": {
            "desktop_head_motion": {"clips": [{
                "clip_id": "DESK_001", "recording_name": "desk",
                "frame_start_inclusive": 0, "frame_end_exclusive": 600,
                "frame_count": 600,
            }]},
            "walking_person": {"clips": [{
                "clip_id": "WALK_001", "recording_name": "walk",
                "frame_start_inclusive": 0, "frame_end_exclusive": 180,
                "frame_count": 180,
            }]},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = download_manifest(
        manifest_path,
        tmp_path / "out",
        ego_archive=archive_path,
        sample_fps=8,
        accept_license=True,
    )
    assert len(result["clips"]) == 2
    assert len(list((tmp_path / "out/clips/DESK_001/frames").glob("*.jpg"))) == 160
    assert len(list((tmp_path / "out/clips/WALK_001/frames").glob("*.jpg"))) == 48
    assert json.loads((tmp_path / "out/clips/DESK_001/clip.json").read_text())["hololens_sequence"] == "desk-seq"
    download_record = json.loads((tmp_path / "out/download_manifest.json").read_text())
    assert len(download_record["manifest_sha256"]) == 64


def test_exo_only_download_uses_existing_frame_csv(tmp_path: Path):
    data_root = tmp_path / "data"
    clip_root = data_root / "clips" / "DESK_001"
    clip_root.mkdir(parents=True)
    clip = {
        "clip_id": "DESK_001",
        "category": "desktop_head_motion",
        "recording_name": "rec",
        "frame_count": 2,
        "exo_frame_count": 0,
    }
    (clip_root / "clip.json").write_text(json.dumps(clip), encoding="utf-8")
    (data_root / "download_manifest.json").write_text(
        json.dumps({"schema_version": "egobody_demo_download_v1", "clips": [clip]}),
        encoding="utf-8",
    )
    (clip_root / "frames.csv").write_text(
        "output_index,source_frame_id,source_timestamp,filename\n"
        "0,10,100,a.jpg\n"
        "1,20,200,b.jpg\n",
        encoding="utf-8",
    )
    archive = tmp_path / "exo.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("release/kinect_color/rec/master/frame_00010.jpg", b"ten")
        handle.writestr("release/kinect_color/rec/master/frame_00020.jpg", b"twenty")
    result = download_exo_only(data_root, exo_archive=archive, require_all=True)
    assert result["materialized_frame_count"] == 2
    assert result["missing_frame_count"] == 0
    assert sorted(path.name for path in (clip_root / "exo_frames").glob("*.jpg")) == [
        "000000_000010.jpg",
        "000001_000020.jpg",
    ]
    assert json.loads((clip_root / "clip.json").read_text())["exo_frame_count"] == 2
