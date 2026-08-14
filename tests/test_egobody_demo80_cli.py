import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ego_video_camera.egobody_demo80_cli as demo80_cli
from ego_video_camera.egobody_demo80_cli import (
    LoadedClip,
    _read_frame_rows,
    _run_summary_path,
    load_demo_clip,
    preflight_run,
    select_clips,
    validate_clip,
)


def _pose_line(timestamp: int) -> str:
    pose = np.eye(4).reshape(-1)
    values = [str(timestamp), "1000", "1000", *(f"{value:g}" for value in pose)]
    return ",".join(values)


def _write_clip(root: Path) -> dict:
    clip = {
        "clip_id": "DESK_001",
        "category": "desktop_head_motion",
        "recording_name": "recording",
        "hololens_sequence": "sequence",
        "duration_s": 0.25,
    }
    clip_root = root / "clips" / clip["clip_id"]
    (clip_root / "frames").mkdir(parents=True)
    (clip_root / "reference").mkdir()
    (clip_root / "clip.json").write_text(
        json.dumps({**clip, "sample_fps": 8, "frame_count": 2, "exo_frame_count": 0}),
        encoding="utf-8",
    )
    with (clip_root / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("output_index", "source_frame_id", "source_timestamp", "filename"))
        writer.writerow((0, 10, 100, "000000_000010.jpg"))
        writer.writerow((1, 14, 200, "000001_000014.jpg"))
    for name in ("000000_000010.jpg", "000001_000014.jpg"):
        (clip_root / "frames" / name).write_bytes(b"jpeg")
    (clip_root / "reference" / "sequence_pv.txt").write_text(
        "960,540,1920,1080\n" + _pose_line(100) + "\n" + _pose_line(200) + "\n",
        encoding="utf-8",
    )
    return clip


def test_clip_selection_supports_ids_categories_and_all():
    clips = [
        {"clip_id": "DESK_001", "category": "desktop_head_motion"},
        {"clip_id": "WALK_001", "category": "walking_person"},
    ]
    assert [item["clip_id"] for item in select_clips(clips, clip_ids=["DESK_001"])] == [
        "DESK_001"
    ]
    assert [item["clip_id"] for item in select_clips(clips, category="walking")] == [
        "WALK_001"
    ]
    assert len(select_clips(clips, run_all=True)) == 2
    with pytest.raises(ValueError, match="Unknown clip"):
        select_clips(clips, clip_ids=["MISSING"])


def test_clip_local_loader_follows_frames_csv_and_allows_missing_exo(tmp_path: Path):
    clip = _write_clip(tmp_path)
    loaded = load_demo_clip(tmp_path, clip)
    assert [record.frame_id for record in loaded.records] == [10, 14]
    assert [record.timestamp for record in loaded.records] == [100, 200]
    assert [mapping.ego_image.name for mapping in loaded.mappings] == [
        "000000_000010.jpg",
        "000001_000014.jpg",
    ]
    assert all(mapping.exo_image is None for mapping in loaded.mappings)
    assert loaded.clip["runtime_sample_fps"] == 8


def test_validate_requires_exo_when_requested(tmp_path: Path):
    clip = _write_clip(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing 2/2 exo frames"):
        validate_clip(tmp_path, clip, require_exo=True, decode_images=False)


def test_frame_csv_rejects_non_monotonic_source_order(tmp_path: Path):
    path = tmp_path / "frames.csv"
    path.write_text(
        "output_index,source_frame_id,source_timestamp,filename\n"
        "0,10,200,a.jpg\n"
        "1,14,100,b.jpg\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timestamp must be strictly increasing"):
        _read_frame_rows(path)


def test_run_summary_path_defaults_to_output_root_and_supports_worker_paths(tmp_path: Path):
    output_root = tmp_path / "output"
    assert _run_summary_path(None, output_root) == output_root / "run_summary.json"
    assert _run_summary_path("launcher_logs/gpu0.json", output_root) == (
        output_root / "launcher_logs/gpu0.json"
    ).resolve()
    absolute = tmp_path / "summaries/gpu1.json"
    assert _run_summary_path(str(absolute), output_root) == absolute.resolve()


def _preflight_config(tmp_path: Path) -> dict:
    executable = tmp_path / "tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    return {
        "da3": {
            "checkpoint_path": str(tmp_path / "missing-checkpoint"),
            "source_root": str(tmp_path / "missing-source"),
        },
        "runtime": {
            "ffmpeg_path": str(executable),
            "ffprobe_path": str(executable),
        },
    }


def test_preflight_skips_model_source_and_cuda_checks_without_inference(
    monkeypatch, tmp_path: Path
):
    source_clip = {"clip_id": "DESK_001"}
    loaded = LoadedClip(
        clip={"clip_id": "DESK_001", "recording_name": "recording"},
        records=[],
        mappings=[],
    )
    monkeypatch.setattr(demo80_cli, "load_master_camera", lambda root: object())
    monkeypatch.setattr(demo80_cli, "load_demo_clip", lambda root, clip: loaded)
    monkeypatch.setattr(demo80_cli, "load_T_K_W", lambda root, recording: np.eye(4))
    monkeypatch.setattr(
        demo80_cli.torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA should not be inspected for --no-run-da3"),
    )

    result = preflight_run(
        root=tmp_path,
        data_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        metadata_root=tmp_path,
        selected=[source_clip],
        config=_preflight_config(tmp_path),
        run_da3=False,
    )

    assert result == [loaded]


def test_preflight_requires_model_source_and_cuda_for_inference(monkeypatch, tmp_path: Path):
    loaded = LoadedClip(
        clip={"clip_id": "DESK_001", "recording_name": "recording"},
        records=[],
        mappings=[],
    )
    monkeypatch.setattr(demo80_cli, "load_master_camera", lambda root: object())
    monkeypatch.setattr(demo80_cli, "load_demo_clip", lambda root, clip: loaded)
    monkeypatch.setattr(demo80_cli, "load_T_K_W", lambda root, recording: np.eye(4))
    monkeypatch.setattr(demo80_cli.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as raised:
        preflight_run(
            root=tmp_path,
            data_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            metadata_root=tmp_path,
            selected=[{"clip_id": "DESK_001"}],
            config=_preflight_config(tmp_path),
            run_da3=True,
        )

    message = str(raised.value)
    assert "Missing DA3 checkpoint file" in message
    assert "Missing DA3 source package" in message
    assert "CUDA is unavailable" in message
    assert "For missing exo frames" not in message


def test_preflight_recovery_command_uses_effective_data_root_and_manifest(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "selected data"
    manifest_path = tmp_path / "selected manifest.json"
    loaded = LoadedClip(
        clip={"clip_id": "DESK_001", "recording_name": "recording"},
        records=[],
        mappings=[SimpleNamespace(exo_image=None)],
    )
    monkeypatch.setattr(demo80_cli, "load_master_camera", lambda root: object())
    monkeypatch.setattr(demo80_cli, "load_demo_clip", lambda root, clip: loaded)
    monkeypatch.setattr(demo80_cli, "load_T_K_W", lambda root, recording: np.eye(4))

    with pytest.raises(RuntimeError) as raised:
        preflight_run(
            root=tmp_path,
            data_root=data_root,
            manifest_path=manifest_path,
            metadata_root=tmp_path,
            selected=[{"clip_id": "DESK_001"}],
            config=_preflight_config(tmp_path),
            run_da3=False,
        )

    message = str(raised.value)
    assert f"--data-root '{data_root}'" in message
    assert f"--manifest '{manifest_path}'" in message
    assert f"--netrc-file {demo80_cli.DEFAULT_NETRC}" in message
