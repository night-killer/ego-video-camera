import csv
import inspect
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import ego_video_camera.robot_pipeline as robot_pipeline
import ego_video_camera.robot_io as robot_io
from ego_video_camera.camera_models import CameraModel
from ego_video_camera.da3_adapter import EXPECTED_DA3_COMMIT
from ego_video_camera.eval_dataset_download import load_plan
from ego_video_camera.robot_cli import _format_exo_not_ready, _resolved_config, parser
from ego_video_camera.robot_commands import generate_robot_gpu_commands
from ego_video_camera.robot_io import (
    ACTIVE_EGO_MODEL_LABEL,
    RobotClip,
    RobotFrame,
    load_robot_clip,
    robot_demo_selection,
    robot_exo_readiness,
    subsample_indices,
)
from ego_video_camera.robot_mock_pipeline import run_robot_mock_pipeline
from ego_video_camera.robot_pipeline import (
    _robot_da3_resume_matches,
    load_robot_da3_poses,
    postprocess_robot_da3,
    run_robot_da3,
)
from ego_video_camera.serialization import read_json, write_json
from ego_video_camera.transforms import Sim3
from ego_video_camera.visualization import semantic_pose_directions


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = "/data/aigc/cyb/zxgu/env/worldsearcher/bin/ffmpeg"
FFPROBE = "/data/aigc/cyb/zxgu/env/worldsearcher/bin/ffprobe"


def _robot_clip(tmp_path: Path, count: int = 3) -> RobotClip:
    frames = []
    for index in range(count):
        pose = np.eye(4)
        pose[:3, 3] = [0.1 * index, 0.02 * index * index, 2.0]
        frames.append(
            RobotFrame(
                output_index=index * 2,
                timeline_sec=index / 5.0,
                ego_timestamp_ms=1000 + index * 200,
                ego_image=tmp_path / f"ego_{index}.png",
                reference_from_ego=pose,
                exo_timestamp_ms=1000 + index * 200,
                sync_delta_ms=0,
                exo_source_frame_index=index,
                exo_image=tmp_path / f"exo_{index}.png",
                reference_from_exo=np.eye(4),
                synchronized=True,
            )
        )
    return RobotClip(
        dataset="droid_wrist",
        sequence_id="synthetic",
        clip_dir=tmp_path,
        reference_type="robot_kinematic_camera_to_base",
        source_fps=10.0,
        fps=5.0,
        duration_sec=count / 5.0,
        frames=tuple(frames),
        exo_camera=CameraModel(
            np.asarray([[100, 0, 50], [0, 100, 50], [0, 0, 1.0]]),
            np.zeros(5),
            100,
            100,
        ),
        exo_manifest={"quality": {}},
    )


def _da3_config(checkpoint: Path) -> dict:
    return {
        "da3": {
            "source_root": "thirdparty/Depth-Anything-3",
            "checkpoint_path": str(checkpoint),
            "input_resolution": 504,
            "window_size": 60,
            "window_overlap": 30,
            "confidence_threshold": 1.5,
        }
    }


def test_opencv_semantic_axes_are_right_up_and_gaze():
    right, up, gaze = semantic_pose_directions(np.eye(3), "opencv_camera")
    np.testing.assert_array_equal(right, [1, 0, 0])
    np.testing.assert_array_equal(up, [0, -1, 0])
    np.testing.assert_array_equal(gaze, [0, 0, 1])


def test_robot_demo_uses_active_ego_foundation_model_display_label():
    assert ACTIVE_EGO_MODEL_LABEL == "Active Ego Foundation Model"
    width = cv2.getTextSize(
        ACTIVE_EGO_MODEL_LABEL, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )[0][0]
    assert width < 900


def test_robot_da3_loader_keeps_official_c2w_unchanged(tmp_path: Path):
    pose = np.eye(4)[None]
    pose[0, :3, :3] = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pose[0, :3, 3] = [1, 2, 3]
    np.savez(
        tmp_path / "da3_poses_raw.npz",
        c2w=pose,
        confidence=[2.0],
        frame_ids=[0],
        timestamps=[1000],
    )
    loaded = load_robot_da3_poses(tmp_path)
    np.testing.assert_array_equal(loaded["c2w"], pose)
    interpretation = read_json(tmp_path / "pose_basis_interpretation.json")
    assert interpretation["camera_basis_change_applied"] is False
    assert interpretation["source"] == "da3_poses_raw.npz[c2w]"


def test_robot_inference_receives_only_ego_rgb_and_native_opencv_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clip = _robot_clip(tmp_path)
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(robot_pipeline, "run_da3_streaming", fake_run)
    run_robot_da3(
        repo_root=ROOT,
        clip=clip,
        output_dir=tmp_path / "output",
        config=_da3_config(tmp_path / "checkpoint"),
    )
    assert captured["image_paths"] == clip.ego_images
    assert captured["output_pose_basis"] == "opencv"
    assert "extrinsics" not in captured
    assert "intrinsics" not in captured
    assert all("exo" not in str(path.name) for path in captured["image_paths"])
    source = inspect.getsource(run_robot_da3)
    inference_call = source[source.index("run_da3_streaming(") :]
    assert "reference_from" not in inference_call


def test_robot_resume_requires_matching_native_basis_config(tmp_path: Path):
    clip = _robot_clip(tmp_path, count=2)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"weight")
    da3_dir = tmp_path / "da3"
    da3_dir.mkdir()
    np.savez(
        da3_dir / "da3_poses_raw.npz",
        frame_ids=clip.frame_ids,
        timestamps=clip.timestamps_ms,
    )
    write_json(da3_dir / "da3_poses_raw.json", {"complete": True})
    write_json(
        da3_dir / "da3_resolved_config.json",
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
            "output_pose_basis": "opencv",
            "input_count": 2,
        },
    )
    config = _da3_config(checkpoint)
    assert _robot_da3_resume_matches(da3_dir, clip, config)
    resolved = read_json(da3_dir / "da3_resolved_config.json")
    resolved["output_pose_basis"] = "egobody_pv"
    write_json(da3_dir / "da3_resolved_config.json", resolved)
    assert not _robot_da3_resume_matches(da3_dir, clip, config)


def test_robot_sampling_supports_deterministic_10_to_5_fps():
    np.testing.assert_array_equal(subsample_indices(10, 10, 5), [0, 2, 4, 6, 8])
    with pytest.raises(ValueError, match="upsample"):
        subsample_indices(10, 10, 20)


def test_robot_exo_preflight_reports_all_missing_clips_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    selections = [
        {"dataset": "droid_wrist", "sequence_id": "ready"},
        {"dataset": "rh20t_wrist", "sequence_id": "missing_a"},
        {"dataset": "rh20t_wrist", "sequence_id": "missing_b"},
    ]

    def fake_status(path: Path):
        if path.name == "ready":
            return "ready", {"status": "ready"}, None
        return "missing", None, "exo artifacts are missing"

    monkeypatch.setattr(robot_io, "exo_clip_status", fake_status)
    report = robot_exo_readiness(tmp_path, selections)
    assert not report["ok"]
    assert [item["sequence_id"] for item in report["missing_clips"]] == [
        "missing_a",
        "missing_b",
    ]
    message = _format_exo_not_ready(
        report,
        config_path=tmp_path / "config.yaml",
        plan_path=tmp_path / "plan.yaml",
        data_root=tmp_path,
        output_root=tmp_path / "output",
    )
    assert "DA3 was not started" in message
    assert "--dataset rh20t --prepare-exo" in message
    assert "27.4 GB RH20T cfg3 archive" in message
    assert "--rh20t-archive /path/to/RH20T_cfg3.tar.gz" in message


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_robot_clip_reader_keeps_mapping_and_subsamples_without_interpolation(
    tmp_path: Path,
):
    sequence = "synthetic"
    clip_dir = tmp_path / "droid_wrist" / "clips" / sequence
    (clip_dir / "frames").mkdir(parents=True)
    (clip_dir / "exo" / "frames").mkdir(parents=True)
    ego_rows = []
    exo_rows = []
    exo_pose_rows = []
    reference_rows = []
    matrix_fields = [f"m{i}{j}" for i in range(4) for j in range(4)]
    for index in range(4):
        ego_name = f"ego_{index}.png"
        exo_name = f"exo_{index}.png"
        cv2.imwrite(str(clip_dir / "frames" / ego_name), np.zeros((40, 60, 3), np.uint8))
        cv2.imwrite(
            str(clip_dir / "exo" / "frames" / exo_name),
            np.zeros((50, 100, 3), np.uint8),
        )
        ego_rows.append(
            {
                "output_index": index,
                "source_timestamp": 1000 + index * 100,
                "filename": ego_name,
            }
        )
        exo_rows.append(
            {
                "output_index": index,
                "ego_timestamp_ms": 1000 + index * 100,
                "exo_timestamp_ms": 1000 + index * 100,
                "delta_ms": 0,
                "source_frame_index": 10 + index,
                "filename": exo_name,
                "synchronized": 1,
            }
        )
        pose = np.eye(4)
        pose[2, 3] = 2
        pose_values = {
            field: value for field, value in zip(matrix_fields, pose.reshape(-1))
        }
        exo_pose_rows.append(
            {
                "output_index": index,
                "source_frame_index": 10 + index,
                "ego_timestamp_ms": 1000 + index * 100,
                "exo_timestamp_ms": 1000 + index * 100,
                "delta_ms": 0,
                "valid": 1,
                **pose_values,
            }
        )
        reference_rows.append(
            {
                "output_index": index,
                "source_frame_index": index,
                "estimated_capture_ms": 1000 + index * 100,
                "clip_time_s": index / 10,
                "tx": 0,
                "ty": 0,
                "tz": 2,
                "rx_xyz_rad": 0,
                "ry_xyz_rad": 0,
                "rz_xyz_rad": 0,
            }
        )
    _write_csv(clip_dir / "frames.csv", ego_rows[0].keys(), ego_rows)
    _write_csv(clip_dir / "exo" / "frames.csv", exo_rows[0].keys(), exo_rows)
    _write_csv(
        clip_dir / "exo" / "camera_to_reference.csv",
        exo_pose_rows[0].keys(),
        exo_pose_rows,
    )
    _write_csv(
        clip_dir / "reference" / "camera_to_robot_base.csv",
        reference_rows[0].keys(),
        reference_rows,
    )
    write_json(
        clip_dir / "exo" / "manifest.json",
        {
            "status": "ready",
            "reference_type": "robot_kinematic_camera_to_base",
            "camera": {
                "matrix": [[100, 0, 50], [0, 100, 25], [0, 0, 1]],
                "distortion_coefficients": [],
                "width": 100,
                "height": 50,
            },
        },
    )
    clip = load_robot_clip(
        tmp_path,
        "droid_wrist",
        sequence,
        sample_fps=5,
        source_fps=10,
        verify_exo=False,
    )
    np.testing.assert_array_equal(clip.frame_ids, [0, 2])
    assert [frame.exo_source_frame_index for frame in clip.frames] == [10, 12]
    assert all(not hasattr(frame, "interpolated") for frame in clip.frames)


def test_robot_gpu_commands_have_actions_and_valid_shell(tmp_path: Path):
    plan = load_plan(ROOT / "configs" / "ego_pose_eval_robot_interaction_rgb.yaml")
    clips = robot_demo_selection(plan)
    content = generate_robot_gpu_commands(
        ROOT,
        ROOT / "configs" / "robot_interaction_da3_demo.yaml",
        tmp_path / "data",
        tmp_path / "output",
        tmp_path / "checkpoint",
        clips,
    )
    script = tmp_path / "gpu_commands.sh"
    script.write_text(content, encoding="utf-8")
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "formal-all) run_formal_all" in content
    assert "prepare-exo) run_prepare_exo" in content
    assert "run_prepare_exo()" in content
    assert "droid) run_droid" in content
    assert "rh20t) run_rh20t" in content
    assert "compose) run_compose" in content
    assert "--input-resolution \\" + "\n    336" in content
    assert "CUDA_VISIBLE_DEVICES=7" in content


def test_robot_config_cli_overrides_environment_and_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "yaml_data"),
                "output_root": str(tmp_path / "output"),
                "dataset_plan": str(
                    ROOT / "configs" / "ego_pose_eval_robot_interaction_rgb.yaml"
                ),
                "runtime": {
                    "python_path": "python",
                    "ffmpeg_path": FFMPEG,
                    "ffprobe_path": FFPROBE,
                },
                "da3": {
                    "sample_fps": 10,
                    "checkpoint_path": str(tmp_path / "checkpoint"),
                    "source_root": "source",
                    "input_resolution": 504,
                    "window_size": 60,
                    "window_overlap": 30,
                    "confidence_threshold": 1.5,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EGO_SAMPLE_FPS", "7")
    args = parser().parse_args(
        ["--config", str(config_path), "--sample-fps", "5"]
    )
    config, _, _, data_root, _ = _resolved_config(args)
    assert config["da3"]["sample_fps"] == 5
    assert data_root == (tmp_path / "yaml_data").resolve()


def test_robot_mock_is_h264_triptych_with_isolated_markers(tmp_path: Path):
    report = run_robot_mock_pipeline(
        tmp_path, FFMPEG, FFPROBE, frame_count=4, fps=2
    )
    stream = report["ffprobe"]["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (1920, 1080)
    assert int(stream.get("nb_read_frames") or stream.get("nb_frames")) == 4
    validation = report["panel_validation"]
    assert report["model_display_label"] == "Active Ego Foundation Model"
    assert validation["right_exo_backgrounds_same_source"]
    assert validation["top_reference_primary_pixel_count"] > 0
    assert validation["top_da3_marker_absent"]
    assert validation["bottom_da3_primary_pixel_count"] > 0
    assert validation["bottom_reference_marker_absent"]


def test_robot_native_c2w_postprocess_renders_prefix_and_oracle_without_fill(
    tmp_path: Path,
):
    clip = _robot_clip(tmp_path, count=8)
    for index, frame in enumerate(clip.frames):
        ego = np.full((80, 120, 3), (20 + index, 80, 120), dtype=np.uint8)
        exo = np.full((100, 100, 3), (70, 50 + index, 30), dtype=np.uint8)
        cv2.imwrite(str(frame.ego_image), ego)
        cv2.imwrite(str(frame.exo_image), exo)
    angle = 0.35
    transform = Sim3(
        1.6,
        np.asarray(
            [
                [np.cos(angle), 0, -np.sin(angle)],
                [0, 1, 0],
                [np.sin(angle), 0, np.cos(angle)],
            ],
            dtype=float,
        ),
        np.asarray([0.7, -0.2, 0.4]),
    )
    source = transform.inverse().apply_c2w_poses(clip.reference_from_ego)
    output = tmp_path / "output"
    da3 = output / "da3"
    da3.mkdir(parents=True)
    np.savez(
        da3 / "da3_poses_raw.npz",
        c2w=source,
        confidence=np.full(len(source), 2.5),
        frame_ids=clip.frame_ids,
        timestamps=clip.timestamps_ms,
    )
    config = {
        "runtime": {"ffmpeg_path": FFMPEG, "ffprobe_path": FFPROBE},
        "da3": {"confidence_threshold": 1.5},
        "clip": {"calibration_prefix_sec": 3, "maximum_prefix_ratio": 0.3},
        "alignment": {
            "minimum_translation_span_m": 0.01,
            "minimum_rank_ratio": 0.001,
        },
    }
    result = postprocess_robot_da3(
        clip=clip,
        output_dir=output,
        config=config,
        render_comparison=True,
        evaluate=True,
    )
    prefix = result["metrics"]["sim3_prefix"]
    assert prefix["alignment_status"] == "ok"
    assert prefix["ate_rmse_m"] < 1e-10
    assert prefix["reference_substitution_count"] == 0
    assert prefix["interpolated_ratio"] == 0
    for name in ("comparison_prefix.mp4", "comparison_oracle.mp4"):
        assert (output / name).is_file()
    mapping = read_json(output / "frame_mapping.json")
    assert all(row["da3_pose_valid"] for row in mapping)
    assert all(not row["da3_pose_interpolated"] for row in mapping)
