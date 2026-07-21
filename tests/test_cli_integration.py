from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from conftest import make_trajectory
from ego_video_camera.schema import CameraFrame, CameraTrajectory, load_trajectory, save_trajectory
from ego_video_camera.video import H264Writer


def test_trajectory_only_and_da3_dry_run_clis(tmp_path: Path, scene_spec) -> None:
    project_root = Path(__file__).resolve().parents[1]
    original = make_trajectory(scene_spec, trajectory_type="keyframes", count=2)
    second = original.frames[1]
    keyframes = CameraTrajectory(
        trajectory_type="keyframes",
        coordinate_system=original.coordinate_system,
        scene=original.scene,
        video=original.video,
        frames=[
            original.frames[0],
            CameraFrame(
                frame_index=2,
                timestamp_seconds=2 / original.video.fps,
                camera_to_world=second.camera_to_world,
                K=second.K,
            ),
        ],
    )
    keyframes_path = tmp_path / "keyframes.json"
    gt_dir = tmp_path / "gt"
    save_trajectory(keyframes_path, keyframes)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "02_interpolate_and_render.py"),
            "--keyframes",
            str(keyframes_path),
            "--output-dir",
            str(gt_dir),
            "--trajectory-only",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    dense = load_trajectory(gt_dir / "gt_trajectory.json")
    assert len(dense.frames) == 3
    manifest = json.loads((gt_dir / "render_manifest.json").read_text())
    assert manifest["status"] == "trajectory_only"

    video_path = gt_dir / "gt_video.mp4"
    black = np.zeros((dense.video.height, dense.video.width, 3), dtype=np.uint8)
    with H264Writer(
        video_path,
        width=dense.video.width,
        height=dense.video.height,
        fps=dense.video.fps,
    ) as writer:
        for _ in dense.frames:
            writer.write(black)

    da3_root = tmp_path / "da3_source"
    checkpoint = tmp_path / "checkpoint"
    (da3_root / "src" / "depth_anything_3").mkdir(parents=True)
    (da3_root / "src" / "depth_anything_3" / "api.py").write_text("# preflight fixture\n")
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    dry_run_dir = tmp_path / "da3_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "03_predict_da3_camera.py"),
            "--video",
            str(video_path),
            "--gt-trajectory",
            str(gt_dir / "gt_trajectory.json"),
            "--output-dir",
            str(dry_run_dir),
            "--da3-root",
            str(da3_root),
            "--model-dir",
            str(checkpoint),
            "--dry-run",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    preflight = json.loads((dry_run_dir / "preflight.json").read_text())
    assert preflight["status"] == "ready"
    assert preflight["video"]["decoded_frames"] == 3
