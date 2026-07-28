from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from .common import WorkerContext, poses_from_t_q


def _replace_link(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _stage_private_tree(context: WorkerContext) -> tuple[Path, Path, Path]:
    root = context.output_dir / "work" / "hawor_private"
    sequence_root = root / "benchmark"
    frames_dir = sequence_root / "extracted_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    for index, source in enumerate(context.image_paths):
        name = f"{index:04d}.jpg"
        expected.add(name)
        _replace_link(frames_dir / name, source)
    for path in frames_dir.iterdir():
        if path.name not in expected and (path.is_file() or path.is_symlink()):
            path.unlink()

    _replace_link(root / "weights" / "external" / "detector.pt", context.checkpoint(0))
    _replace_link(root / "weights" / "external" / "droid.pth", context.checkpoint(3))
    hawor_checkpoint = root / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
    _replace_link(hawor_checkpoint, context.checkpoint(1))
    _replace_link(
        root / "weights" / "hawor" / "checkpoints" / "infiller.pt",
        context.checkpoint(2),
    )
    model_config = context.checkpoint(1).parent.parent / "model_config.yaml"
    if not model_config.is_file():
        raise FileNotFoundError(f"Missing HaWoR model config: {model_config}")
    _replace_link(root / "weights" / "hawor" / "model_config.yaml", model_config)

    _replace_link(root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl", context.checkpoint(6))
    _replace_link(
        root / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl",
        context.checkpoint(5),
    )
    mean_parameters = context.repo / "_DATA" / "data" / "mano_mean_params.npz"
    if mean_parameters.is_file():
        _replace_link(root / "_DATA" / "data" / "mano_mean_params.npz", mean_parameters)

    _replace_link(root / "thirdparty" / "DROID-SLAM", context.repo / "thirdparty" / "DROID-SLAM")
    metric_root = root / "thirdparty" / "Metric3D"
    metric_root.mkdir(parents=True, exist_ok=True)
    _replace_link(metric_root / "metric.py", context.repo / "thirdparty" / "Metric3D" / "metric.py")
    _replace_link(metric_root / "mono", context.repo / "thirdparty" / "Metric3D" / "mono")
    _replace_link(
        metric_root / "data_info", context.repo / "thirdparty" / "Metric3D" / "data_info"
    )
    _replace_link(
        metric_root / "weights" / "metric_depth_vit_large_800k.pth",
        context.checkpoint(4),
    )
    video_path = root / "benchmark.mp4"
    video_path.touch(exist_ok=True)
    return root, video_path, hawor_checkpoint


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run(context: WorkerContext):
    private_root, video_path, checkpoint = _stage_private_tree(context)
    for import_path in (context.repo, context.repo / "thirdparty" / "Metric3D"):
        value = str(import_path)
        if value not in sys.path:
            sys.path.insert(0, value)

    with _working_directory(private_root):
        from scripts.scripts_test_video.detect_track_video import detect_track_video
        from scripts.scripts_test_video.hawor_slam import hawor_slam
        from scripts.scripts_test_video.hawor_video import hawor_motion_estimation

        args = SimpleNamespace(
            video_path=str(video_path),
            input_type="file",
            img_focal=None,
            checkpoint=str(checkpoint),
            infiller_weight=str(
                private_root / "weights" / "hawor" / "checkpoints" / "infiller.pt"
            ),
            vis_mode="world",
        )
        context.mark_model_ready()
        start, end, sequence_folder, _ = detect_track_video(args)
        hawor_motion_estimation(args, start, end, sequence_folder)
        hawor_slam(args, start, end)

    slam_path = (
        private_root
        / "benchmark"
        / "SLAM"
        / f"hawor_slam_w_scale_{start}_{end}.npz"
    )
    with np.load(slam_path, allow_pickle=False) as result:
        vectors = np.asarray(result["traj"], dtype=np.float64)
        scale = float(np.asarray(result["scale"]).reshape(-1)[0])
    if vectors.shape != (len(context.frames), 7):
        raise ValueError(
            f"HaWoR returned trajectory {vectors.shape}, expected {(len(context.frames), 7)}"
        )
    vectors[:, :3] *= scale
    context.mark_first_prediction()
    return context.expected_trajectory(
        poses_from_t_q(vectors, quaternion_order="xyzw"),
        metadata={
            "target_fps_frames_staged_directly": True,
            "hand_masks_used": bool(context.parameters.get("use_hand_masks", True)),
            "metric_scale": scale,
            "native_output": str(slam_path),
        },
    )
