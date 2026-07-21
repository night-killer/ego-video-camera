from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

from ego_video_camera.da3 import DEFAULT_MODEL_DIR, run_da3_camera_prediction
from ego_video_camera.gaussian import (
    load_gaussian_scene,
    render_trajectory_video,
)
from ego_video_camera.interpolation import interpolate_keyframes
from ego_video_camera.comparison import render_comparisons
from ego_video_camera.schema import load_trajectory, save_trajectory
from ego_video_camera.trajectory_visualization import write_trajectory_visualizations
from ego_video_camera.video import video_info


@pytest.mark.gpu
@pytest.mark.integration
def test_real_keyframes_complete_gpu_pipeline(tmp_path: Path) -> None:
    """Opt-in smoke test for stages 2--5 using a manually authored trajectory."""

    keyframes_value = os.environ.get("EGO_CAMERA_GPU_SMOKE_KEYFRAMES")
    if not keyframes_value:
        pytest.skip("set EGO_CAMERA_GPU_SMOKE_KEYFRAMES to run the full GPU smoke test")
    if not torch.cuda.is_available():
        pytest.skip("the full pipeline smoke test requires CUDA")

    keyframes = load_trajectory(Path(keyframes_value))
    gt = interpolate_keyframes(keyframes)
    assert len(gt.frames) <= 180, "GPU smoke input exceeds the DA3 v1 frame limit"
    gt_path = tmp_path / "gt_trajectory.json"
    gt_video = tmp_path / "gt_video.mp4"
    save_trajectory(gt_path, gt)

    device = os.environ.get("EGO_CAMERA_GPU_SMOKE_DEVICE", "cuda:0")
    scene = load_gaussian_scene(Path(gt.scene.ply_path), device=device)
    assert render_trajectory_video(scene, gt, gt_video) == len(gt.frames)
    del scene
    gc.collect()
    torch.cuda.empty_cache()

    project_root = Path(__file__).resolve().parents[1]
    da3_manifest = run_da3_camera_prediction(
        video_path=gt_video,
        gt_trajectory_path=gt_path,
        output_dir=tmp_path / "da3",
        da3_root=project_root / "third_party" / "depth-anything-3",
        model_dir=Path(
            os.environ.get(
                "EGO_CAMERA_DA3_MODEL",
                os.environ.get("DA3_MODEL", str(DEFAULT_MODEL_DIR)),
            )
        ),
        device=device,
        process_res=504,
        max_frames=180,
    )
    predicted = load_trajectory(Path(da3_manifest["aligned_trajectory"]))

    scene = load_gaussian_scene(Path(gt.scene.ply_path), device=device)
    visualization = write_trajectory_visualizations(
        scene, gt, predicted, tmp_path / "trajectory_visualization"
    )
    comparison = render_comparisons(scene, gt, predicted, tmp_path / "comparison")

    assert len(visualization["views"]) == 2
    for view in visualization["views"]:
        info = video_info(Path(view["video_path"]), decode_count=True)
        assert info["decoded_frames"] == len(gt.frames)
    for path in comparison["outputs"].values():
        info = video_info(Path(path), decode_count=True)
        assert info["decoded_frames"] == len(gt.frames)
