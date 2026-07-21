from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ego_video_camera.camera import fov_y_to_intrinsics
from ego_video_camera.gaussian import GaussianScene, render_camera


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rasterization test requires a GPU")
def test_single_gaussian_gsplat_rasterization() -> None:
    try:
        import gsplat  # noqa: F401
    except ImportError:
        pytest.skip("gsplat is not installed")

    device = torch.device("cuda")
    scene = GaussianScene(
        means=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device),
        quaternions=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
        scales=torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float32, device=device),
        opacities=torch.tensor([0.95], dtype=torch.float32, device=device),
        harmonics=torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32, device=device),
        sh_degree=0,
        source_path=Path("synthetic.ply"),
        robust_bounds_min=np.array([-0.2, -0.2, 1.8]),
        robust_bounds_max=np.array([0.2, 0.2, 2.2]),
    )
    image = render_camera(
        scene,
        np.eye(4),
        fov_y_to_intrinsics(64, 48, 65.0),
        width=64,
        height=48,
    )
    assert image.shape == (48, 64, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0
