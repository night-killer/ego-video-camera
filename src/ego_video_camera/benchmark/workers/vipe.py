from __future__ import annotations

import os

import numpy as np

from .common import WorkerContext


def _install_private_droid(context: WorkerContext) -> None:
    torch_home = context.output_dir / "work" / "torch_home"
    destination = torch_home / "hub" / "droid_slam"
    destination.mkdir(parents=True, exist_ok=True)
    link = destination / "droid.pth"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(context.checkpoint(1))
    os.environ["TORCH_HOME"] = str(torch_home)


def run(context: WorkerContext):
    import cv2
    import torch
    from omegaconf import OmegaConf
    from vipe.priors.depth.dav3 import DepthAnything3Model
    from vipe.slam.system import SLAMSystem
    from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
    from vipe.utils.cameras import CameraType
    from vipe.utils.model_cache import ModelCache

    _install_private_droid(context)

    class ManifestStream(VideoStream):
        def __init__(self):
            first = cv2.imread(context.frames[0]["image_path"], cv2.IMREAD_COLOR)
            if first is None:
                raise FileNotFoundError(context.frames[0]["image_path"])
            self._size = first.shape[:2]

        def frame_size(self):
            return self._size

        def name(self):
            return context.manifest["sequence_id"]

        def fps(self):
            return float(context.manifest["target_fps"])

        def __len__(self):
            return len(context.frames)

        def attributes(self):
            return {FrameAttribute.INTRINSICS, FrameAttribute.CAMERA_TYPE}

        def __iter__(self):
            for index, row in enumerate(context.frames):
                image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(row["image_path"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                intrinsic = np.asarray(row["intrinsic"], dtype=np.float32)
                yield VideoFrame(
                    raw_frame_idx=index,
                    rgb=torch.from_numpy(image.copy()).float().div(255.0),
                    intrinsics=torch.as_tensor(
                        [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
                    ),
                    camera_type=CameraType.PINHOLE,
                )

    config = OmegaConf.load(context.repo / "configs" / "slam" / "default.yaml")
    config.optimize_intrinsics = False
    config.keyframe_depth = "dav3"
    config.visualize = False
    cache = ModelCache()
    cache._models["depth/dav3"] = DepthAnything3Model(weights_path=str(context.checkpoint(0)))
    system = SLAMSystem(torch.device("cuda"), config, model_cache=cache)
    context.mark_model_ready()
    output = system.run([ManifestStream()], camera_type=CameraType.PINHOLE)
    c2w = output.trajectory.matrix().detach().float().cpu().numpy()
    context.mark_first_prediction()
    return context.expected_trajectory(
        c2w,
        metadata={
            "keyframe_depth": "dav3",
            "optimize_intrinsics": False,
            "ba_residual": float(output.ba_residual),
        },
    )
