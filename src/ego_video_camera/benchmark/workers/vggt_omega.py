from __future__ import annotations

import numpy as np

from ..windowing import local_window_trajectory, stitch_pose_windows, window_slices
from .common import WorkerContext, w2c_to_c2w


PATCH_SIZE = 16
CHECKPOINT_RESOLUTION = 512


def validate_image_resolution(parameters: dict) -> int:
    resolution = int(parameters.get("image_resolution", CHECKPOINT_RESOLUTION))
    if resolution != CHECKPOINT_RESOLUTION:
        raise ValueError(
            "VGGT-Omega-1B-512 requires image_resolution=512, "
            f"got {resolution}"
        )
    if resolution % PATCH_SIZE != 0:
        raise ValueError(
            f"VGGT-Omega image_resolution must be divisible by {PATCH_SIZE}, "
            f"got {resolution}"
        )
    return resolution


def run(context: WorkerContext):
    resolution = validate_image_resolution(context.parameters)
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().eval()
    state = torch.load(context.checkpoint(0), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to("cuda")
    context.mark_model_ready()

    window_size = int(context.parameters.get("window_size", 200))
    overlap = int(context.parameters.get("overlap", 40))
    windows = []
    for window_index, bounds in enumerate(window_slices(len(context.frames), window_size, overlap)):
        rows = context.frames[bounds]
        image_paths = [row["image_path"] for row in rows]
        images = load_and_preprocess_images(image_paths, image_resolution=resolution).to("cuda")
        with torch.inference_mode():
            prediction = model(images)
            extrinsic, _ = encoding_to_camera(
                prediction["pose_enc"], prediction["images"].shape[-2:]
            )
        values = extrinsic.detach().float().cpu().numpy()
        if values.ndim == 4:
            values = values[0]
        windows.append(local_window_trajectory(rows, w2c_to_c2w(values)))
        if window_index == 0:
            context.mark_first_prediction()
        del images, prediction, extrinsic
        torch.cuda.empty_cache()

    result = stitch_pose_windows(
        windows,
        timestamp_ns=np.asarray([row["timestamp_ns"] for row in context.frames]),
        frame_id=np.asarray([row["frame_id"] for row in context.frames]),
    )
    result.metadata.update(
        {
            "method_id": context.manifest["method_id"],
            "seed": context.seed,
            "window_size": window_size,
            "overlap": overlap,
        }
    )
    return result
