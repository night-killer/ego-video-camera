from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from .common import WorkerContext, local_dinov2_hub, poses_from_t_q


def _add_import_paths(context: WorkerContext) -> None:
    for path in (
        context.repo / "Depth-Anything",
        context.repo / "UniDepth",
        context.repo / "base" / "droid_slam",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _relative_disparities(context: WorkerContext) -> list[np.ndarray]:
    import cv2
    import torch
    import torch.nn.functional as functional
    from depth_anything.dpt import DPT_DINOv2
    from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    from torchvision.transforms import Compose

    hub_index = int(context.parameters.get("dinov2_torchhub_checkpoint_index", 4))
    hub_repository = context.checkpoint(hub_index)
    with local_dinov2_hub(hub_repository):
        model = DPT_DINOv2(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
            localhub=True,
        ).cuda()
    model.load_state_dict(torch.load(context.checkpoint(1), map_location="cpu"), strict=True)
    model.eval()
    transform = Compose(
        [
            Resize(
                width=768,
                height=768,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            PrepareForNet(),
        ]
    )
    output = []
    with torch.inference_mode():
        for path in context.image_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            height, width = rgb.shape[:2]
            tensor = torch.from_numpy(transform({"image": rgb})["image"])[None].cuda()
            disparity = model(tensor)
            disparity = functional.interpolate(
                disparity[None], (height, width), mode="bilinear", align_corners=False
            )[0, 0]
            output.append(disparity.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return output


def _metric_depth_and_fov(
    context: WorkerContext,
) -> tuple[list[np.ndarray], np.ndarray]:
    import cv2
    import torch
    from unidepth.models import UniDepthV2

    model_dir = context.checkpoint(2).parent
    model = UniDepthV2.from_pretrained(
        str(model_dir), local_files_only=True
    ).cuda().eval()
    depths: list[np.ndarray] = []
    fovs = []
    with torch.inference_mode():
        for path in context.image_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            if width > height:
                final_width, final_height = 640, int(round(640 * height / width))
            else:
                final_width, final_height = int(round(640 * width / height)), 640
            resized = cv2.resize(rgb, (final_width, final_height), cv2.INTER_AREA)
            prediction = model.infer(torch.from_numpy(resized).permute(2, 0, 1))
            depth = prediction["depth"][0, 0].float().cpu().numpy()
            focal = float(prediction["intrinsics"][0, 0, 0].float().cpu())
            fovs.append(np.degrees(2.0 * np.arctan(depth.shape[1] / (2.0 * focal))))
            depths.append(depth)
    del model
    torch.cuda.empty_cache()
    return depths, np.asarray(fovs, dtype=np.float64)


def _alignment(
    disparities: list[np.ndarray], metric_depths: list[np.ndarray]
) -> tuple[float, float, float]:
    import cv2

    if len(disparities) != len(metric_depths) or not disparities:
        raise ValueError("MegaSaM depth priors do not match the input frame count")
    scales, shifts, resized_disparities = [], [], []
    for disparity, metric_depth in zip(disparities, metric_depths):
        relative = cv2.resize(
            disparity,
            (metric_depth.shape[1], metric_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        ).astype(np.float64)
        metric_disparity = 1.0 / np.maximum(metric_depth.astype(np.float64), 1e-8)
        invalid = (metric_depth < 2.0) & (relative < 0.02)
        metric_disparity[invalid] = 1e-2
        relative_centered = relative - np.median(relative)
        metric_centered = metric_disparity - np.median(metric_disparity)
        usable = np.abs(relative_centered) > 1e-8
        ratios = metric_centered[usable] / relative_centered[usable]
        ratios = ratios[np.isfinite(ratios)]
        if not len(ratios):
            raise ValueError("MegaSaM disparity alignment is degenerate")
        scale = float(np.median(ratios))
        shift = float(np.median(metric_disparity - scale * relative))
        scales.append(scale)
        shifts.append(shift)
        resized_disparities.append(relative)

    products = np.asarray(scales) * np.asarray(shifts)
    representative = int(np.argmin(np.abs(products - np.median(products))))
    scale, shift = scales[representative], shifts[representative]
    aligned = np.concatenate(
        [(scale * value + shift).reshape(-1) for value in resized_disparities]
    )
    aligned = aligned[np.isfinite(aligned)]
    if not len(aligned):
        raise ValueError("MegaSaM aligned disparities are non-finite")
    normalize_scale = float(np.percentile(aligned, 98) / 2.0)
    if not np.isfinite(normalize_scale) or normalize_scale <= 0:
        raise ValueError(f"MegaSaM produced invalid depth normalization {normalize_scale}")
    return scale, shift, normalize_scale


def _stream(
    context: WorkerContext,
    disparities: list[np.ndarray],
    alignment: tuple[float, float, float],
    intrinsic: np.ndarray,
) -> Iterator[tuple[int, object, object, object, object]]:
    import cv2
    import torch
    import torch.nn.functional as functional

    scale, shift, normalize_scale = alignment
    for index, (path, disparity) in enumerate(zip(context.image_paths, disparities)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read {path}")
        height, width = image.shape[:2]
        target_height = int(height * np.sqrt((384.0 * 512.0) / (height * width)))
        target_width = int(width * np.sqrt((384.0 * 512.0) / (height * width)))
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        target_height -= target_height % 8
        target_width -= target_width % 8
        image_tensor = torch.as_tensor(image[:target_height, :target_width]).permute(2, 0, 1)[None]

        depth = np.clip(
            1.0 / np.maximum((scale * disparity + shift) / normalize_scale, 1e-8),
            1e-4,
            1e4,
        )
        depth[depth < 1e-2] = 0.0
        depth_tensor = functional.interpolate(
            torch.as_tensor(depth)[None, None],
            (target_height, target_width),
            mode="nearest-exact",
        )[0, 0]
        mask = torch.ones_like(depth_tensor)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        vector = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        vector[0::2] *= target_width / width
        vector[1::2] *= target_height / height
        yield index, image_tensor, depth_tensor, vector, mask


def run(context: WorkerContext):
    import cv2
    import torch

    _add_import_paths(context)
    disparities = _relative_disparities(context)
    metric_depths, fovs = _metric_depth_and_fov(context)
    alignment = _alignment(disparities, metric_depths)
    first = cv2.imread(str(context.image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(context.image_paths[0])
    height, width = first.shape[:2]
    focal = width / (2.0 * np.tan(np.radians(float(np.median(fovs))) / 2.0))
    intrinsic = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    del metric_depths
    torch.cuda.empty_cache()

    from droid import Droid

    parameters = context.parameters
    first_item = next(_stream(context, disparities, alignment, intrinsic))
    args = SimpleNamespace(
        weights=str(context.checkpoint(0)),
        buffer=max(int(parameters.get("buffer", 1024)), 2 * len(context.frames) + 32),
        image_size=[first_item[1].shape[2], first_item[1].shape[3]],
        disable_vis=True,
        stereo=False,
        beta=float(parameters.get("beta", 0.3)),
        filter_thresh=float(parameters.get("filter_thresh", 2.0)),
        warmup=int(parameters.get("warmup", 8)),
        keyframe_thresh=float(parameters.get("keyframe_thresh", 2.0)),
        frontend_thresh=float(parameters.get("frontend_thresh", 12.0)),
        frontend_window=int(parameters.get("frontend_window", 25)),
        frontend_radius=int(parameters.get("frontend_radius", 2)),
        frontend_nms=int(parameters.get("frontend_nms", 1)),
        backend_thresh=float(parameters.get("backend_thresh", 16.0)),
        backend_radius=int(parameters.get("backend_radius", 2)),
        backend_nms=int(parameters.get("backend_nms", 3)),
        upsample=False,
    )
    tracker = Droid(args)
    context.mark_model_ready()
    last = None
    for last in _stream(context, disparities, alignment, intrinsic):
        timestamp, image, depth, intrinsics, mask = last
        tracker.track(timestamp, image, depth, intrinsics=intrinsics, mask=mask)
    if last is None:
        raise ValueError("MegaSaM received no frames")
    timestamp, image, depth, intrinsics, mask = last
    tracker.track_final(timestamp, image, depth, intrinsics=intrinsics, mask=mask)
    termination = tracker.terminate(
        _stream(context, disparities, alignment, intrinsic),
        _opt_intr=bool(parameters.get("optimize_intrinsics", True)),
        full_ba=bool(parameters.get("full_ba", True)),
        scene_name=context.manifest["sequence_id"],
    )
    vectors = np.asarray(termination[0], dtype=np.float64)
    if len(vectors) != len(context.frames):
        raise ValueError(f"MegaSaM returned {len(vectors)}/{len(context.frames)} dense poses")
    c2w = np.linalg.inv(poses_from_t_q(vectors, quaternion_order="xyzw"))
    context.mark_first_prediction()
    return context.expected_trajectory(
        c2w,
        metadata={
            "full_ba": bool(parameters.get("full_ba", True)),
            "dense_termination_output": True,
            "fov_deg": float(np.median(fovs)),
            "depth_alignment": {
                "scale": alignment[0],
                "shift": alignment[1],
                "normalization": alignment[2],
            },
        },
    )
