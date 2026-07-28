from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .common import WorkerContext, import_file


def run(context: WorkerContext):
    import torch

    demo = import_file("ego_benchmark_lingbot_demo", context.repo / "demo.py")
    args = SimpleNamespace(
        mode="streaming",
        model_path=str(context.checkpoint(0)),
        image_size=int(context.parameters.get("image_size", 518)),
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=max(500, len(context.frames) + 8),
        kv_cache_sliding_window=320,
        num_scale_frames=int(context.parameters.get("num_scale_frames", 8)),
        use_sdpa=True,
        camera_num_iterations=4,
    )
    images, _, _ = demo.load_images(
        image_folder=str(context.stage_frames()),
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    device = torch.device("cuda")
    model = demo.load_model(args, device)
    context.mark_model_ready()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=int(context.parameters.get("keyframe_interval", 1)),
            output_device=torch.device("cpu"),
        )
    predictions, _ = demo.postprocess(predictions, predictions.get("images", images))
    c2w = np.asarray(predictions["extrinsic"], dtype=np.float64)
    if c2w.shape[-2:] == (3, 4):
        homogeneous = np.repeat(np.eye(4)[None], len(c2w), axis=0)
        homogeneous[:, :3] = c2w
        c2w = homogeneous
    context.mark_first_prediction()
    return context.expected_trajectory(
        c2w,
        metadata={
            "mode": "streaming",
            "num_scale_frames": args.num_scale_frames,
            "keyframe_interval": int(context.parameters.get("keyframe_interval", 1)),
        },
    )
