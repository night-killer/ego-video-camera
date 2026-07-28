from __future__ import annotations

import copy
import sys

import numpy as np
import yaml

from .common import WorkerContext, import_file, local_dinov2_hub, w2c_to_c2w


def run_direct(context: WorkerContext):
    import torch
    from depth_anything_3.api import DepthAnything3

    model_dir = context.checkpoint(0).parent
    model = DepthAnything3.from_pretrained(str(model_dir)).to("cuda").eval()
    context.mark_model_ready()
    prediction = model.inference(
        [str(path) for path in context.image_paths],
        ref_view_strategy=str(context.parameters.get("ref_view_strategy", "saddle_balanced")),
    )
    c2w = w2c_to_c2w(np.asarray(prediction.extrinsics))
    context.mark_first_prediction()
    del model
    torch.cuda.empty_cache()
    return context.expected_trajectory(c2w, metadata={"mode": "direct"})


def run_streaming(context: WorkerContext):
    streaming_root = context.repo / "da3_streaming"
    sys.path.insert(0, str(streaming_root))
    module = import_file("ego_benchmark_da3_streaming", streaming_root / "da3_streaming.py")
    base_path = streaming_root / "configs" / "base_config.yaml"
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["Weights"].update(
        {
            "DA3_CONFIG": str(context.checkpoint(0)),
            "DA3": str(context.checkpoint(1)),
            "SALAD": str(context.checkpoint(2)),
        }
    )
    model_cfg = config["Model"]
    model_cfg["chunk_size"] = int(context.parameters.get("chunk_size", 120))
    model_cfg["overlap"] = int(context.parameters.get("overlap", 60))
    if len(context.frames) <= model_cfg["chunk_size"]:
        model_cfg["overlap"] = 0
    model_cfg["loop_enable"] = bool(context.parameters.get("loop_enable", True))
    model_cfg["align_lib"] = str(context.parameters.get("align_lib", "torch"))
    model_cfg["save_depth_conf_result"] = False
    model_cfg["delete_temp_files"] = True
    # The benchmark needs poses only; suppress PLY serialization without changing
    # any prediction or alignment computation.
    module.save_confident_pointcloud_batch = lambda **_: None

    image_dir = context.stage_frames()
    native_output = context.output_dir / "work" / "da3_streaming"
    hub_index = int(context.parameters.get("dinov2_torchhub_checkpoint_index", 3))
    with local_dinov2_hub(context.checkpoint(hub_index)):
        runner = module.DA3_Streaming(str(image_dir), str(native_output), config)
        context.mark_model_ready()
        try:
            runner.run()
            pose_path = native_output / "camera_poses.txt"
            values = np.loadtxt(pose_path, dtype=np.float64).reshape(-1, 4, 4)
            context.mark_first_prediction()
        finally:
            runner.close()
    return context.expected_trajectory(
        values,
        metadata={
            "mode": "streaming",
            "chunk_size": model_cfg["chunk_size"],
            "overlap": model_cfg["overlap"],
            "native_pose_file": str(pose_path),
        },
    )
