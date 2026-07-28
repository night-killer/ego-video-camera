from __future__ import annotations

import os

import numpy as np

from .common import WorkerContext, local_dinov2_hub, poses_from_t_q


class _NullViewer:
    def __init__(self, *args, **kwargs):
        self.server = None


def _install_private_salad(context: WorkerContext) -> None:
    torch_home = context.output_dir / "work" / "torch_home"
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    link = checkpoint_dir / "dino_salad.ckpt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(context.checkpoint(1))
    os.environ["TORCH_HOME"] = str(torch_home)


def run(context: WorkerContext):
    import torch
    from vggt.models.vggt import VGGT
    import vggt_slam.solver as solver_module

    _install_private_salad(context)
    solver_module.Viewer = _NullViewer
    parameters = context.parameters
    hub_index = int(parameters.get("dinov2_torchhub_checkpoint_index", 2))
    with local_dinov2_hub(context.checkpoint(hub_index)):
        solver = solver_module.Solver(
            init_conf_threshold=float(parameters.get("conf_threshold", 25.0)),
            lc_thres=float(parameters.get("lc_thres", 0.95)),
            vis_voxel_size=None,
            vis_imgs=False,
        )
    model = VGGT()
    state = torch.load(context.checkpoint(0), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.eval().to(torch.bfloat16).to("cuda")
    context.mark_model_ready()

    image_names = [str(path) for path in context.image_paths]
    submap_size = int(parameters.get("submap_size", 16))
    overlap = int(parameters.get("overlap", 1))
    max_loops = int(parameters.get("max_loops", 1))
    subset: list[str] = []
    first_prediction = False
    for image_index, image_name in enumerate(image_names):
        subset.append(image_name)
        if len(subset) == submap_size + overlap or image_index == len(image_names) - 1:
            prediction = solver.run_predictions(subset, model, max_loops, None, None)
            solver.add_points(prediction)
            solver.graph.optimize()
            if not first_prediction:
                context.mark_first_prediction()
                first_prediction = True
            subset = subset[-overlap:] if overlap else []

    pose_path = context.output_dir / "work" / "vggt_slam_poses.txt"
    solver.map.write_poses_to_file(str(pose_path), solver.graph, kitti_format=False)
    rows = np.loadtxt(pose_path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    by_id = {int(round(row[0])): row[1:8] for row in rows}
    vectors = np.full((len(context.frames), 7), np.nan, dtype=np.float64)
    for index, frame in enumerate(context.frames):
        if int(frame["frame_id"]) in by_id:
            vectors[index] = by_id[int(frame["frame_id"])]
    valid = np.isfinite(vectors).all(axis=1)
    poses = np.full((len(vectors), 4, 4), np.nan, dtype=np.float64)
    poses[valid] = poses_from_t_q(vectors[valid], quaternion_order="xyzw")
    return context.expected_trajectory(
        poses,
        valid=valid,
        metadata={
            "all_input_frames_used": True,
            "viewer_enabled": False,
            "submap_size": submap_size,
            "overlap": overlap,
        },
    )
