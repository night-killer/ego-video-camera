from __future__ import annotations

import sys
from types import MethodType, SimpleNamespace
from typing import Iterator

import numpy as np

from .common import WorkerContext, poses_from_t_q


class _FlowArguments(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def _droid_stream(context: WorkerContext) -> Iterator[tuple[int, object, object]]:
    import cv2
    import torch

    for index, row in enumerate(context.frames):
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read {row['image_path']}")
        height, width = image.shape[:2]
        scale = np.sqrt((384.0 * 512.0) / (height * width))
        target_height, target_width = int(height * scale), int(width * scale)
        image = cv2.resize(image, (target_width, target_height))
        target_height -= target_height % 8
        target_width -= target_width % 8
        tensor = torch.as_tensor(image[:target_height, :target_width]).permute(2, 0, 1)[None]
        intrinsic = np.asarray(row["intrinsic"], dtype=np.float32)
        vector = torch.as_tensor(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
            dtype=torch.float32,
        )
        vector[0::2] *= target_width / width
        vector[1::2] *= target_height / height
        yield index, tensor, vector


def _run_droid(context: WorkerContext) -> np.ndarray:
    import torch

    droid_path = context.repo.parent / "DROID-SLAM" / "droid_slam"
    if str(droid_path) not in sys.path:
        sys.path.insert(0, str(droid_path))
    from droid import Droid

    first = next(_droid_stream(context))
    args = SimpleNamespace(
        weights=str(context.checkpoint(2)),
        buffer=max(512, len(context.frames) + 32),
        image_size=[first[1].shape[2], first[1].shape[3]],
        disable_vis=True,
        stereo=False,
        beta=0.3,
        filter_thresh=2.4,
        warmup=8,
        keyframe_thresh=4.0,
        frontend_thresh=16.0,
        frontend_window=25,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=22.0,
        backend_radius=2,
        backend_nms=3,
        upsample=False,
    )
    tracker = Droid(args)
    for timestamp, image, intrinsic in _droid_stream(context):
        tracker.track(timestamp, image, intrinsics=intrinsic)
    vectors = np.asarray(tracker.terminate(_droid_stream(context)), dtype=np.float64)
    del tracker
    torch.cuda.empty_cache()
    return poses_from_t_q(vectors, quaternion_order="xyzw")


def _raft_flows(context: WorkerContext) -> np.ndarray:
    import cv2
    import torch

    core = context.repo.parent / "MegaSaM" / "cvd_opt" / "core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from raft import RAFT

    arguments = _FlowArguments(small=False, dropout=0.0, alternate_corr=False)
    wrapped = torch.nn.DataParallel(RAFT(arguments))
    wrapped.load_state_dict(torch.load(context.checkpoint(3), map_location="cpu"))
    model = wrapped.module.cuda().eval()
    images = []
    for path in context.image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA))
    flows = []
    iterations = int(context.parameters.get("raft_iterations", 20))
    with torch.inference_mode():
        for first, second in zip(images[:-1], images[1:]):
            image1 = torch.from_numpy(first).permute(2, 0, 1)[None].float().cuda()
            image2 = torch.from_numpy(second).permute(2, 0, 1)[None].float().cuda()
            _, flow, _ = model(image1, image2, iters=iterations, test_mode=True)
            flows.append(flow[0].permute(1, 2, 0).float().cpu().numpy())
    del model, wrapped
    torch.cuda.empty_cache()
    return np.asarray(flows, dtype=np.float32)


def _options() -> SimpleNamespace:
    return SimpleNamespace(
        window=90,
        n_dec_layers=2,
        n_head=4,
        d_k=256,
        d_v=256,
        d_model=256,
        dist_scale=10.0,
        input_of_feats=False,
        freeze_of_cnn=False,
        normal_window=90,
        normal_n_dec_layers=4,
        normal_n_head=4,
        normal_d_k=256,
        normal_d_v=256,
        normal_d_model=256,
    )


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(np.linalg.norm(source), 1e-12)
    target /= max(np.linalg.norm(target), 1e-12)
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.asarray([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.asarray([0.0, 1.0, 0.0])
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def run(context: WorkerContext):
    import torch
    import pytorch3d.transforms as transforms

    if len(context.frames) == 1:
        context.mark_model_ready()
        context.mark_first_prediction()
        return context.expected_trajectory(
            np.eye(4, dtype=np.float64)[None],
            metadata={"first_pose_identity": True, "gt_horizontal_alignment": False},
        )

    slam = _run_droid(context)
    origin_inverse = np.linalg.inv(slam[0])
    slam = np.einsum("ij,njk->nik", origin_inverse, slam)
    flows = _raft_flows(context)

    from egoego.model.head_estimation_transformer import HeadFormer
    from egoego.model.head_normal_estimation_transformer import HeadNormalFormer
    import egoego.model.resnet as resnet_module

    device = torch.device("cuda")
    options = _options()
    original_resnet18 = resnet_module.models.resnet18

    def local_resnet18(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs.pop("weights", None)
        try:
            return original_resnet18(*args, weights=None, **kwargs)
        except TypeError:
            return original_resnet18(*args, pretrained=False, **kwargs)

    resnet_module.models.resnet18 = local_resnet18
    try:
        head_model = HeadFormer(options, device)
    finally:
        resnet_module.models.resnet18 = original_resnet18
    head_state = torch.load(context.checkpoint(1), map_location=device)
    head_model.load_state_dict(head_state["transformer_encoder_state_dict"])
    head_model = head_model.to(device).eval()

    gravity_model = HeadNormalFormer(options, device, eval_whole_pipeline=True)
    gravity_state = torch.load(context.checkpoint(0), map_location=device)
    gravity_model.load_state_dict(gravity_state["transformer_encoder_state_dict"])
    gravity_model = gravity_model.to(device).eval()
    context.mark_model_ready()

    frame_delta = np.diff(
        np.asarray([row["timestamp_ns"] for row in context.frames], dtype=np.float64)
    )
    integration_dt = float(np.median(frame_delta) * 1e-9)
    original_va2rot = head_model.va2rot

    def va2rot_with_manifest_time(self, current, velocities, dt=integration_dt):
        return original_va2rot(current, velocities, dt=integration_dt)

    head_model.va2rot = MethodType(va2rot_with_manifest_time, head_model)

    slam_translation = torch.from_numpy(slam[:, :3, 3]).float()[None].to(device)
    slam_rotation = torch.from_numpy(slam[:, :3, :3]).float()[None].to(device)
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    head_pose_seed = torch.zeros((1, len(slam), 7), device=device)
    head_pose_seed[:, :, 3:] = identity_quaternion
    data = {
        "of": torch.from_numpy(flows)[None].to(device),
        "seq_len": torch.tensor([len(flows)], device=device),
        "head_pose": head_pose_seed,
        "aligned_slam_trans": slam_translation,
        "aligned_slam_rot_quat": transforms.matrix_to_quaternion(slam_rotation),
        "aligned_slam_rot_mat": slam_rotation,
    }
    with torch.inference_mode():
        head_prediction = head_model.forward_for_eval(data)
        gravity_prediction = gravity_model.forward(
            {
                "head_trans": slam_translation,
                "head_rot_mat": slam_rotation,
                "seq_len": torch.tensor([len(slam)], device=device),
            }
        )

    predicted_normal = gravity_prediction["pred_normal"][0].float().cpu().numpy()
    gravity_rotation = _rotation_between(predicted_normal, np.asarray([0.0, 0.0, 1.0]))
    predicted_scale = float(head_prediction["pred_scale"].float().cpu())
    if not np.isfinite(predicted_scale) or predicted_scale <= 0:
        raise ValueError(f"EgoEgo predicted invalid metric scale {predicted_scale}")
    increments = np.diff(slam[:, :3, 3], axis=0)
    increments = predicted_scale * (increments @ gravity_rotation.T)
    translations = np.vstack((np.zeros(3), np.cumsum(increments, axis=0)))

    quaternions = head_prediction["head_pose"][0, :, 3:]
    rotations = transforms.quaternion_to_matrix(quaternions).float().cpu().numpy()
    count = min(len(context.frames), len(rotations), len(translations))
    if count != len(context.frames):
        raise ValueError(f"EgoEgo returned {count}/{len(context.frames)} poses")
    rotations = np.einsum("ij,njk->nik", rotations[0].T, rotations)
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    poses[0] = np.eye(4)
    context.mark_first_prediction()
    return context.expected_trajectory(
        poses,
        metadata={
            "droid_camera": True,
            "raft_optical_flow": True,
            "headformer": str(context.parameters.get("head_model", "ares")),
            "gravitynet": True,
            "predicted_metric_scale": predicted_scale,
            "integration_dt_sec": integration_dt,
            "first_pose_identity": True,
            "gt_horizontal_alignment": False,
        },
    )
