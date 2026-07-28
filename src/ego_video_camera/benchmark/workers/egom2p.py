from __future__ import annotations

import numpy as np

from ..windowing import stitch_pose_windows, window_slices
from .common import WorkerContext
from .motion_tokens import camera_window, rgb_clip


def run(context: WorkerContext):
    import torch
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    from egom2p.data.modality_info import MODALITY_INFO
    from egom2p.models.generate import (
        GenerationSampler,
        build_chained_generation_schedules,
        init_empty_target_modality,
        init_full_input_modality,
    )
    from egom2p.utils.data_constants import CAM_MEAN, CAM_STD
    from run_training_egom2p import get_model
    from run_training_vqvae import get_model as get_tokenizer_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera_checkpoint = torch.load(context.checkpoint(1), map_location="cpu", weights_only=False)
    camera_tokenizer = get_tokenizer_model(camera_checkpoint["args"], device)
    camera_tokenizer.load_state_dict(camera_checkpoint["model"])
    camera_tokenizer = camera_tokenizer.eval()

    checkpoint = torch.load(context.checkpoint(0), map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if not hasattr(args, "in_domains"):
        args.in_domains = ["tok_rgb"]
    if not hasattr(args, "out_domains"):
        args.out_domains = ["tok_cam"]
    domains = sorted(set(args.in_domains) | set(args.out_domains))
    modality_info = {domain: MODALITY_INFO[domain] for domain in domains}
    model = get_model(args, modality_info)
    model.load_state_dict(checkpoint["model"])
    model = model.eval().to(device)
    sampler = GenerationSampler(model)
    cosmos = CausalVideoTokenizer(checkpoint_enc=str(context.checkpoint(2)), device=device)
    schedule = build_chained_generation_schedules(
        cond_domains=["tok_rgb"],
        target_domains=["tok_cam"],
        tokens_per_target=[30],
        autoregression_schemes=["roar"],
        decoding_steps=[3],
        token_decoding_schedules=["linear"],
        temps=[0.01],
        temp_schedules=["constant"],
        cfg_scales=[2.0],
        cfg_schedules=["constant"],
        cfg_grow_conditioning=True,
    )
    context.mark_model_ready()

    frame_count = int(context.parameters.get("rgb_frames", 16))
    resolution = int(context.parameters.get("resolution", 256))
    window_size = max(2, int(round(float(context.parameters.get("window_sec", 2.0)) * 10.0)))
    overlap = int(round(float(context.parameters.get("overlap_sec", 1.0)) * 10.0))
    mean = np.asarray(CAM_MEAN, dtype=np.float64).reshape(1, 1, -1)
    std = np.asarray(CAM_STD, dtype=np.float64).reshape(1, 1, -1)
    windows = []
    for window_index, bounds in enumerate(window_slices(len(context.frames), window_size, overlap)):
        rows = context.frames[bounds]
        clip = rgb_clip(rows, frame_count, resolution)
        with torch.inference_mode():
            encoded = cosmos(clip, temporal_window=frame_count)
        tokens = torch.as_tensor(np.asarray(encoded), dtype=torch.int64).reshape(1, -1).to(device)
        sample = {
            "tok_rgb": {
                "tensor": tokens,
                "input_mask": torch.zeros(1, 5120, dtype=torch.bool, device=device),
                "target_mask": torch.ones(1, 5120, dtype=torch.bool, device=device),
            }
        }
        sample = init_empty_target_modality(sample, MODALITY_INFO, "tok_cam", 1, 30, device)
        sample = init_full_input_modality(sample, MODALITY_INFO, "tok_rgb", device)
        generated = sampler.generate(
            sample, schedule, verbose=False, seed=context.seed, top_p=0.8, top_k=0.0
        )
        decoded = camera_tokenizer.decode_tokens(generated["tok_cam"]["tensor"])
        camera_9d = decoded.detach().float().cpu().numpy() * std + mean
        windows.append(camera_window(rows, np.squeeze(camera_9d, axis=0)))
        if window_index == 0:
            context.mark_first_prediction()
        del sample, generated, decoded, clip, encoded
        if torch.cuda.is_available():
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
            "window_sec": 2.0,
            "overlap_sec": 1.0,
        }
    )
    return result
