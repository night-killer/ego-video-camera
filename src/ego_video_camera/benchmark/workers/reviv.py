from __future__ import annotations

import numpy as np

from ..windowing import stitch_pose_windows, window_slices
from .common import WorkerContext, import_file
from .motion_tokens import camera_window, rgb_clip


def run(context: WorkerContext):
    import torch
    from tokenizers import Tokenizer

    demo = import_file("ego_benchmark_reviv_demo", context.repo / "demo_infer.py")
    model, all_domains = demo.load_main_model(str(context.checkpoint(0)))
    pathway = demo.select_pathway(all_domains)
    expected_pathway = "512" if int(context.parameters.get("resolution", 512)) == 512 else "256"
    if pathway != expected_pathway:
        raise ValueError(f"Configured ReViV pathway {expected_pathway}, checkpoint is {pathway}")
    camera_tokenizer = demo.load_motion_tokenizer(str(context.checkpoint(1)))
    sampler = demo.GenerationSampler(model)
    text_tokenizer = Tokenizer.from_file(
        str(context.repo / "reviv" / "utils" / "tokenizer" / "trained" / "text_tokenizer_reviv_wordpiece_30k.json")
    )
    cosmos = demo.CausalVideoTokenizer(
        checkpoint_enc=str(context.checkpoint(4)), device=demo.DEVICE
    )
    context.mark_model_ready()

    mean = np.load(context.checkpoint(2)).reshape(1, 1, -1)
    std = np.load(context.checkpoint(3)).reshape(1, 1, -1)
    frame_count = int(context.parameters.get("rgb_frames", 32))
    resolution = int(context.parameters.get("resolution", 512))
    window_size = max(2, int(round(float(context.parameters.get("window_sec", 2.0)) * 10.0)))
    overlap = int(round(float(context.parameters.get("overlap_sec", 1.0)) * 10.0))
    windows = []
    pw = demo.PATHWAYS[pathway]
    for window_index, bounds in enumerate(window_slices(len(context.frames), window_size, overlap)):
        rows = context.frames[bounds]
        clip = rgb_clip(rows, frame_count, resolution)
        with torch.inference_mode():
            encoded = cosmos(clip, temporal_window=frame_count)
        tokens = torch.as_tensor(np.asarray(encoded), dtype=torch.int64).reshape(1, -1).cpu()
        conditions = {pw["cond_domains"][0]: tokens}
        if pw["raw_clip_domain"] in pw["cond_domains"]:
            conditions[pw["raw_clip_domain"]] = torch.from_numpy(clip).float().div(255).mul(2).sub(1)
        sample = demo.build_sample(conditions, "tok_cam", pw["target_tokens"]["tok_cam"], text_tokenizer)
        schedule = demo.build_schedule(pw["cond_domains"], "tok_cam", pw["target_tokens"]["tok_cam"])
        with demo.generation_context("bf16"):
            generated = sampler.generate(
                sample,
                schedule,
                text_tokenizer=text_tokenizer,
                verbose=False,
                seed=context.seed,
                top_p=0.8,
                top_k=0.0,
            )
        decoded = camera_tokenizer.decode_tokens(generated["tok_cam"]["tensor"])
        camera_9d = decoded.detach().float().cpu().numpy() * std + mean
        camera_9d = np.squeeze(camera_9d, axis=0)
        windows.append(camera_window(rows, camera_9d))
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
            "pathway": pathway,
            "window_sec": 2.0,
            "overlap_sec": 1.0,
            "gt_sidecars_used": False,
        }
    )
    return result
