#!/usr/bin/env python3
"""Static imports and lightweight H100 checks for one benchmark environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path


def prepend(*paths: Path) -> None:
    for path in reversed(paths):
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_gpu(require_h100: bool) -> None:
    import torch

    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
    if not require_h100:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    capability = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if capability != (9, 0) or "H100" not in name:
        raise RuntimeError(f"expected H100 sm_90, found {name} capability={capability}")
    torch.ones(1, device="cuda").add_(1).cpu()
    print(f"gpu={name} capability=sm_{capability[0]}{capability[1]}")


def check_droid(root: Path) -> None:
    prepend(root / "thirdparty" / "DROID-SLAM" / "droid_slam")
    check_scatter_lietorch_and_droid()
    importlib.import_module("droid")


def check_worldsearcher(root: Path) -> None:
    import numpy as np

    print(f"numpy={np.__version__}")
    prepend(root / "thirdparty" / "Depth-Anything-3", root / "thirdparty" / "Depth-Anything-3" / "src")
    importlib.import_module("depth_anything_3.api")
    prepend(root / "thirdparty" / "VGGT-Omega")
    importlib.import_module("vggt_omega.models")
    prepend(root / "thirdparty" / "LingBot-Map")
    import_file("_ego_eval_lingbot_demo", root / "thirdparty" / "LingBot-Map" / "demo.py")
    importlib.import_module("cv2")
    importlib.import_module("gdown")

    if importlib.util.find_spec("vipe_ext") is None:
        raise RuntimeError("vipe_ext is not installed; run prepare_worldsearcher.sh")
    prepend(root / "thirdparty" / "ViPE")
    from vipe.ext.lietorch import SE3

    value = SE3.Identity(1, device="cuda").matrix()
    if value.shape != (1, 4, 4):
        raise RuntimeError(f"unexpected ViPE lietorch output: {value.shape}")
    importlib.import_module("vipe.slam.system")


def check_vggt_slam(root: Path) -> None:
    prepend(root / "thirdparty" / "VGGT-SLAM")
    importlib.import_module("salad.eval")
    importlib.import_module("vggt.models.vggt")
    importlib.import_module("vggt_slam.solver")


def check_reviv(root: Path) -> None:
    prepend(root / "thirdparty" / "ReViV")
    importlib.import_module("tokenizers")
    importlib.import_module("reviv.models.generate")
    importlib.import_module("run_training_reviv")
    importlib.import_module("run_training_vqvae")


def check_egom2p(root: Path) -> None:
    prepend(root / "thirdparty" / "EgoM2P")
    importlib.import_module("cosmos_tokenizer.video_lib")
    importlib.import_module("egom2p.models.generate")
    importlib.import_module("run_training_egom2p")
    importlib.import_module("run_training_vqvae")


def check_megasam(root: Path) -> None:
    mega = root / "thirdparty" / "MegaSaM"
    prepend(mega / "Depth-Anything", mega / "UniDepth", mega / "base" / "droid_slam")
    importlib.import_module("depth_anything.dpt")
    importlib.import_module("unidepth.models")
    importlib.import_module("xformers")
    importlib.import_module("droid")
    check_scatter_lietorch_and_droid()

    import torch
    from xformers.ops import memory_efficient_attention

    query = torch.randn(1, 8, 4, 32, device="cuda", dtype=torch.float16)
    attention = memory_efficient_attention(query, query, query)
    if attention.shape != query.shape:
        raise RuntimeError(f"unexpected xFormers output: {attention.shape}")
    torch.cuda.synchronize()


def check_scatter_lietorch_and_droid() -> None:
    import droid_backends
    import torch
    import torch_scatter
    from lietorch import SE3

    SE3.Identity(1, device="cuda").matrix()
    source = torch.ones(2, device="cuda")
    index = torch.tensor([0, 0], device="cuda")
    if torch_scatter.scatter_sum(source, index).item() != 2.0:
        raise RuntimeError("torch-scatter CUDA smoke test failed")
    volume = torch.ones((1, 2, 2, 2, 2), device="cuda")
    coords = torch.zeros((1, 2, 2, 2), device="cuda")
    (correlation,) = droid_backends.corr_index_forward(volume, coords, 0)
    if correlation.shape != (1, 1, 1, 2, 2) or correlation.min().item() != 1.0:
        raise RuntimeError(f"unexpected DROID correlation output: {correlation.shape}")


def check_hawor(root: Path) -> None:
    hawor = root / "thirdparty" / "HaWoR"
    prepend(
        hawor,
        hawor / "thirdparty" / "Metric3D",
        hawor / "thirdparty" / "DROID-SLAM" / "droid_slam",
    )
    importlib.import_module("mmcv")
    importlib.import_module("pytorch3d._C")
    importlib.import_module("scripts.scripts_test_video.detect_track_video")
    importlib.import_module("scripts.scripts_test_video.hawor_video")
    importlib.import_module("scripts.scripts_test_video.hawor_slam")
    check_scatter_lietorch_and_droid()

    import torch
    from pytorch3d.ops import knn_points

    points = torch.rand(1, 4, 3, device="cuda")
    nearest = knn_points(points, points, K=1)
    if nearest.dists.shape != (1, 4, 1) or nearest.dists.max().item() != 0.0:
        raise RuntimeError("PyTorch3D CUDA KNN smoke test failed")


def check_egoego(root: Path) -> None:
    prepend(
        root / "thirdparty" / "EgoEgo",
        root / "thirdparty" / "DROID-SLAM" / "droid_slam",
        root / "thirdparty" / "MegaSaM" / "cvd_opt" / "core",
    )
    importlib.import_module("pytorch3d.transforms")
    importlib.import_module("raft")
    importlib.import_module("egoego.model.head_estimation_transformer")
    importlib.import_module("egoego.model.head_normal_estimation_transformer")
    importlib.import_module("droid")
    check_scatter_lietorch_and_droid()


CHECKS = {
    "worldsearcher": check_worldsearcher,
    "vggt_slam": check_vggt_slam,
    "reviv": check_reviv,
    "egom2p": check_egom2p,
    "droid_slam": check_droid,
    "megasam": check_megasam,
    "hawor": check_hawor,
    "egoego": check_egoego,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(CHECKS), required=True)
    parser.add_argument("--require-h100", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    check_gpu(args.require_h100)
    CHECKS[args.method](root)
    print(f"{args.method}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
