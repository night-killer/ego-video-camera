from __future__ import annotations

import importlib
import gc
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch

from .download import sha256_file
from .serialization import write_json


def _command_version(path: str, args: list[str]) -> str | None:
    if not Path(path).exists() and shutil.which(path) is None:
        return None
    completed = subprocess.run([path, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return completed.stdout.splitlines()[0] if completed.stdout else None


def collect_system_info(repo_root: str | Path, ffmpeg_path: str, ffprobe_path: str) -> dict:
    repo_root = Path(repo_root)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=repo_root, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.splitlines()
    submodules = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    disk = shutil.disk_usage(repo_root)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "ffmpeg": _command_version(ffmpeg_path, ["-version"]),
        "ffprobe": _command_version(ffprobe_path, ["-version"]),
        "git_commit": git_commit,
        "git_status": git_status,
        "submodules": submodules,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
    }


def inspect_checkpoint(checkpoint_path: str | Path) -> dict:
    checkpoint = Path(checkpoint_path)
    config_path = checkpoint / "config.json"
    weight_path = checkpoint / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from safetensors import safe_open

    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    return {
        "checkpoint_path": str(checkpoint),
        "status": "user_validated_local",
        "model_name": config.get("model_name"),
        "config_path": str(config_path),
        "config_size": config_path.stat().st_size,
        "weight_path": str(weight_path),
        "weight_size": weight_path.stat().st_size,
        "weight_sha256": sha256_file(weight_path),
        "tensor_count": len(keys),
        "first_tensor_keys": keys[:10],
        "partial_files_ignored": [str(path) for path in checkpoint.glob("*.partial")],
    }


def check_da3_import(repo_root: str | Path, source_root: str | Path) -> dict:
    from .da3_adapter import activate_da3_source

    source_repo = activate_da3_source(repo_root, source_root)
    source = (source_repo / "src").resolve()
    module = importlib.import_module("depth_anything_3.api")
    module_file = Path(module.__file__).resolve()
    if source not in module_file.parents:
        raise RuntimeError(f"DA3 imported from outside the pinned submodule: {module_file}")
    return {"status": "ok", "module_file": str(module_file), "source_root": str(source)}


def verify_da3_load_only(
    repo_root: str | Path, source_root: str | Path, checkpoint_path: str | Path
) -> dict:
    from .da3_adapter import load_model_only

    started = time.monotonic()
    model = load_model_only(repo_root, source_root, checkpoint_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "ok",
        "operation": "cpu_load_only_no_inference",
        "parameter_count": int(parameter_count),
        "device": str(next(model.parameters()).device),
        "elapsed_seconds": time.monotonic() - started,
    }
    del model
    gc.collect()
    return result


def write_inventories(
    repo_root: str | Path,
    output_root: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    verify_model_load: bool = False,
) -> tuple[dict, dict]:
    system = collect_system_info(repo_root, ffmpeg_path, ffprobe_path)
    model = inspect_checkpoint(checkpoint_path)
    model["source_import"] = check_da3_import(repo_root, source_root)
    model["load_only_check"] = (
        verify_da3_load_only(repo_root, source_root, checkpoint_path)
        if verify_model_load
        else {"status": "not_requested", "operation": "cpu_load_only_no_inference"}
    )
    write_json(Path(output_root) / "system_info.json", system)
    write_json(Path(output_root) / "model_inventory.json", model)
    return system, model
