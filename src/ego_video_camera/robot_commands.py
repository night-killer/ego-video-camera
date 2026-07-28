from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


def _shell_join(parts: list[str]) -> str:
    return (" \\" + "\n  ").join(shlex.quote(str(part)) for part in parts)


def _shell_function(name: str, command: list[str], *, cuda: bool = True) -> str:
    prefix = "CUDA_VISIBLE_DEVICES=7 " if cuda else ""
    rendered = prefix + _shell_join(command)
    body = "\n".join(f"  {line}" for line in rendered.splitlines())
    return f"{name}() {{\n{body}\n}}"


def generate_robot_gpu_commands(
    repo_root: str | Path,
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    checkpoint: str | Path,
    clips: list[dict[str, Any]],
    python_path: str = "/data/aigc/cyb/zxgu/env/worldsearcher/bin/python",
) -> str:
    repo = Path(repo_root).resolve()
    script = repo / "scripts" / "run_robot_demo.py"
    config = Path(config_path).resolve()
    data = Path(data_root).resolve()
    output = Path(output_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    droid = [item for item in clips if item["dataset"] == "droid_wrist"]
    if not droid:
        raise ValueError("Robot GPU commands require at least one DROID clip")
    common = [
        python_path,
        str(script),
        "--config",
        str(config),
        "--data-root",
        str(data),
        "--output-root",
        str(output),
        "--checkpoint",
        str(checkpoint),
    ]

    def formal(dataset: str, *, compose: bool = False) -> list[str]:
        command = [
            *common,
            "--dataset",
            dataset,
            "--run-da3",
            "--render-comparison",
            "--evaluate",
            "--resume",
        ]
        if compose:
            command.append("--compose-all")
        return command

    prepare_exo = [*common, "--dataset", "all", "--prepare-exo"]
    smoke = [
        python_path,
        str(script),
        "--config",
        str(config),
        "--data-root",
        str(data),
        "--output-root",
        str(output / "_gpu_smoke"),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        "droid",
        "--sequence-id",
        str(droid[0]["sequence_id"]),
        "--duration-sec",
        "5",
        "--sample-fps",
        "5",
        "--run-da3",
        "--render-comparison",
        "--evaluate",
        "--resume",
    ]
    compose = [*common, "--dataset", "all", "--compose-all"]
    oom_fps = [
        *formal("all", compose=True),
        "--sample-fps",
        "5",
    ]
    oom_392 = [
        *formal("all", compose=True),
        "--sample-fps",
        "5",
        "--input-resolution",
        "392",
        "--window-size",
        "30",
        "--window-overlap",
        "15",
    ]
    oom_336 = [
        *formal("all", compose=True),
        "--sample-fps",
        "5",
        "--input-resolution",
        "336",
        "--window-size",
        "20",
        "--window-overlap",
        "10",
    ]
    sections = [
        "#!/usr/bin/env bash\nset -euo pipefail",
        "# DA3 robot Ego/Exo commands. An action is required; nothing runs by default.",
        "# Prepare all exo artifacts on CPU. May download the 27.4 GB RH20T cfg3 archive.\n"
        + _shell_function("run_prepare_exo", prepare_exo, cuda=False),
        "# Five-second DROID smoke run\n" + _shell_function("run_smoke", smoke),
        "# Three selected DROID clips\n"
        + _shell_function("run_droid", formal("droid")),
        "# Four selected RH20T clips\n"
        + _shell_function("run_rh20t", formal("rh20t")),
        "# Seven formal clips followed by prefix composition\n"
        + _shell_function("run_formal_all", formal("all", compose=True)),
        "# Compose existing prefix videos\n"
        + _shell_function("run_compose", compose, cuda=False),
        "# OOM fallback 1: 10 to 5 FPS, keep 504 and 60/30\n"
        + _shell_function("run_oom_fps", oom_fps),
        "# OOM fallback 2: 5 FPS, resolution 392 and chunks 30/15\n"
        + _shell_function("run_oom_392", oom_392),
        "# OOM fallback 3: 5 FPS, resolution 336 and chunks 20/10\n"
        + _shell_function("run_oom_336", oom_336),
        """action=${1:-help}
case "$action" in
  prepare-exo) run_prepare_exo ;;
  smoke) run_smoke ;;
  droid) run_droid ;;
  rh20t) run_rh20t ;;
  formal-all) run_formal_all ;;
  compose) run_compose ;;
  oom-fps) run_oom_fps ;;
  oom-392) run_oom_392 ;;
  oom-336) run_oom_336 ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: gpu_commands.sh ACTION

Actions:
  prepare-exo  Prepare all exo data; may download the 27.4 GB RH20T cfg3 archive
  smoke       Five-second DROID real-DA3 smoke test
  droid       Run the three selected DROID clips
  rh20t       Run the four selected RH20T clips
  formal-all  Run all seven clips and compose prefix videos
  compose     Compose already completed prefix videos
  oom-fps     All clips at 5 FPS, resolution 504, chunks 60/30
  oom-392     All clips at 5 FPS, resolution 392, chunks 30/15
  oom-336     All clips at 5 FPS, resolution 336, chunks 20/10
USAGE
    ;;
  *)
    echo "Unknown action: $action (use --help)" >&2
    exit 2
    ;;
esac""",
    ]
    return "\n\n".join(sections) + "\n"
