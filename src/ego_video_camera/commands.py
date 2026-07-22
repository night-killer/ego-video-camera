from __future__ import annotations

import shlex
from pathlib import Path


def _shell_join(parts: list[str]) -> str:
    separator = " \\" + "\n  "
    return separator.join(shlex.quote(str(part)) for part in parts)


def _shell_function(name: str, command: list[str], *, cuda: bool = True) -> str:
    prefix = "CUDA_VISIBLE_DEVICES=0 " if cuda else ""
    rendered = prefix + _shell_join(command)
    body = "\n".join(f"  {line}" for line in rendered.splitlines())
    return f"{name}() {{\n{body}\n}}"


def generate_gpu_commands(
    repo_root: str | Path,
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    checkpoint: str | Path,
    selected: dict,
) -> str:
    repo = Path(repo_root).resolve()
    python = "/data/aigc/cyb/zxgu/env/worldsearcher/bin/python"
    script = repo / "scripts" / "run_egobody_demo.py"
    config = Path(config_path).resolve()
    data = Path(data_root).resolve()
    output = Path(output_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    selected_path = output / "selection" / "selected_clips.json"
    easy = selected["clips"]["easy"]
    common = [
        python,
        str(script),
        "--config",
        str(config),
        "--data-root",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(output),
        "--selected-clips",
        str(selected_path),
    ]
    smoke = [
        python,
        str(script),
        "--config",
        str(config),
        "--data-root",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(output / "_gpu_smoke"),
        "--selected-clips",
        str(selected_path),
        "--sequence-id",
        easy["recording_name"],
        "--duration-sec",
        "5",
        "--sample-fps",
        "5",
        "--run-da3",
        "--render-comparison",
        "--evaluate",
        "--resume",
    ]
    clip_commands = []
    for difficulty in ("easy", "medium", "hard"):
        clip = selected["clips"][difficulty]
        clip_commands.append(
            (
                difficulty,
                [
                    *common,
                    "--sequence-id",
                    clip["recording_name"],
                    "--run-da3",
                    "--render-comparison",
                    "--evaluate",
                    "--resume",
                ],
            )
        )
    compose = [*common, "--compose-all-toys"]
    formal_all = [
        *common,
        "--run-selected-clips",
        "--run-da3",
        "--render-comparison",
        "--evaluate",
        "--compose-all-toys",
        "--resume",
    ]
    oom_fps = [
        *common,
        "--run-selected-clips",
        "--sample-fps",
        "5",
        "--run-da3",
        "--render-comparison",
        "--evaluate",
        "--compose-all-toys",
        "--resume",
    ]
    oom_392 = [
        *common,
        "--run-selected-clips",
        "--sample-fps",
        "5",
        "--input-resolution",
        "392",
        "--window-size",
        "30",
        "--window-overlap",
        "15",
        "--run-da3",
        "--render-comparison",
        "--evaluate",
        "--compose-all-toys",
        "--resume",
    ]
    oom_336 = [
        *common,
        "--run-selected-clips",
        "--sample-fps",
        "5",
        "--input-resolution",
        "336",
        "--window-size",
        "20",
        "--window-overlap",
        "10",
        "--run-da3",
        "--render-comparison",
        "--evaluate",
        "--compose-all-toys",
        "--resume",
    ]
    sections = [
        "#!/usr/bin/env bash\nset -euo pipefail",
        (
            "# Generated from actual EgoBody selection; checkpoint status: "
            "user_validated_local\n"
            "# Commands run only when their explicit action is selected."
        ),
        "# GPU smoke test (Easy, 5 seconds at 5 FPS)\n"
        + _shell_function("run_smoke", smoke),
    ]
    sections.extend(
        f"# {difficulty.capitalize()} formal clip: {selected['clips'][difficulty]['recording_name']}\n"
        + _shell_function(f"run_{difficulty}", command)
        for difficulty, command in clip_commands
    )
    sections.extend(
        [
            "# One normal-config command runs all clips and then composes them\n"
            + _shell_function("run_formal_all", formal_all),
            "# Compose Easy → Medium → Hard after three separate formal runs\n"
            + _shell_function("run_compose", compose, cuda=False),
            "# OOM fallback 1: 8 → 5 FPS; keep 504 and 60/30\n"
            + _shell_function("run_oom_fps", oom_fps),
            "# OOM fallback 2: 392 resolution and 30/15 chunks\n"
            + _shell_function("run_oom_392", oom_392),
            "# OOM fallback 3: 336 resolution and 20/10 chunks; model is unchanged\n"
            + _shell_function("run_oom_336", oom_336),
            """action=${1:-help}
case "$action" in
  smoke) run_smoke ;;
  easy) run_easy ;;
  medium) run_medium ;;
  hard) run_hard ;;
  formal-all) run_formal_all ;;
  all-separate) run_easy; run_medium; run_hard; run_compose ;;
  compose) run_compose ;;
  oom-fps) run_oom_fps ;;
  oom-392) run_oom_392 ;;
  oom-336) run_oom_336 ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: gpu_commands.sh ACTION

Actions:
  smoke        Easy 5-second/5-FPS real-DA3 smoke test
  formal-all   Run Easy, Medium, Hard at the normal config, then compose
  all-separate Run the three per-recording commands, then compose
  easy|medium|hard
               Run one formal clip
  compose      Compose three already completed formal clips
  oom-fps      All clips at 5 FPS, resolution 504, chunks 60/30
  oom-392      All clips at 5 FPS, resolution 392, chunks 30/15
  oom-336      All clips at 5 FPS, resolution 336, chunks 20/10
USAGE
    ;;
  *)
    echo "Unknown action: $action (use --help)" >&2
    exit 2
    ;;
esac""",
        ]
    )
    return "\n\n".join(sections) + "\n"
