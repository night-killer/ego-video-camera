import subprocess
from pathlib import Path

from ego_video_camera.commands import generate_gpu_commands


def test_generated_gpu_commands_are_valid_shell_and_have_real_recordings(tmp_path: Path):
    selected = {
        "clips": {
            "easy": {"recording_name": "recording_easy"},
            "medium": {"recording_name": "recording_medium"},
            "hard": {"recording_name": "recording_hard"},
        }
    }
    content = generate_gpu_commands(
        tmp_path,
        tmp_path / "config.yaml",
        tmp_path / "data",
        tmp_path / "output",
        tmp_path / "checkpoint",
        selected,
    )
    path = tmp_path / "gpu_commands.sh"
    path.write_text(content, encoding="utf-8")
    subprocess.run(["bash", "-n", str(path)], check=True)
    help_result = subprocess.run(
        ["bash", str(path)], check=True, capture_output=True, text=True
    )
    assert "Usage: gpu_commands.sh ACTION" in help_result.stdout
    assert "recording_easy" in content
    assert "recording_medium" in content
    assert "recording_hard" in content
    assert "formal-all) run_formal_all" in content
    assert "--run-selected-clips" in content
    assert "--compose-all-toys" in content
    assert "--input-resolution \\" + "\n    336" in content
