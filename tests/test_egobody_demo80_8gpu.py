import json
import os
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_egobody_demo80_8gpu.sh"


def test_demo80_8gpu_launcher_has_balanced_deterministic_plan(tmp_path: Path):
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    env = {
        **os.environ,
        "DEMO80_PLAN_ONLY": "1",
        "DEMO80_OUTPUT_ROOT": str(tmp_path / "output"),
    }
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    plan = json.loads((tmp_path / "output/multi_gpu_plan.json").read_text())
    assert plan["gpu_count"] == 8
    assert plan["clip_count"] == 80
    assert len(plan["shards"]) == 8
    assert all(shard["clip_count"] == 10 for shard in plan["shards"])
    assert all(shard["expected_sampled_frame_count"] == 1040 for shard in plan["shards"])
    assert plan["shards"][0]["clips"] == [
        "DESK_001", "WALK_001", "DESK_009", "WALK_009", "DESK_017",
        "WALK_017", "DESK_025", "WALK_025", "DESK_033", "WALK_033",
    ]
    clips = [clip for shard in plan["shards"] for clip in shard["clips"]]
    assert len(clips) == len(set(clips)) == 80
    assert "plan-only complete" in completed.stderr


def test_demo80_8gpu_launcher_runs_workers_and_aggregates_summaries(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 0 1 2 3 4 5 6 7\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            command = "validate" if "validate" in args else "run"
            if command == "validate":
                print(json.dumps({"status": "ok", "clip_count": 80}))
                raise SystemExit(0)
            clips = [args[index + 1] for index, value in enumerate(args) if value == "--clip-id"]
            summary_arg = args[args.index("--summary-path") + 1]
            output_arg = Path(args[args.index("--output-root") + 1])
            summary = Path(summary_arg)
            if not summary.is_absolute():
                summary = output_arg / summary
            summary.parent.mkdir(parents=True, exist_ok=True)
            results = [
                {
                    "clip_id": clip_id,
                    "status": "ok",
                    "output_dir": str(output_arg / clip_id),
                    "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
                for clip_id in clips
            ]
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "clip_count": len(results),
                        "succeeded": len(results),
                        "failed": 0,
                        "clips": results,
                    }
                ),
                encoding="utf-8",
            )
            print(json.dumps({"status": "ok", "clip_count": len(results)}))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    stale_summary = output_root / "launcher_logs/gpu0.run_summary.json"
    stale_summary.parent.mkdir(parents=True)
    stale_summary.write_text('{"clips":[{"clip_id":"STALE","status":"ok"}]}', encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DEMO80_RUNNER": str(runner),
        "DEMO80_OUTPUT_ROOT": str(output_root),
    }

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    summary = json.loads((output_root / "run_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["clip_count"] == summary["succeeded"] == 80
    assert summary["failed"] == 0
    assert len(summary["workers"]) == 8
    assert {item["gpu"] for item in summary["clips"]} == set(range(8))
    assert {item["visible_gpu"] for item in summary["clips"]} == {
        str(index) for index in range(8)
    }
    assert all(worker["reported_clip_count"] == 10 for worker in summary["workers"])
    assert "all 80 clips completed" in completed.stderr


def test_demo80_8gpu_launcher_marks_summary_failed_when_worker_exits_nonzero(
    tmp_path: Path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 0 1 2 3 4 5 6 7\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            if "validate" in args:
                print(json.dumps({"status": "ok", "clip_count": 80}))
                raise SystemExit(0)

            clips = [args[index + 1] for index, value in enumerate(args) if value == "--clip-id"]
            summary = Path(args[args.index("--summary-path") + 1])
            summary.parent.mkdir(parents=True, exist_ok=True)
            results = [{"clip_id": clip_id, "status": "ok"} for clip_id in clips]
            summary.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "clip_count": len(results),
                        "succeeded": len(results),
                        "failed": 0,
                        "clips": results,
                    }
                ),
                encoding="utf-8",
            )
            raise SystemExit(7 if os.environ.get("CUDA_VISIBLE_DEVICES") == "3" else 0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DEMO80_RUNNER": str(runner),
        "DEMO80_OUTPUT_ROOT": str(output_root),
    }

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    summary = json.loads((output_root / "run_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["worker_failures"] == 1
    assert summary["failed"] == 0
    failed_worker = next(worker for worker in summary["workers"] if worker["gpu"] == 3)
    assert failed_worker["exit_code"] == 7


def test_demo80_8gpu_launcher_removes_stale_summary_before_validation(
    tmp_path: Path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 0 1 2 3 4 5 6 7\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        "import sys\nraise SystemExit(7 if 'validate' in sys.argv else 0)\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    stale_summary = output_root / "run_summary.json"
    stale_summary.write_text(
        json.dumps({"status": "ok", "clip_count": 80}),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DEMO80_RUNNER": str(runner),
        "DEMO80_OUTPUT_ROOT": str(output_root),
    }

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert not stale_summary.exists()
