import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ego_video_camera.benchmark.preflight import preflight_report
from ego_video_camera.benchmark.registry import sequence_intrinsics
from ego_video_camera.benchmark.schema import MethodSpec, SequenceRecord
from ego_video_camera.benchmark.workers import droid_slam, motion_tokens, vipe
from ego_video_camera.benchmark.workers.vggt_omega import validate_image_resolution
from ego_video_camera.openloris import read_camera_intrinsics


ROOT = Path(__file__).resolve().parents[1]


OPENLORIS_YAML = """%YAML:1.0
d400_color_optical_frame:
   sensor_type: camera
   sensor_name: d400_color_optical_frame
   fps: 30
   width: 848
   height: 480
   model: pinhole
   intrinsics: !!opencv-matrix
      rows: 1
      cols: 4
      dt: d
      data: [ 6.1145098876953125e+02, 4.3320397949218750e+02,
          6.1148571777343750e+02, 2.4947302246093750e+02 ]
"""


def _method(tmp_path: Path, **overrides) -> MethodSpec:
    values = {
        "method_id": "method",
        "family": "method",
        "display_name": "Method",
        "adapter": "example:run",
        "conda_env": "test",
        "repo": tmp_path,
        "checkpoint_paths": (),
        "seeds": (0,),
        "input_intrinsics": "not_used",
        "causal": False,
        "metric_scale": False,
    }
    values.update(overrides)
    return MethodSpec(**values)


def _sequence(tmp_path: Path, dataset_id: str = "dataset", frame_count: int = 2):
    clip_json = tmp_path / "clip.json"
    clip_json.write_text("{}\n", encoding="utf-8")
    return SequenceRecord(
        dataset_id=dataset_id,
        sequence_id="sequence",
        clip_dir=tmp_path,
        clip_json=clip_json,
        input_path=tmp_path,
        duration_sec=frame_count / 10.0,
        target_fps=10.0,
        reference_grade="A",
        reference_type="test",
        stratum="test",
        start_sec=0.0,
        frame_count=frame_count,
        input_kind="frames",
    )


def test_droid_worker_adds_source_directory_once(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(droid_slam.sys, "path", list(sys.path))
    context = SimpleNamespace(repo=tmp_path)
    expected = str(tmp_path / "droid_slam")

    droid_slam._add_import_path(context)
    droid_slam._add_import_path(context)

    assert droid_slam.sys.path[0] == expected
    assert droid_slam.sys.path.count(expected) == 1


def test_vggt_omega_rejects_non_checkpoint_resolution(tmp_path: Path):
    assert validate_image_resolution({"image_resolution": 512}) == 512
    with pytest.raises(ValueError, match="requires image_resolution=512"):
        validate_image_resolution({"image_resolution": 518})

    report = preflight_report(
        {"benchmark": {}, "_repo_root": str(tmp_path)},
        [
            _method(
                tmp_path,
                method_id="vggt_omega",
                family="vggt_omega",
                parameters={"image_resolution": 518},
            )
        ],
        [],
        check_environments=False,
    )
    failed = [check for check in report["checks"] if not check["ok"]]
    assert report["status"] == "failed"
    assert [check["kind"] for check in failed] == ["method_parameter"]


def test_rgb_clip_falls_back_to_pillow(tmp_path: Path, monkeypatch):
    path = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), color=(240, 20, 10)).save(path)

    import cv2

    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: None)
    clip = motion_tokens.rgb_clip(
        [{"image_path": str(path)}], frame_count=2, resolution=4
    )

    assert clip.shape == (1, 2, 4, 4, 3)
    assert clip.dtype == np.uint8
    assert np.array_equal(clip[0, 0, 0, 0], [240, 20, 10])


def test_vipe_manifest_tensors_share_requested_device():
    import torch

    image = np.full((3, 4, 3), 255, dtype=np.uint8)
    intrinsic = np.asarray(
        [[500.0, 0.0, 2.0], [0.0, 510.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    rgb, camera = vipe._frame_tensors(image, intrinsic, torch.device("cpu"))

    assert rgb.device == camera.device == torch.device("cpu")
    assert rgb.dtype == camera.dtype == torch.float32
    assert torch.equal(camera, torch.tensor([500.0, 510.0, 2.0, 1.5]))
    assert torch.all(rgb == 1.0)


def test_openloris_opencv_yaml_and_registry_fallback(tmp_path: Path):
    reference = tmp_path / "reference"
    reference.mkdir()
    sensors = reference / "sensors.yaml"
    sensors.write_text(OPENLORIS_YAML, encoding="utf-8")
    (reference / "camera.json").write_text(
        json.dumps(
            {
                "camera_key": "d400_color_optical_frame",
                "projection_model": "pinhole",
                "width": 848,
                "height": 480,
            }
        ),
        encoding="utf-8",
    )

    calibration = read_camera_intrinsics(sensors)
    assert calibration["fx"] == pytest.approx(611.4509887695312)
    assert calibration["fy"] == pytest.approx(611.4857177734375)
    assert calibration["cx"] == pytest.approx(433.2039794921875)
    assert calibration["cy"] == pytest.approx(249.4730224609375)

    matrices = sequence_intrinsics(
        _sequence(tmp_path, dataset_id="openloris_office"), 2
    )
    assert matrices is not None
    assert matrices.shape == (2, 3, 3)
    assert matrices[0, 0, 0] == pytest.approx(calibration["fx"])
    assert matrices[0, 1, 1] == pytest.approx(calibration["fy"])


def test_preflight_rejects_missing_required_intrinsics(tmp_path: Path):
    sequence = _sequence(tmp_path)
    method = _method(
        tmp_path,
        method_id="requires_camera",
        input_intrinsics="provided",
    )
    report = preflight_report(
        {"benchmark": {}, "_repo_root": str(tmp_path)},
        [method],
        [sequence],
        check_environments=False,
    )

    intrinsics = [check for check in report["checks"] if check["kind"] == "intrinsics"]
    assert report["status"] == "failed"
    assert len(intrinsics) == 1
    assert intrinsics[0]["target"] == sequence.key
    assert intrinsics[0]["ok"] is False
    assert "requires_camera" in intrinsics[0]["detail"]


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "thirdparty" / "DROID-SLAM" / "droid_slam" / "factor_graph.py",
        ROOT
        / "thirdparty"
        / "HaWoR"
        / "thirdparty"
        / "DROID-SLAM"
        / "droid_slam"
        / "factor_graph.py",
    ],
)
def test_droid_proximity_graph_handles_empty_edges(path: Path):
    source = path.read_text(encoding="utf-8")
    function = source[source.index("def add_proximity_factors") :]
    assert function.index("if not es:") < function.index("torch.as_tensor(es")


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "thirdparty" / "DROID-SLAM" / "droid_slam" / "droid_backend.py",
        ROOT
        / "thirdparty"
        / "HaWoR"
        / "thirdparty"
        / "DROID-SLAM"
        / "droid_slam"
        / "droid_backend.py",
    ],
)
def test_droid_backend_skips_empty_global_graph(path: Path):
    source = path.read_text(encoding="utf-8")
    function = source[source.index("def __call__") :]
    assert function.index("if graph.ii.numel() == 0:") < function.index(
        "graph.update_lowmem"
    )


def test_hawor_uses_adaptive_rasterizer_bins():
    source = (
        ROOT
        / "thirdparty"
        / "HaWoR"
        / "scripts"
        / "scripts_test_video"
        / "hawor_video.py"
    ).read_text(encoding="utf-8")
    assert "bin_size = None" in source
    assert "max_faces_per_bin = 20000" in source


def test_worldsearcher_preparation_pins_gdown_dependencies():
    preparation = (ROOT / "scripts/install_eval_envs/prepare_worldsearcher.sh").read_text(
        encoding="utf-8"
    )
    verification = (ROOT / "scripts/install_eval_envs/verify_env.py").read_text(
        encoding="utf-8"
    )
    for requirement in (
        "PySocks==1.7.1",
        "soupsieve==2.6",
        "beautifulsoup4==4.12.3",
        "gdown==5.2.0",
    ):
        assert requirement in preparation
    assert 'importlib.import_module("gdown")' in verification


def test_launcher_reports_after_reconciliation_failure(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "benchmark-python"
    fake_python.write_text(
        f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args and args[0] == "-":
    os.execv(sys.executable, [sys.executable, *args])

command = next(
    (
        value
        for value in args
        if value in {{"preflight", "plan", "run", "evaluate", "report"}}
    ),
    "unknown",
)
label = "shard" if command == "run" and "--sequences" in args else command
log_path = Path(os.environ["FAKE_CALL_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(label + "\\n")

output_root = Path(os.environ["FAKE_OUTPUT_ROOT"])
if command == "plan":
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = {{
        "sequences": [
            {{
                "dataset_id": "dataset",
                "sequence_id": "sequence",
                "duration_sec": 1.0,
            }}
        ]
    }}
    plan = {{
        "run_count": 1,
        "runs": [{{"dataset_id": "dataset", "sequence_id": "sequence"}}],
    }}
    (output_root / "inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    (output_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
elif command == "run" and "--sequences" not in args:
    raise SystemExit(7)
elif command == "report":
    (output_root / "report-generated").touch()
print("{{}}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\nprintf '0\\n'\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    output_root = tmp_path / "output"
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "schema_version: 1\nrepo_root: .\nbenchmark:\n  output_root: output\n  gpu: 0\n",
        encoding="utf-8",
    )
    call_log = tmp_path / "calls.log"
    environment = {
        **os.environ,
        "PATH": os.pathsep.join((str(fake_bin), os.environ.get("PATH", ""))),
        "BENCHMARK_PYTHON": str(fake_python),
        "BENCHMARK_CONFIG": str(config),
        "BENCHMARK_OUTPUT_ROOT": str(output_root),
        "BENCHMARK_GPUS": "0",
        "EVAL_ENV_ROOT": str(tmp_path / "envs"),
        "FAKE_CALL_LOG": str(call_log),
        "FAKE_OUTPUT_ROOT": str(output_root),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_pose_benchmark_8gpu.sh")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "preflight",
        "plan",
        "shard",
        "run",
        "evaluate",
        "report",
    ]
    assert (output_root / "report-generated").is_file()
    assert "reconcile=7, evaluate=0, report=0" in result.stderr
