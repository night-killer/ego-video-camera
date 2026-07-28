from __future__ import annotations

import csv
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from .common import WorkerContext


_TRACKING_LABELS = {
    -1: "system_not_ready",
    0: "no_images_yet",
    1: "not_initialized",
    2: "tracking",
    3: "recently_lost",
    4: "lost",
    5: "tracking_klt",
}


def _extract_vocabulary(archive: Path, output_dir: Path) -> Path:
    destination = output_dir / "ORBvoc.txt"
    if destination.is_file():
        return destination
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = [member for member in handle.getmembers() if Path(member.name).name == "ORBvoc.txt"]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(f"Expected one ORBvoc.txt file in {archive}")
        source = handle.extractfile(members[0])
        if source is None:
            raise ValueError(f"Cannot read ORBvoc.txt from {archive}")
        temporary = output_dir / ".ORBvoc.txt.tmp"
        with temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
    return destination


def _write_inputs(context: WorkerContext) -> tuple[Path, Path]:
    import cv2

    work = context.output_dir / "work" / "orb_slam3"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "frames.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in context.frames:
            writer.writerow([row["frame_id"], float(row["timestamp_ns"]) * 1e-9, row["image_path"]])
    first_image = cv2.imread(context.frames[0]["image_path"], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise FileNotFoundError(context.frames[0]["image_path"])
    height, width = first_image.shape[:2]
    intrinsic = np.asarray(context.frames[0]["intrinsic"], dtype=np.float64)
    settings = work / "camera.yaml"
    settings.write_text(
        "\n".join(
            [
                "%YAML:1.0",
                'File.version: "1.0"',
                'Camera.type: "PinHole"',
                f"Camera1.fx: {intrinsic[0, 0]:.12g}",
                f"Camera1.fy: {intrinsic[1, 1]:.12g}",
                f"Camera1.cx: {intrinsic[0, 2]:.12g}",
                f"Camera1.cy: {intrinsic[1, 2]:.12g}",
                "Camera1.k1: 0.0",
                "Camera1.k2: 0.0",
                "Camera1.p1: 0.0",
                "Camera1.p2: 0.0",
                f"Camera.fps: {float(context.manifest['target_fps']):.12g}",
                "Camera.RGB: 0",
                f"Camera.width: {width}",
                f"Camera.height: {height}",
                "ORBextractor.nFeatures: 1500",
                "ORBextractor.scaleFactor: 1.2",
                "ORBextractor.nLevels: 8",
                "ORBextractor.iniThFAST: 20",
                "ORBextractor.minThFAST: 7",
                "Viewer.KeyFrameSize: 0.05",
                "Viewer.KeyFrameLineWidth: 1.0",
                "Viewer.GraphLineWidth: 0.9",
                "Viewer.PointSize: 2.0",
                "Viewer.CameraSize: 0.08",
                "Viewer.CameraLineWidth: 3.0",
                "Viewer.ViewpointX: 0.0",
                "Viewer.ViewpointY: -0.7",
                "Viewer.ViewpointZ: -1.8",
                "Viewer.ViewpointF: 500.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path, settings


def run(context: WorkerContext):
    executable = Path(str(context.parameters["executable"]))
    if not executable.is_absolute():
        executable = context.repo.parent.parent / executable
    if not executable.is_file():
        raise FileNotFoundError(
            f"ORB-SLAM3 benchmark runner is not built: {executable}. "
            "Run scripts/build_orb_slam3_benchmark.sh in its prepared environment."
        )
    manifest_path, settings = _write_inputs(context)
    vocabulary = _extract_vocabulary(
        context.checkpoint(0), context.output_dir / "work" / "orb_slam3" / "vocabulary"
    )
    native_output = context.output_dir / "work" / "orb_slam3" / "poses.tsv"
    context.mark_model_ready()
    subprocess.run(
        [str(executable), str(vocabulary), str(settings), str(manifest_path), str(native_output)],
        check=True,
        cwd=context.repo,
    )
    rows = list(csv.DictReader(native_output.open(encoding="utf-8"), delimiter="\t"))
    if len(rows) != len(context.frames):
        raise ValueError(f"ORB-SLAM3 returned {len(rows)}/{len(context.frames)} rows")
    poses = np.full((len(rows), 4, 4), np.nan, dtype=np.float64)
    states = np.asarray([int(row["tracking_state"]) for row in rows])
    valid = np.asarray([row["valid"] == "1" for row in rows])
    for index, row in enumerate(rows):
        if valid[index]:
            poses[index] = np.asarray(
                [float(row[f"m{r}{c}"]) for r in range(4) for c in range(4)]
            ).reshape(4, 4)
    confidence = np.asarray([float(row["tracked_points"]) for row in rows])
    if np.nanmax(confidence, initial=0.0) > 0:
        confidence /= np.nanmax(confidence)
    confidence[~valid] = np.nan
    context.mark_first_prediction()
    return context.expected_trajectory(
        poses,
        valid=valid,
        confidence=confidence,
        tracking_state=np.asarray([_TRACKING_LABELS.get(state, f"state_{state}") for state in states]),
        metadata={"viewer_enabled": False, "native_output": str(native_output)},
    )
