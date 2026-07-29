import csv
from pathlib import Path

import numpy as np

from ego_video_camera.benchmark.reference import load_reference
from ego_video_camera.benchmark.schema import FrameRecord, SequenceRecord


def _matrix(translation_x: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 3] = translation_x
    return matrix


def test_egobody_reference_matches_rows_by_filename_without_output_index(
    tmp_path: Path,
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    trajectory_path = reference_dir / "pv_trajectory.csv"
    columns = [f"t{row}{column}" for row in range(4) for column in range(4)]
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "frame_id", "filename", *columns))
        for filename, source_frame, translation in (
            ("second.jpg", 2004, 2.0),
            ("first.jpg", 2001, 1.0),
        ):
            writer.writerow(
                (0, source_frame, filename, *_matrix(translation).reshape(-1))
            )

    frames = [
        FrameRecord(0, 0, tmp_path / "first.jpg"),
        FrameRecord(1, 100_000_000, tmp_path / "second.jpg"),
    ]
    sequence = SequenceRecord(
        dataset_id="egobody",
        sequence_id="sequence",
        clip_dir=tmp_path,
        clip_json=tmp_path / "clip.json",
        input_path=tmp_path,
        duration_sec=0.2,
        target_fps=10.0,
        reference_grade="B_device_reference",
        reference_type="hololens_device_tracking",
        stratum="test",
        start_sec=0.0,
        frame_count=2,
        input_kind="frames",
    )

    trajectory = load_reference(sequence, frames)

    assert trajectory.valid.all()
    np.testing.assert_allclose(trajectory.c2w[:, 0, 3], [1.0, 2.0])
