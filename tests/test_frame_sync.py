from pathlib import Path

import numpy as np

from ego_video_camera.egobody_io import (
    PVRecord,
    attach_pv_images,
    sample_records,
    synchronize_exact_frame_ids,
)


def test_exact_frame_id_sync_does_not_use_array_index(tmp_path: Path):
    pv = tmp_path / "PV"
    exo = tmp_path / "master"
    pv.mkdir()
    exo.mkdir()
    (pv / "10000000_frame_00042.jpg").touch()
    (pv / "20000000_frame_00107.jpg").touch()
    (exo / "frame_00107.jpg").touch()
    records = [
        PVRecord(10000000, 1, 1, np.eye(4)),
        PVRecord(20000000, 1, 1, np.eye(4)),
    ]
    records = attach_pv_images(records, pv)
    mappings = synchronize_exact_frame_ids(records, exo)
    assert mappings[0].ego_frame_id == 42
    assert mappings[0].exo_frame_id is None
    assert mappings[1].ego_frame_id == 107
    assert mappings[1].exo_frame_id == 107
    assert mappings[1].exo_timestamp is None
    assert mappings[1].sync_basis == "exact_frame_id"


def test_sampling_uses_end_exclusive_duration():
    records = []
    for index in range(201):
        records.append(
            PVRecord(
                int(index * 1_000_000),
                1,
                1,
                np.eye(4),
                index,
                Path(f"/{index}.jpg"),
            )
        )
    sampled = sample_records(records, sample_fps=8, start_sec=0, duration_sec=20)
    assert len(sampled) == 160
