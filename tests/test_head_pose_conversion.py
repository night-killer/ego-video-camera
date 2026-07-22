import numpy as np

from ego_video_camera.egobody_io import HeadRecord, load_head_tracking, nearest_head_record
from ego_video_camera.head_pose_conversion import calibrate_camera_to_head, camera_to_head_poses


def test_fixed_camera_to_head_calibration_and_conversion():
    frame_count = 30
    cameras = np.repeat(np.eye(4)[None], frame_count, axis=0)
    cameras[:, 0, 3] = np.linspace(0, 1, frame_count)
    offset = np.eye(4)
    offset[:3, 3] = [0.0, 0.12, -0.08]
    heads = camera_to_head_poses(cameras, offset)
    result = calibrate_camera_to_head(cameras, heads)
    assert result.status == "head_pose"
    assert np.allclose(result.T_E_Q_fixed, offset)
    assert np.allclose(camera_to_head_poses(cameras, result.T_E_Q_fixed), heads)


def test_head_calibration_falls_back_when_coverage_is_low():
    cameras = np.repeat(np.eye(4)[None], 30, axis=0)
    heads = cameras.copy()
    valid = np.zeros(30, dtype=bool)
    valid[:5] = True
    result = calibrate_camera_to_head(cameras, heads, valid)
    assert result.status == "proxy"
    assert result.T_E_Q_fixed is None


def test_head_tracking_loader_sorts_small_out_of_order_runs(tmp_path):
    rows = np.asarray(
        [[timestamp, *np.eye(4).reshape(-1)] for timestamp in (30, 10, 20)],
        dtype=float,
    )
    path = tmp_path / "head.csv"
    np.savetxt(path, rows, delimiter=",")
    records = load_head_tracking(path)
    assert [record.timestamp for record in records] == [10, 20, 30]


def test_nearest_head_record_skips_invalid_duplicate():
    pose = np.eye(4)
    records = [
        HeadRecord(10_000_000, pose, False),
        HeadRecord(10_000_000, pose, True),
        HeadRecord(10_100_000, pose, True),
    ]
    match = nearest_head_record(10_000_000, records, tolerance_ms=50.0)
    assert match is records[1]
