import numpy as np

from ego_video_camera.camera_models import CameraModel


def test_pinhole_projection_and_behind_camera():
    camera = CameraModel(np.asarray([[100, 0, 50], [0, 100, 40], [0, 0, 1.0]]), np.zeros(5))
    pixels, valid = camera.project(np.asarray([[1, 2, 10], [0, 0, -1]], dtype=float))
    assert np.allclose(pixels[0], [60, 60])
    assert valid.tolist() == [True, False]
    assert np.isnan(pixels[1]).all()


def test_projection_uses_raw_kinect_distortion_coefficients():
    camera = CameraModel(
        np.asarray([[100, 0, 50], [0, 100, 40], [0, 0, 1.0]]),
        np.asarray([0.1, 0.0, 0.0, 0.0, 0.0]),
    )
    pixels, valid = camera.project(np.asarray([[1.0, 0.0, 2.0]]))
    assert valid.tolist() == [True]
    assert np.allclose(pixels[0], [101.25, 40.0])
