from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from ego_video_camera.gaussian import inspect_gaussian_ply, read_gaussian_arrays


def write_tiny_gaussian_ply(path: Path) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        *((f"f_rest_{index}", "f4") for index in range(9)),
        ("opacity", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    values = np.zeros(2, dtype=dtype)
    values["x"] = [1, 2]
    values["z"] = [3, 4]
    values["scale_0"] = np.log([0.5, 2.0])
    values["scale_1"] = np.log([1.0, 3.0])
    values["scale_2"] = np.log([1.5, 4.0])
    values["opacity"] = [0.0, np.log(3.0)]
    values["rot_0"] = [2.0, 1.0]
    values["f_dc_0"] = [0.1, 0.2]
    for index in range(9):
        values[f"f_rest_{index}"] = [float(index + 1), 0.0]
    PlyData([PlyElement.describe(values, "vertex")], text=False).write(str(path))


def test_gaussian_ply_mapping(tmp_path: Path) -> None:
    path = tmp_path / "tiny.ply"
    write_tiny_gaussian_ply(path)
    _, layout = inspect_gaussian_ply(path)
    assert layout.vertex_count == 2
    assert layout.sh_degree == 1
    arrays = read_gaussian_arrays(path)
    np.testing.assert_allclose(arrays.means, [[1, 0, 3], [2, 0, 4]])
    np.testing.assert_allclose(arrays.scales[0], [0.5, 1.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(arrays.quaternions[0], [1, 0, 0, 0])
    np.testing.assert_allclose(arrays.opacities, [0.5, 0.75], rtol=1e-6)
    assert arrays.harmonics.shape == (2, 4, 3)
    np.testing.assert_allclose(
        arrays.harmonics[0],
        [[0.1, 0.0, 0.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]],
    )
