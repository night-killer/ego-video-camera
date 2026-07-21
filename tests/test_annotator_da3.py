from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_trajectory
from ego_video_camera.annotator import build_annotation_context, create_annotation_app
from ego_video_camera.da3 import prediction_to_raw_trajectory
from ego_video_camera.schema import load_trajectory


def test_annotator_config_and_save(tmp_path: Path) -> None:
    ply_dir = tmp_path / "ply"
    camera_dir = tmp_path / "camera"
    spz_dir = tmp_path / "spz"
    for directory in (ply_dir, camera_dir, spz_dir):
        directory.mkdir()
    ply = ply_dir / "abcd1234.ply"; ply.write_bytes(b"ply\n")
    spz = spz_dir / "abcd1234.spz"; spz.write_bytes(b"spz")
    camera = camera_dir / "abcd1234.json"
    camera.write_text(json.dumps({
        "status":"available", "resource_id":"abcd1234",
        "coordinate_transform":"SuperSplat [x, y, z] -> local SPZ [-x, y, -z]",
        "spz_camera":{"position":[1,2,3],"target":[0,0,0],"fov":65,"radius":4},
        "supersplat_camera":{"position":[-1,2,-3],"target":[0,0,0],"fov":65},
    }))
    output = tmp_path / "keyframes.json"
    context = build_annotation_context(ply_path=ply, camera_json_path=camera, output_path=output)
    app = create_annotation_app(context)
    client = TestClient(app)
    config = client.get("/api/config")
    assert config.status_code == 200
    assert config.json()["asset_kind"] == "spz"
    K = [[300,0,448],[0,300,252],[0,0,1]]
    payload = {
        "schema_version":"camera_trajectory.v1", "trajectory_type":"keyframes",
        "coordinate_system":"ignored", "camera_axes":"opencv_rdf_x_right_y_down_z_forward",
        "scene":config.json()["scene"],
        "video":{"width":896,"height":504,"fps":15,"fov_y_degrees":65},
        "frames":[
            {"frame_index":0,"timestamp_seconds":0,"camera_to_world":np.eye(4).tolist(),"K":K},
            {"frame_index":15,"timestamp_seconds":1,"camera_to_world":np.eye(4).tolist(),"K":K},
        ], "source":{}
    }
    response = client.post("/api/save", json=payload)
    assert response.status_code == 200, response.text
    assert load_trajectory(output).coordinate_system == "supersplat_source_ply_world"


def test_da3_prediction_conversion_scales_intrinsics(scene_spec) -> None:
    gt = make_trajectory(scene_spec, count=3)
    extrinsics = np.repeat(np.eye(4)[None], 3, axis=0)
    extrinsics[:, 0, 3] = [0, -1, -2]
    intrinsics = np.repeat(np.asarray([[[200,0,250],[0,210,140],[0,0,1]]]), 3, axis=0)
    raw = prediction_to_raw_trajectory(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        processed_hw=(280, 500),
        gt=gt,
        source={},
    )
    poses, Ks = raw.matrices()
    np.testing.assert_allclose(poses[:, 0, 3], [0, 1, 2])
    assert Ks[0, 0, 0] == pytest.approx(200 * 896 / 500)
    assert Ks[0, 1, 1] == pytest.approx(210 * 504 / 280)
