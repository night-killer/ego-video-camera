import json
from pathlib import Path

import numpy as np

from ego_video_camera.serialization import write_json


def test_nonfinite_values_are_written_as_standard_json_null(tmp_path: Path):
    path = tmp_path / "values.json"
    write_json(path, {"values": np.asarray([1.0, np.nan, np.inf])})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"values": [1.0, None, None]}
