from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    sys.path.insert(0, str(source))
    return root
