"""Small filesystem helpers shared by the command-line tools."""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any


class PipelineInputError(ValueError):
    """Raised when an input artifact violates a pipeline contract."""


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON using an adjacent temporary file and an atomic rename."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_object(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineInputError(f"JSON file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineInputError(
            f"invalid JSON in {source}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise PipelineInputError(f"expected a JSON object in {source}")
    return payload


def require_input_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PipelineInputError(f"{label} does not exist: {resolved}")
    if resolved.stat().st_size <= 0:
        raise PipelineInputError(f"{label} is empty: {resolved}")
    return resolved


def finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineInputError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise PipelineInputError(f"{label} must be finite")
    return number
