from __future__ import annotations

import fnmatch
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..serialization import write_json
from .schema import MethodSpec, RunSpec, SequenceRecord


def _selected(value: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def build_run_matrix(
    methods: Iterable[MethodSpec],
    sequences: Iterable[SequenceRecord],
    output_root: str | Path,
    *,
    method_patterns: tuple[str, ...] = (),
    dataset_patterns: tuple[str, ...] = (),
    sequence_patterns: tuple[str, ...] = (),
) -> list[RunSpec]:
    root = Path(output_root)
    records = list(sequences)
    known_sequences = {sequence.key for sequence in records}
    runs: list[RunSpec] = []
    for method in methods:
        if not _selected(method.method_id, method_patterns):
            continue
        missing_subset = set(method.subset) - known_sequences
        if missing_subset:
            raise ValueError(
                f"Method {method.method_id} references unknown subset sequences: "
                + ", ".join(sorted(missing_subset))
            )
        for sequence in records:
            if method.subset and sequence.key not in method.subset:
                continue
            if not _selected(sequence.dataset_id, dataset_patterns):
                continue
            if not (
                _selected(sequence.sequence_id, sequence_patterns)
                or _selected(sequence.key, sequence_patterns)
            ):
                continue
            for seed in method.seeds:
                run_id = (
                    f"{method.method_id}/{sequence.dataset_id}/"
                    f"{sequence.sequence_id}/seed_{seed}"
                )
                runs.append(
                    RunSpec(
                        run_id=run_id,
                        method=method,
                        sequence=sequence,
                        seed=seed,
                        output_dir=root / "runs" / run_id,
                    )
                )
    return runs


def validate_inventory(config: dict[str, Any], sequences: Iterable[SequenceRecord]) -> None:
    expected = {
        str(key): int(value)
        for key, value in config["datasets"].get("expected_counts", {}).items()
    }
    actual = Counter(sequence.dataset_id for sequence in sequences)
    if expected and actual != Counter(expected):
        keys = sorted(set(expected) | set(actual))
        details = ", ".join(
            f"{key}={actual.get(key, 0)} (expected {expected.get(key, 0)})" for key in keys
        )
        raise ValueError(f"Benchmark dataset inventory mismatch: {details}")


def matrix_summary(runs: Iterable[RunSpec]) -> dict[str, Any]:
    records = list(runs)
    by_method = Counter(run.method.method_id for run in records)
    by_dataset = Counter(run.sequence.dataset_id for run in records)
    canonical = sum(run.method.canonical for run in records)
    return {
        "schema_version": 1,
        "run_count": len(records),
        "canonical_run_count": canonical,
        "ablation_run_count": len(records) - canonical,
        "by_method": dict(sorted(by_method.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "runs": [
            {
                "run_id": run.run_id,
                "method_id": run.method.method_id,
                "method_display_name": run.method.display_name,
                "canonical": run.method.canonical,
                "dataset_id": run.sequence.dataset_id,
                "sequence_id": run.sequence.sequence_id,
                "reference_grade": run.sequence.reference_grade,
                "seed": run.seed,
                "output_dir": str(run.output_dir),
                "status": "pending",
            }
            for run in records
        ],
    }


def write_run_plan(path: str | Path, runs: Iterable[RunSpec]) -> dict[str, Any]:
    payload = matrix_summary(runs)
    write_json(path, payload)
    return payload
