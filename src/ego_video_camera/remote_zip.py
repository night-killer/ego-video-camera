from __future__ import annotations

import fcntl
import hashlib
import os
import struct
import tempfile
import zipfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .download import (
    _copy_exact,
    _download_segment,
    _reset_download_cancellation,
    _segment_ranges,
    _terminate_active_curls,
    inspect_remote,
    validate_netrc,
)
from .serialization import read_json, write_json


EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"


@dataclass(frozen=True)
class ZipDirectoryInfo:
    entry_count: int
    offset: int
    size: int


def parse_zip_directory(tail: bytes, tail_start: int) -> ZipDirectoryInfo:
    eocd_position = tail.rfind(EOCD_SIGNATURE)
    if eocd_position < 0 or len(tail) - eocd_position < 22:
        raise zipfile.BadZipFile("ZIP end-of-central-directory record was not found")
    _, _, _, _, entries, directory_size, directory_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_position
    )
    if entries != 0xFFFF and directory_size != 0xFFFFFFFF and directory_offset != 0xFFFFFFFF:
        return ZipDirectoryInfo(int(entries), int(directory_offset), int(directory_size))
    locator_position = tail.rfind(ZIP64_LOCATOR_SIGNATURE, 0, eocd_position)
    if locator_position < 0:
        raise zipfile.BadZipFile("ZIP64 locator was not found")
    _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", tail, locator_position)
    relative_offset = int(zip64_offset) - tail_start
    if relative_offset < 0 or len(tail) - relative_offset < 56:
        raise zipfile.BadZipFile("ZIP64 end record is outside the downloaded tail")
    values = struct.unpack_from("<4sQ2H2L4Q", tail, relative_offset)
    if values[0] != ZIP64_EOCD_SIGNATURE:
        raise zipfile.BadZipFile("Invalid ZIP64 end record signature")
    return ZipDirectoryInfo(int(values[7]), int(values[9]), int(values[8]))


def _merge_intervals(
    intervals: Iterable[tuple[int, int]], gap_bytes: int
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, inclusive_end in sorted(intervals):
        if not merged or start > merged[-1][1] + gap_bytes + 1:
            merged.append((start, inclusive_end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, inclusive_end))
    return merged


class RemoteZipCache:
    """Sparse local ZIP cache backed only by authenticated official byte ranges."""

    def __init__(
        self,
        data_root: str | Path,
        netrc_file: str | Path,
        connections: int = 8,
        cache_root: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.netrc = validate_netrc(netrc_file)
        self.connections = max(1, int(connections))
        if cache_root is None:
            namespace = hashlib.sha256(str(self.data_root).encode("utf-8")).hexdigest()[:16]
            cache_root = (
                Path(tempfile.gettempdir())
                / "ego_video_camera_remote_zip_cache"
                / namespace
            )
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def _paths(self, name: str) -> tuple[Path, Path, Path]:
        return (
            self.root / name,
            self.root / f"{name}.index.json",
            self.root / "_segments" / name,
        )

    @contextmanager
    def _exclusive(self, name: str):
        """Serialize sparse writes from separate resume/select processes."""

        lock_path = self.root / f".{name}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _materialize_ranges(
        self,
        name: str,
        remote,
        cache_path: Path,
        ranges: Iterable[tuple[int, int]],
    ) -> None:
        _, _, segment_root = self._paths(name)
        pieces = []
        for start, inclusive_end in ranges:
            requested_size = inclusive_end - start + 1
            adaptive_size = min(
                32 * 1024**2,
                max(
                    1024**2,
                    (requested_size + self.connections - 1) // self.connections,
                ),
            )
            pieces.extend(
                _segment_ranges(
                    start,
                    inclusive_end + 1,
                    segment_size=adaptive_size,
                )
            )
        if not pieces:
            return
        _reset_download_cancellation()
        segment_root.mkdir(parents=True, exist_ok=True)
        planned_names = {
            f"{start:012d}_{inclusive_end:012d}.part"
            for start, inclusive_end in pieces
        }
        for stale in segment_root.glob("*.part"):
            if stale.name not in planned_names:
                stale.unlink()
        for start, inclusive_end in pieces:
            segment_path = segment_root / f"{start:012d}_{inclusive_end:012d}.part"
            if segment_path.exists() and not (
                0 <= segment_path.stat().st_size <= inclusive_end - start + 1
            ):
                segment_path.unlink()
        jobs = []
        executor = ThreadPoolExecutor(max_workers=self.connections)
        try:
            for start, inclusive_end in pieces:
                segment_path = segment_root / f"{start:012d}_{inclusive_end:012d}.part"
                jobs.append(
                    executor.submit(
                        _download_segment,
                        remote,
                        self.netrc,
                        segment_path,
                        start,
                        inclusive_end,
                    )
                )
            completed = [future.result() for future in as_completed(jobs)]
        except BaseException:
            _terminate_active_curls()
            for future in jobs:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        with cache_path.open("r+b") as target:
            for start, inclusive_end, segment_path in sorted(completed):
                target.seek(start)
                with segment_path.open("rb") as source:
                    _copy_exact(source, target, inclusive_end - start + 1)
                segment_path.unlink(missing_ok=True)
            target.flush()
            os.fsync(target.fileno())

    def ensure_index(self, name: str) -> Path:
        with self._exclusive(name):
            return self._ensure_index_unlocked(name)

    def _ensure_index_unlocked(self, name: str) -> Path:
        remote = inspect_remote(name, self.netrc)
        cache_path, metadata_path, _ = self._paths(name)
        if cache_path.is_file() and metadata_path.is_file():
            metadata = read_json(metadata_path)
            if (
                metadata.get("content_length") == remote.content_length
                and metadata.get("etag") == remote.etag
                and metadata.get("index_status") == "central_directory_ready"
                and cache_path.stat().st_size == remote.content_length
            ):
                with zipfile.ZipFile(cache_path) as archive:
                    archive.infolist()
                return cache_path
        cache_path.unlink(missing_ok=True)
        cache_path.touch()
        os.truncate(cache_path, remote.content_length)
        tail_size = min(remote.content_length, 128 * 1024)
        tail_start = remote.content_length - tail_size
        self._materialize_ranges(
            name,
            remote,
            cache_path,
            [(tail_start, remote.content_length - 1)],
        )
        with cache_path.open("rb") as handle:
            handle.seek(tail_start)
            tail = handle.read(tail_size)
        directory = parse_zip_directory(tail, tail_start)
        self._materialize_ranges(
            name,
            remote,
            cache_path,
            [(directory.offset, directory.offset + directory.size - 1)],
        )
        with zipfile.ZipFile(cache_path) as archive:
            actual_count = len(archive.infolist())
        if actual_count != directory.entry_count:
            raise zipfile.BadZipFile(
                f"Central-directory entry count mismatch: {actual_count} != {directory.entry_count}"
            )
        write_json(
            metadata_path,
            {
                "source": remote.url,
                "content_length": remote.content_length,
                "etag": remote.etag,
                "accept_ranges": remote.accept_ranges,
                "index_status": "central_directory_ready",
                "entry_count": actual_count,
                "central_directory_offset": directory.offset,
                "central_directory_size": directory.size,
                "cache_kind": "sparse_index_not_full_archive",
                "storage_kind": "local_sparse_scratch",
                "materialized_members": [],
                "credentials_recorded": False,
            },
        )
        return cache_path

    def ensure_members(
        self,
        name: str,
        members: Iterable[str],
        merge_gap_bytes: int = 4 * 1024**2,
    ) -> Path:
        cache_path = self.ensure_index(name)
        with self._exclusive(name):
            return self._ensure_members_unlocked(
                name,
                members,
                merge_gap_bytes=merge_gap_bytes,
                cache_path=cache_path,
            )

    def _ensure_members_unlocked(
        self,
        name: str,
        members: Iterable[str],
        merge_gap_bytes: int,
        cache_path: Path,
    ) -> Path:
        _, metadata_path, _ = self._paths(name)
        metadata = read_json(metadata_path)
        already = set(metadata.get("materialized_members", []))
        requested = sorted(set(members) - already)
        if not requested:
            return cache_path
        remote = inspect_remote(name, self.netrc)
        with zipfile.ZipFile(cache_path) as archive:
            by_name = {entry.filename: entry for entry in archive.infolist()}
        missing = [member for member in requested if member not in by_name]
        if missing:
            raise KeyError(f"ZIP members not found in {name}: {missing[:3]}")
        intervals = []
        for member in requested:
            entry = by_name[member]
            # Fetch a conservative 4 KiB local-header allowance followed by
            # the complete compressed payload. EgoBody headers are much smaller.
            start = int(entry.header_offset)
            inclusive_end = min(
                remote.content_length - 1,
                start + 4096 + int(entry.compress_size) - 1,
            )
            intervals.append((start, inclusive_end))
        self._materialize_ranges(
            name,
            remote,
            cache_path,
            _merge_intervals(intervals, merge_gap_bytes),
        )
        with zipfile.ZipFile(cache_path) as archive:
            for member in requested:
                with archive.open(member) as source:
                    while source.read(8 * 1024 * 1024):
                        pass
        metadata["materialized_members"] = sorted(already | set(requested))
        write_json(metadata_path, metadata)
        return cache_path
