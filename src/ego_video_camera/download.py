from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
import zipfile
import zlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import urlparse

from .serialization import read_json, write_json


OFFICIAL_BASE_URL = "https://egobody.ethz.ch/data/dataset/"
REQUIRED_FILES = (
    "data_info_release.csv",
    "data_splits.csv",
    "calibrations.zip",
    "kinect_cam_params.zip",
    "egocentric_gaze.zip",
    "egocentric_color.zip",
    "kinect_color.zip",
)

CANONICAL_DATA_ROOTS = (
    "calibrations",
    "kinect_cam_params",
    "egocentric_color",
    "kinect_color",
    "egocentric_gaze",
)

_ACTIVE_CURLS: set[subprocess.Popen] = set()
_ACTIVE_CURLS_LOCK = threading.Lock()
_CANCEL_DOWNLOADS = threading.Event()


@contextmanager
def _exclusive_file_lock(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class EgoBodyDownloadError(RuntimeError):
    """A sanitized official-download failure that never embeds curl arguments."""

    def __init__(self, name: str, http_status: int | None, reason: str) -> None:
        self.name = name
        self.http_status = http_status
        self.reason = reason
        message = f"Official EgoBody download failed for {name}: {reason}"
        if http_status is not None:
            message += f" (HTTP {http_status})"
        super().__init__(message)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    content_length: int
    etag: str | None
    accept_ranges: str | None


def _etag_arguments(remote: RemoteFile) -> list[str]:
    return ["--header", f"If-Match: {remote.etag}"] if remote.etag else []


def validate_netrc(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("The supplied netrc path must not be a symbolic link")
    path = candidate.resolve()
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("The supplied netrc path is not a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("EgoBody netrc must not be accessible by group or other users")
    if not stat.S_IMODE(info.st_mode) & stat.S_IRUSR:
        raise PermissionError("EgoBody netrc must be readable by its owner")
    return path


def official_url(name: str) -> str:
    if name not in REQUIRED_FILES:
        raise ValueError(f"File is not in the approved EgoBody manifest: {name}")
    url = OFFICIAL_BASE_URL + name
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "egobody.ethz.ch":
        raise ValueError("Only official EgoBody HTTPS URLs are allowed")
    return url


def _curl_base(netrc: Path) -> list[str]:
    return [
        "curl",
        "--netrc-file",
        str(netrc),
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
    ]


def _run_curl(
    name: str,
    netrc: Path,
    arguments: list[str],
    require_status: set[int] | None = None,
) -> str:
    marker = "__EGO_HTTP_STATUS__:"
    process = subprocess.Popen(
        _curl_base(netrc) + arguments + ["--write-out", f"\n{marker}%{{http_code}}\n"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with _ACTIVE_CURLS_LOCK:
        _ACTIVE_CURLS.add(process)
    try:
        stdout, _ = process.communicate()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        with _ACTIVE_CURLS_LOCK:
            _ACTIVE_CURLS.discard(process)
    status: int | None = None
    output_lines = []
    for line in stdout.replace("\r", "").splitlines():
        if line.startswith(marker):
            try:
                status = int(line[len(marker) :])
            except ValueError:
                status = None
        else:
            output_lines.append(line)
    if process.returncode != 0:
        reason = "authentication_expired_or_rejected" if status in {401, 403} else "network_or_server_error"
        raise EgoBodyDownloadError(name, status, reason)
    if status is not None and status >= 400:
        reason = "authentication_expired_or_rejected" if status in {401, 403} else "official_server_error"
        raise EgoBodyDownloadError(name, status, reason)
    if require_status is not None and status not in require_status:
        raise EgoBodyDownloadError(name, status, "unexpected_http_status")
    return "\n".join(output_lines)


def _terminate_active_curls() -> None:
    _CANCEL_DOWNLOADS.set()
    with _ACTIVE_CURLS_LOCK:
        processes = list(_ACTIVE_CURLS)
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _reset_download_cancellation() -> None:
    _CANCEL_DOWNLOADS.clear()


def inspect_remote(name: str, netrc_file: str | Path) -> RemoteFile:
    netrc = validate_netrc(netrc_file)
    url = official_url(name)
    header_text = _run_curl(name, netrc, ["--head", url])
    values: dict[str, str] = {}
    for raw_line in header_text.splitlines():
        if ":" in raw_line:
            key, value = raw_line.split(":", 1)
            values[key.strip().lower()] = value.strip()
    if "content-length" not in values:
        raise EgoBodyDownloadError(name, None, "missing_content_length")
    length = int(values["content-length"])
    return RemoteFile(name, url, length, values.get("etag"), values.get("accept-ranges"))


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_identity(remote: RemoteFile) -> dict:
    return {
        "name": remote.name,
        "url": remote.url,
        "content_length": remote.content_length,
        "etag": remote.etag,
        "credentials_recorded": False,
    }


def _identity_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".remote.json")


def _bind_or_validate_remote_identity(remote: RemoteFile, payload_path: Path) -> Path:
    """Bind resumable bytes to the official object identity without secrets."""

    sidecar = _identity_sidecar(payload_path)
    expected = _remote_identity(remote)
    if sidecar.is_file():
        try:
            recorded = read_json(sidecar)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise EgoBodyDownloadError(
                remote.name, None, "invalid_remote_identity_sidecar"
            ) from error
        if any(recorded.get(key) != expected[key] for key in expected):
            raise EgoBodyDownloadError(remote.name, None, "remote_identity_changed")
    else:
        write_json(
            sidecar,
            {
                **expected,
                "adopted_existing_payload": bool(
                    payload_path.is_file() and payload_path.stat().st_size > 0
                ),
            },
        )
    return sidecar


def _verify_remote_unchanged(remote: RemoteFile, netrc: Path) -> None:
    current = inspect_remote(remote.name, netrc)
    expected = _remote_identity(remote)
    actual = _remote_identity(current)
    if any(actual[key] != expected[key] for key in expected):
        raise EgoBodyDownloadError(remote.name, None, "remote_identity_changed")


def validate_downloaded_payload(path: str | Path) -> dict:
    """Validate a completed object before its atomic promotion from ``.part``."""

    path = Path(path)
    member_count = None
    crc_all_members = None
    if path.suffix == ".zip" or path.name.endswith(".zip.part"):
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            for entry in entries:
                _canonical_member_name(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(
                        f"Refusing ZIP archive containing a symbolic link: {entry.filename}"
                    )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(f"CRC mismatch in ZIP member: {bad_member}")
            member_count = len(entries)
            crc_all_members = True
    return {
        "sha256": sha256_file(path),
        "zip_member_count": member_count,
        "zip_crc_all_members": crc_all_members,
    }


def download_file(remote: RemoteFile, archive_root: str | Path, netrc_file: str | Path) -> dict:
    netrc = validate_netrc(netrc_file)
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    final_path = archive_root / remote.name
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.is_file() and final_path.stat().st_size == remote.content_length:
        _bind_or_validate_remote_identity(remote, final_path)
        status = "existing"
        validation = validate_downloaded_payload(final_path)
    else:
        partial_path.touch(exist_ok=True)
        partial_identity = _bind_or_validate_remote_identity(remote, partial_path)
        if partial_path.exists() and partial_path.stat().st_size > remote.content_length:
            raise ValueError(f"Partial file is larger than the official object: {partial_path}")
        if not partial_path.is_file() or partial_path.stat().st_size != remote.content_length:
            for attempt in range(9):
                try:
                    _run_curl(
                        remote.name,
                        netrc,
                        [
                            "--continue-at",
                            "-",
                            *_etag_arguments(remote),
                            "--output",
                            str(partial_path),
                            remote.url,
                        ],
                    )
                except EgoBodyDownloadError as error:
                    if error.reason == "authentication_expired_or_rejected" or attempt == 8:
                        raise
                    time.sleep(5)
                    continue
                if partial_path.stat().st_size == remote.content_length:
                    break
        if partial_path.stat().st_size != remote.content_length:
            raise IOError(
                f"Size mismatch for {remote.name}: {partial_path.stat().st_size} != {remote.content_length}"
            )
        _verify_remote_unchanged(remote, netrc)
        validation = validate_downloaded_payload(partial_path)
        partial_path.replace(final_path)
        write_json(_identity_sidecar(final_path), _remote_identity(remote))
        partial_identity.unlink(missing_ok=True)
        status = "downloaded"
    return {
        "name": remote.name,
        "url": remote.url,
        "content_length": remote.content_length,
        "etag": remote.etag,
        "accept_ranges": remote.accept_ranges,
        "local_path": str(final_path),
        **validation,
        "status": status,
    }


def _segment_ranges(
    start: int,
    end: int,
    segment_size: int = 32 * 1024**2,
) -> list[tuple[int, int]]:
    remaining = end - start
    if remaining <= 0:
        return []
    return [
        (segment_start, min(segment_start + segment_size, end) - 1)
        for segment_start in range(start, end, segment_size)
    ]


def migrate_legacy_segment_file(
    segment_path: str | Path,
    start: int,
    inclusive_end: int,
) -> bool:
    """Convert one old absolute-offset range file to compact storage in place."""

    segment_path = Path(segment_path)
    expected_size = inclusive_end - start + 1
    if segment_path.exists() and segment_path.stat().st_size > expected_size:
        legacy_size = segment_path.stat().st_size
        if start <= legacy_size <= inclusive_end + 1:
            # Older builds represented a range by a sparse file whose logical
            # size began at the absolute remote offset. Some shared filesystems
            # allocate those holes physically. Convert its downloaded suffix
            # to the compact relative representation without discarding bytes.
            migrated = segment_path.with_suffix(segment_path.suffix + ".relative")
            with segment_path.open("rb") as source, migrated.open("wb") as target:
                source.seek(start)
                _copy_exact(source, target, legacy_size - start)
                target.flush()
                os.fsync(target.fileno())
            migrated.replace(segment_path)
            return True
        else:
            segment_path.unlink()
    return False


def _download_segment(
    remote: RemoteFile,
    netrc: Path,
    segment_path: Path,
    start: int,
    inclusive_end: int,
) -> tuple[int, int, Path]:
    """Download one bounded HTTP range into a relative segment file.

    Full-object downloads use ``curl -C -`` in :func:`download_file`.
    Bounded ranges must not combine curl's automatic continuation Range with
    another bounded Range header because some servers return both requested
    ranges. Completed fixed ranges are therefore the parallel resume units;
    an interrupted, confirmed HTTP 206 response can still be retained and its
    missing suffix requested on the next attempt.
    """
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = inclusive_end - start + 1
    migrate_legacy_segment_file(segment_path, start, inclusive_end)
    segment_path.touch(exist_ok=True)
    response_path = segment_path.with_suffix(segment_path.suffix + ".transfer")
    header_path = response_path.with_suffix(response_path.suffix + ".headers")

    def response_headers_match(expected_start: int, maximum_size: int) -> bool:
        """Validate a response left behind by an interrupted curl process.

        curl writes response headers before the body.  Keeping the header file
        beside ``.transfer`` lets a later process prove that those bytes came
        from the expected HTTP 206 range and unchanged official object before
        appending them to the durable segment.
        """

        if not header_path.is_file():
            return False
        try:
            raw_headers = header_path.read_text(encoding="iso-8859-1")
        except OSError:
            return False
        blocks = re.split(r"\r?\n\r?\n", raw_headers)
        for block in reversed(blocks):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines or not lines[0].upper().startswith("HTTP/"):
                continue
            status_parts = lines[0].split()
            if len(status_parts) < 2 or status_parts[1] != "206":
                return False
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
            match = re.fullmatch(
                r"bytes\s+(\d+)-(\d+)/(\d+)",
                headers.get("content-range", ""),
                flags=re.IGNORECASE,
            )
            if match is None:
                return False
            range_start, range_end, total = map(int, match.groups())
            if (
                range_start != expected_start
                or range_end != inclusive_end
                or total != remote.content_length
            ):
                return False
            try:
                content_length = int(headers.get("content-length", ""))
            except ValueError:
                return False
            if content_length != maximum_size:
                return False
            if remote.etag is not None and headers.get("etag") != remote.etag:
                return False
            return True
        return False

    def discard_response() -> None:
        response_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)

    def retain_response(
        maximum_size: int,
        expected_start: int,
        *,
        status_already_validated: bool = False,
    ) -> int:
        if not response_path.is_file():
            header_path.unlink(missing_ok=True)
            return 0
        received = response_path.stat().st_size
        if (
            received <= 0
            or received > maximum_size
            or not (
                status_already_validated
                or response_headers_match(expected_start, maximum_size)
            )
        ):
            discard_response()
            return 0
        with segment_path.open("ab") as target, response_path.open("rb") as source:
            _copy_exact(source, target, received)
            target.flush()
            os.fsync(target.fileno())
        discard_response()
        return received

    for attempt in range(9):
        if _CANCEL_DOWNLOADS.is_set():
            raise InterruptedError("Parallel EgoBody download was cancelled")
        current = segment_path.stat().st_size
        if current < 0 or current > expected_size:
            raise IOError(
                f"Invalid resumable segment size for {remote.name}: "
                f"{current} not in [0, {expected_size}]"
            )
        if current == expected_size:
            break
        remote_start = start + current
        remaining = expected_size - current
        # Recover a range body left by a killed prior process only when its
        # persisted headers prove it was the exact expected HTTP 206 response.
        retain_response(remaining, remote_start)
        current = segment_path.stat().st_size
        if current == expected_size:
            break
        remote_start = start + current
        remaining = expected_size - current
        discard_response()
        try:
            _run_curl(
                remote.name,
                netrc,
                [
                    "--range",
                    f"{remote_start}-{inclusive_end}",
                    *_etag_arguments(remote),
                    "--dump-header",
                    str(header_path),
                    "--output",
                    str(response_path),
                    remote.url,
                ],
                require_status={206},
            )
        except EgoBodyDownloadError as error:
            if error.http_status == 206:
                retain_response(
                    remaining,
                    remote_start,
                    status_already_validated=True,
                )
            else:
                discard_response()
            if segment_path.stat().st_size == expected_size:
                break
            if error.reason == "authentication_expired_or_rejected" or attempt == 8:
                raise
            time.sleep(5)
            continue
        received = response_path.stat().st_size if response_path.is_file() else 0
        if received != remaining:
            retain_response(
                remaining,
                remote_start,
                status_already_validated=True,
            )
            if attempt == 8:
                raise IOError(
                    f"Incomplete HTTP range for {remote.name}: {received} != {remaining}"
                )
            time.sleep(5)
            continue
        retain_response(
            remaining,
            remote_start,
            status_already_validated=True,
        )
    if segment_path.stat().st_size != expected_size:
        raise IOError(
            f"Incomplete segment for {remote.name}: {segment_path.stat().st_size} != {expected_size}"
        )
    discard_response()
    return start, inclusive_end, segment_path


def _copy_exact(source, target, length: int, chunk_size: int = 16 * 1024 * 1024) -> None:
    remaining = length
    while remaining:
        chunk = source.read(min(chunk_size, remaining))
        if not chunk:
            raise IOError("Unexpected EOF while assembling resumable download segments")
        target.write(chunk)
        remaining -= len(chunk)


def download_file_parallel(
    remote: RemoteFile,
    archive_root: str | Path,
    netrc_file: str | Path,
    connections: int,
) -> dict:
    if connections < 2:
        return download_file(remote, archive_root, netrc_file)
    if (remote.accept_ranges or "").lower() != "bytes":
        raise EgoBodyDownloadError(remote.name, None, "official_server_does_not_support_ranges")
    netrc = validate_netrc(netrc_file)
    _reset_download_cancellation()
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    final_path = archive_root / remote.name
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.is_file() and final_path.stat().st_size == remote.content_length:
        _bind_or_validate_remote_identity(remote, final_path)
        status = "existing"
        validation = validate_downloaded_payload(final_path)
    else:
        if not partial_path.exists():
            partial_path.touch()
        partial_identity = _bind_or_validate_remote_identity(remote, partial_path)
        prefix_size = partial_path.stat().st_size
        if prefix_size > remote.content_length:
            raise ValueError(f"Partial file is larger than the official object: {partial_path}")
        ranges = _segment_ranges(prefix_size, remote.content_length)
        segment_root = archive_root / "_segments" / remote.name
        planned_names = {
            f"{start:012d}_{inclusive_end:012d}.part"
            for start, inclusive_end in ranges
        }
        if segment_root.is_dir():
            for stale in segment_root.glob("*.part"):
                if stale.name not in planned_names:
                    stale.unlink()
        for start, inclusive_end in ranges:
            segment_path = segment_root / f"{start:012d}_{inclusive_end:012d}.part"
            migrate_legacy_segment_file(segment_path, start, inclusive_end)
        jobs = []
        executor = ThreadPoolExecutor(max_workers=connections)
        try:
            for start, inclusive_end in ranges:
                segment_path = segment_root / f"{start:012d}_{inclusive_end:012d}.part"
                jobs.append(
                    executor.submit(
                        _download_segment,
                        remote,
                        netrc,
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
        completed.sort(key=lambda item: item[0])
        expected_start = prefix_size
        with partial_path.open("r+b") as target:
            target.seek(prefix_size)
            for start, inclusive_end, segment_path in completed:
                if start != expected_start:
                    raise IOError(
                        f"Non-contiguous segment assembly for {remote.name}: {start} != {expected_start}"
                    )
                with segment_path.open("rb") as source:
                    _copy_exact(source, target, inclusive_end - start + 1)
                expected_start = inclusive_end + 1
            target.flush()
            os.fsync(target.fileno())
        if partial_path.stat().st_size != remote.content_length:
            raise IOError(
                f"Size mismatch for {remote.name}: {partial_path.stat().st_size} != {remote.content_length}"
            )
        _verify_remote_unchanged(remote, netrc)
        validation = validate_downloaded_payload(partial_path)
        partial_path.replace(final_path)
        write_json(_identity_sidecar(final_path), _remote_identity(remote))
        partial_identity.unlink(missing_ok=True)
        for _, _, segment_path in completed:
            segment_path.unlink(missing_ok=True)
        if segment_root.is_dir() and not any(segment_root.iterdir()):
            segment_root.rmdir()
        status = f"downloaded_parallel_{connections}"
    return {
        "name": remote.name,
        "url": remote.url,
        "content_length": remote.content_length,
        "etag": remote.etag,
        "accept_ranges": remote.accept_ranges,
        "local_path": str(final_path),
        **validation,
        "status": status,
    }


def _record_download_result(data_root: Path, result: dict) -> None:
    manifest_path = data_root / "download_manifest.json"
    with _exclusive_file_lock(data_root / ".download_manifest.lock"):
        existing_files: list[dict] = []
        if manifest_path.is_file():
            try:
                existing = read_json(manifest_path)
                if existing.get("source") == OFFICIAL_BASE_URL:
                    existing_files = list(existing.get("files", []))
            except (OSError, ValueError, json.JSONDecodeError):
                existing_files = []
        by_name = {
            item["name"]: item
            for item in [*existing_files, result]
            if isinstance(item, dict) and item.get("name") in REQUIRED_FILES
        }
        order = {name: index for index, name in enumerate(REQUIRED_FILES)}
        write_json(
            manifest_path,
            {
                "source": OFFICIAL_BASE_URL,
                "credentials_recorded": False,
                "files": sorted(
                    by_name.values(), key=lambda item: order[item["name"]]
                ),
            },
        )
        blocked_path = data_root / "download_blocked.json"
        if blocked_path.is_file():
            try:
                blocked = read_json(blocked_path)
            except (OSError, ValueError, json.JSONDecodeError):
                blocked = {}
            if blocked.get("file") == result.get("name"):
                blocked_path.unlink(missing_ok=True)


def _record_download_blocked(data_root: Path, error: EgoBodyDownloadError) -> None:
    authentication = error.reason == "authentication_expired_or_rejected"
    with _exclusive_file_lock(data_root / ".download_manifest.lock"):
        write_json(
            data_root / "download_blocked.json",
            {
                "status": "blocked",
                "reason": error.reason,
                "http_status": error.http_status,
                "file": error.name,
                "source": OFFICIAL_BASE_URL,
                "credentials_recorded": False,
                "action": (
                    "Renew EgoBody official authentication and resume the same command"
                    if authentication
                    else "Review the affected partial and official object identity before resuming"
                ),
            },
        )


def download_required(
    data_root: str | Path,
    netrc_file: str | Path,
    names: Iterable[str] = REQUIRED_FILES,
    connections: int = 1,
) -> list[dict]:
    data_root = Path(data_root)
    archive_root = data_root / "_archives"
    results = []
    for name in names:
        try:
            with _exclusive_file_lock(archive_root / f".{name}.download.lock"):
                remote = inspect_remote(name, netrc_file)
                result = (
                    download_file_parallel(
                        remote, archive_root, netrc_file, connections
                    )
                    if connections > 1
                    else download_file(remote, archive_root, netrc_file)
                )
                results.append(result)
        except EgoBodyDownloadError as error:
            _record_download_blocked(data_root, error)
            raise EgoBodyDownloadError(error.name, error.http_status, error.reason) from None
        _record_download_result(data_root, result)
    return results


def _safe_destination(root: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    destination = (root / Path(*member.parts)).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"ZIP member escapes extraction root: {member_name}")
    return destination


def _canonical_member_name(member_name: str) -> str:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    for marker in CANONICAL_DATA_ROOTS:
        if marker in member.parts:
            return str(PurePosixPath(*member.parts[member.parts.index(marker) :]))
    return str(member)


def _crc32_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def extract_members(archive_path: str | Path, data_root: str | Path, members: Iterable[str]) -> list[Path]:
    data_root = Path(data_root)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        by_name = {entry.filename: entry for entry in archive.infolist()}
        for name in members:
            entry = by_name.get(name)
            if entry is None or entry.is_dir():
                continue
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Refusing to extract ZIP symlink: {name}")
            destination = _safe_destination(data_root, _canonical_member_name(name))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                destination.is_file()
                and destination.stat().st_size == entry.file_size
                and _crc32_file(destination) == entry.CRC
            ):
                extracted.append(destination)
                continue
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with archive.open(entry) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if temporary.stat().st_size != entry.file_size or _crc32_file(temporary) != entry.CRC:
                temporary.unlink(missing_ok=True)
                raise zipfile.BadZipFile(f"CRC or size mismatch while extracting {name}")
            temporary.replace(destination)
            extracted.append(destination)
    return extracted


def extract_base_data(data_root: str | Path) -> dict[str, int]:
    root = Path(data_root)
    archives = root / "_archives"
    counts: dict[str, int] = {}
    for csv_name in ("data_info_release.csv", "data_splits.csv"):
        source = archives / csv_name
        destination = root / csv_name
        if source.is_file():
            shutil.copy2(source, destination)
            counts[csv_name] = 1
    for zip_name in ("calibrations.zip", "kinect_cam_params.zip"):
        path = archives / zip_name
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                members = [item.filename for item in archive.infolist() if not item.is_dir()]
            counts[zip_name] = len(extract_members(path, root, members))
    ego_archive = archives / "egocentric_color.zip"
    if ego_archive.is_file():
        with zipfile.ZipFile(ego_archive) as archive:
            pv_members = [
                item.filename for item in archive.infolist() if not item.is_dir() and item.filename.endswith("_pv.txt")
            ]
        counts["pv_txt"] = len(extract_members(ego_archive, root, pv_members))
    return counts


def find_archive_members(archive_path: str | Path, predicate) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return [item.filename for item in archive.infolist() if not item.is_dir() and predicate(item.filename)]
