from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote, urlencode

import requests
import yaml
from PIL import Image

from .download import EgoBodyDownloadError
from .egobody_io import (
    EXO_IMAGE_RE,
    PV_IMAGE_RE,
    PVCalibration,
    PVRecord,
    parse_pv_file,
    sample_records,
)
from .http_archives import HttpRangeClient, RemoteTar, RemoteZip, TarMember
from .openloris import read_camera_intrinsics
from .remote_zip import RemoteZipCache


LEGACY_DATASET_ORDER = (
    "adt",
    "egobody",
    "monado",
    "princeton365",
    "hot3d",
    "incrowd_vi",
    "lamaria",
)
SUPPORTED_DATASETS = frozenset(
    (
        *LEGACY_DATASET_ORDER,
        "tum_rgbd",
        "bonn_rgbd_dynamic",
        "openloris_office",
        "droid_wrist",
        "holoassist",
        "rh20t_wrist",
        "stera10m",
    )
)
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


class DatasetDownloadError(RuntimeError):
    pass


class _GoogleDriveFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.inputs: dict[str, str] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "form" and values.get("id") == "download-form":
            self.action = values.get("action")
        elif tag == "input" and values.get("name") and values.get("value"):
            self.inputs[str(values["name"])] = str(values["value"])


@dataclass(frozen=True)
class DownloadContext:
    plan: dict[str, Any]
    data_root: Path
    target_fps: int
    workers: int
    keep_source: bool
    ffmpeg: str
    aria_mode: str
    accept_aria_licenses: bool
    egobody_netrc_file: Path | None
    accept_egobody_license: bool
    egobody_with_exo: bool
    adt_cdn_file: Path | None
    hot3d_cdn_file: Path | None
    hot3d_downloader: Path | None
    rh20t_archive: Path | None
    archive_tool: str
    robot_with_exo: bool = False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise DatasetDownloadError(
                    f"Unsafe ZIP member in {archive_path.name}: {entry.filename}"
                )
            target = (destination / relative).resolve()
            if root != target and root not in target.parents:
                raise DatasetDownloadError(
                    f"ZIP member escapes destination: {entry.filename}"
                )
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with archive.open(entry) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            os.replace(partial, target)
            extracted.append(target)
    return extracted


def load_plan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation dataset plan schema")
    profile = payload["profile"]
    datasets = payload["datasets"]
    order = tuple(profile.get("dataset_order", datasets))
    if len(order) != len(set(order)):
        raise ValueError("Profile dataset_order contains duplicates")
    missing = set(datasets) - set(order)
    if missing:
        raise ValueError(f"Profile dataset_order is missing datasets: {sorted(missing)}")
    unexpected_order = set(order) - set(datasets)
    if unexpected_order:
        raise ValueError(
            f"Profile dataset_order contains absent datasets: {sorted(unexpected_order)}"
        )
    unexpected = set(datasets) - SUPPORTED_DATASETS
    if unexpected:
        raise ValueError(f"Plan contains unsupported datasets: {sorted(unexpected)}")
    count = 0
    duration = 0.0
    sequence_keys: set[tuple[str, str]] = set()
    for dataset_name in order:
        dataset = datasets[dataset_name]
        clips = dataset["clips"]
        actual_duration = sum(float(clip["duration_s"]) for clip in clips)
        if len(clips) != int(dataset["clip_count"]):
            raise ValueError(f"{dataset_name}: clip_count does not match clips")
        if abs(actual_duration - float(dataset["duration_seconds"])) > 1e-6:
            raise ValueError(
                f"{dataset_name}: duration_seconds does not match clip durations"
            )
        for clip in clips:
            key = (dataset_name, str(clip["sequence"]))
            if key in sequence_keys:
                raise ValueError(f"Duplicate source sequence in plan: {key}")
            sequence_keys.add(key)
            if float(clip["start_s"]) < 0 or float(clip["duration_s"]) <= 0:
                raise ValueError(f"Invalid clip window: {key}")
        count += len(clips)
        duration += actual_duration
    if count != int(profile["clip_count"]):
        raise ValueError("Profile clip_count does not equal dataset total")
    if abs(duration - float(profile["duration_seconds"])) > 1e-6:
        raise ValueError("Profile duration_seconds does not equal dataset total")
    payload["_dataset_order"] = order
    payload["_plan_path"] = str(path.resolve())
    return payload


def selected_datasets(plan: dict[str, Any], value: str | None) -> tuple[str, ...]:
    order = tuple(plan["_dataset_order"])
    if not value or value == "all":
        return order
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(plan["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return tuple(name for name in order if name in requested)


def plan_summary(plan: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    rows = []
    total_clips = 0
    total_seconds = 0.0
    for name in names:
        item = plan["datasets"][name]
        clips = int(item["clip_count"])
        seconds = float(item["duration_seconds"])
        total_clips += clips
        total_seconds += seconds
        rows.append(
            {
                "dataset": name,
                "title": item["title"],
                "clips": clips,
                "seconds": seconds,
                "minutes": round(seconds / 60.0, 2),
                "access": item["access"],
                "reference_grade": item["reference_grade"],
            }
        )
    return {
        "profile": plan["profile"]["id"],
        "target_fps": plan["profile"]["target_fps"],
        "datasets": rows,
        "total_clips": total_clips,
        "total_seconds": total_seconds,
        "total_minutes": round(total_seconds / 60.0, 2),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"Profile: {summary['profile']}  target={summary['target_fps']} FPS\n"
        f"{'dataset':<15} {'clips':>5} {'minutes':>8}  {'access':<24} reference"
    )
    for row in summary["datasets"]:
        print(
            f"{row['dataset']:<15} {row['clips']:>5} {row['minutes']:>8.2f}  "
            f"{row['access']:<24} {row['reference_grade']}"
        )
    print(
        f"TOTAL           {summary['total_clips']:>5} "
        f"{summary['total_minutes']:>8.2f}"
    )


def download_https(
    url: str,
    destination: str | Path,
    timeout_s: float = 60.0,
    retries: int = 5,
    request_headers: dict[str, str] | None = None,
    trust_env: bool = True,
) -> Path:
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS sources are accepted: {url}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_name(destination.name + ".part")
    session = requests.Session()
    session.trust_env = trust_env
    session.headers["User-Agent"] = "ego-video-camera-eval-data/1.0"
    if request_headers:
        session.headers.update(request_headers)
    last_error: Exception | None = None
    for attempt in range(retries):
        completed = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={completed}-"} if completed else {}
        try:
            with session.get(
                url,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=timeout_s,
            ) as response:
                if completed and response.status_code == 200:
                    partial.unlink(missing_ok=True)
                    completed = 0
                elif completed and response.status_code != 206:
                    raise DatasetDownloadError(
                        f"Resume range rejected with HTTP {response.status_code}"
                    )
                response.raise_for_status()
                mode = "ab" if completed else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(partial, destination)
            return destination
        except (OSError, requests.RequestException, DatasetDownloadError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise DatasetDownloadError(f"Download failed: {url}") from last_error


def _google_drive_download_url(file_id: str) -> str:
    response = requests.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "ego-video-camera-eval-data/1.0",
        },
        timeout=60,
    )
    response.raise_for_status()
    if response.status_code == 206 and response.headers.get("Content-Range"):
        return response.url
    parser = _GoogleDriveFormParser()
    parser.feed(response.text)
    if (
        parser.action != "https://drive.usercontent.google.com/download"
        or parser.inputs.get("id") != file_id
        or not parser.inputs.get("uuid")
    ):
        if "Quota exceeded" in response.text:
            raise DatasetDownloadError(
                f"Google Drive quota is exhausted for public file {file_id}"
            )
        raise DatasetDownloadError(
            f"Google Drive did not return a valid download form for {file_id}"
        )
    return parser.action + "?" + urlencode(parser.inputs)


def download_google_drive_ranges(
    file_id: str,
    destination: str | Path,
    expected_size: int,
    workers: int = 4,
    chunk_size: int = 128 * 1024 * 1024,
    mirror_url: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download a large public Drive object using verified finite ranges.

    A byte-identical HTTPS mirror may fill remaining ranges when Drive applies
    quota throttling.  ``expected_sha256`` gates the assembled object before it
    is promoted from the partial path.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mirror_url is not None and not mirror_url.startswith("https://"):
        raise DatasetDownloadError("Google Drive mirror must use HTTPS")
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise DatasetDownloadError("Expected SHA-256 must contain 64 hex digits")
    if destination.is_file() and destination.stat().st_size == expected_size:
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return destination
        raise DatasetDownloadError(
            f"Existing Google Drive object failed SHA-256: {destination}"
        )
    partial = destination.with_name(destination.name + ".part")
    state_path = destination.with_name(destination.name + ".ranges.json")
    range_count = (expected_size + chunk_size - 1) // chunk_size
    identity = {
        "file_id": file_id,
        "expected_size": expected_size,
        "chunk_size": chunk_size,
        "range_count": range_count,
    }
    completed: set[int] = set()
    valid_state = False
    if state_path.is_file() and partial.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            valid_state = state.get("identity") == identity
            if valid_state:
                completed = {int(value) for value in state.get("completed", [])}
        except (OSError, TypeError, ValueError):
            valid_state = False
    if not valid_state or partial.stat().st_size != expected_size:
        completed = set()
        with partial.open("wb") as handle:
            handle.truncate(expected_size)
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(
            state_path,
            {"identity": identity, "completed": []},
        )
    pending = [index for index in range(range_count) if index not in completed]
    if not pending:
        if expected_sha256 is not None and _sha256(partial) != expected_sha256:
            raise DatasetDownloadError(
                "Completed Google Drive/mirror partial failed SHA-256"
            )
        os.replace(partial, destination)
        state_path.unlink(missing_ok=True)
        return destination

    state_lock = threading.Lock()
    url_lock = threading.Lock()
    current_url = [mirror_url or _google_drive_download_url(file_id)]
    file_descriptor = os.open(partial, os.O_RDWR)

    def refresh_url(previous: str) -> str:
        with url_lock:
            if current_url[0] == previous:
                current_url[0] = _google_drive_download_url(file_id)
            return current_url[0]

    def transfer(index: int) -> int:
        start = index * chunk_size
        end = min(expected_size, start + chunk_size) - 1
        expected = end - start + 1
        session = requests.Session()
        session.headers["User-Agent"] = "ego-video-camera-eval-data/1.0"
        url = current_url[0]
        last_error: Exception | None = None
        for attempt in range(10):
            try:
                with session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    allow_redirects=True,
                    stream=True,
                    timeout=(30, 180),
                ) as response:
                    content_range = response.headers.get("Content-Range", "")
                    wanted_range = f"bytes {start}-{end}/{expected_size}"
                    if (
                        response.status_code != 206
                        or content_range != wanted_range
                        or response.headers.get("Content-Type", "").startswith("text/html")
                    ):
                        raise DatasetDownloadError(
                            f"Google Drive rejected range {start}-{end}: "
                            f"HTTP {response.status_code}, {content_range!r}"
                        )
                    written = 0
                    for payload in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not payload:
                            continue
                        os.pwrite(file_descriptor, payload, start + written)
                        written += len(payload)
                    if written != expected:
                        raise DatasetDownloadError(
                            f"Short Google Drive range {start}-{end}: {written}/{expected}"
                        )
                os.fsync(file_descriptor)
                with state_lock:
                    completed.add(index)
                    _write_json(
                        state_path,
                        {"identity": identity, "completed": sorted(completed)},
                    )
                    progress_step = max(1, range_count // 20)
                    if len(completed) == range_count or len(completed) % progress_step == 0:
                        print(
                            f"[google-drive] {len(completed)}/{range_count} ranges "
                            f"({100.0 * len(completed) / range_count:.1f}%)",
                            flush=True,
                        )
                return index
            except (OSError, requests.RequestException, DatasetDownloadError) as error:
                last_error = error
                if attempt + 1 < 10:
                    time.sleep(min(2**attempt, 16))
                    if mirror_url is not None:
                        url = mirror_url
                    else:
                        try:
                            url = refresh_url(url)
                        except (
                            OSError,
                            requests.RequestException,
                            DatasetDownloadError,
                        ):
                            pass
        raise DatasetDownloadError(
            f"Unable to download Google Drive range {start}-{end}"
        ) from last_error

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(transfer, index) for index in pending]
            for future in as_completed(futures):
                future.result()
    finally:
        os.close(file_descriptor)
    if len(completed) != range_count:
        raise DatasetDownloadError(
            f"Google Drive range download incomplete: {len(completed)}/{range_count}"
        )
    if expected_sha256 is not None:
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            raise DatasetDownloadError(
                "Assembled Google Drive/mirror object failed SHA-256: "
                f"{actual_sha256} != {expected_sha256}"
            )
    os.replace(partial, destination)
    state_path.unlink(missing_ok=True)
    return destination


def ffmpeg_clip(
    source: Path,
    destination: Path,
    start_s: float,
    duration_s: float,
    fps: int,
    ffmpeg: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        try:
            _validate_video_clip(destination, duration_s, fps, ffmpeg)
            return destination
        except (
            DatasetDownloadError,
            OSError,
            subprocess.CalledProcessError,
            ValueError,
        ):
            destination.unlink()
    partial = destination.with_name(destination.stem + ".part" + destination.suffix)
    partial.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_s:.6f}",
        "-an",
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    subprocess.run(command, check=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise DatasetDownloadError(f"ffmpeg produced no output for {source}")
    os.replace(partial, destination)
    _validate_video_clip(destination, duration_s, fps, ffmpeg)
    return destination


def _validate_video_clip(
    path: Path, expected_duration_s: float, expected_fps: int, ffmpeg: str
) -> None:
    sibling = Path(ffmpeg).resolve().with_name("ffprobe")
    ffprobe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
    if not ffprobe:
        raise DatasetDownloadError("ffprobe is required to validate generated clips")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams", [])
    if not streams:
        raise DatasetDownloadError(f"Generated clip has no video stream: {path}")
    duration = float(payload.get("format", {}).get("duration", 0))
    frames = int(streams[0].get("nb_frames", 0))
    rate_text = str(streams[0].get("avg_frame_rate", "0/1"))
    numerator, denominator = rate_text.split("/", 1)
    rate = float(numerator) / float(denominator)
    expected_frames = int(round(expected_duration_s * expected_fps))
    if duration < expected_duration_s - 0.15:
        raise DatasetDownloadError(
            f"Generated clip is too short: {duration:.3f}s < {expected_duration_s:.3f}s"
        )
    if frames < expected_frames - 1:
        raise DatasetDownloadError(
            f"Generated clip has too few frames: {frames} < {expected_frames}"
        )
    if abs(rate - expected_fps) > 1e-3:
        raise DatasetDownloadError(
            f"Generated clip FPS mismatch: {rate} != {expected_fps}"
        )


def _clip_record(
    dataset: str,
    clip: dict[str, Any],
    output: Path,
    reference_files: list[Path],
    video_or_frames: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    files = []
    for path in [output, *reference_files]:
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "dataset": dataset,
        "sequence_id": clip["sequence"],
        "start_s": float(clip["start_s"]),
        "duration_s": float(clip["duration_s"]),
        "stratum": clip.get("stratum"),
        "input": video_or_frames,
        "files": files,
        "notes": notes or [],
    }


def _existing_clip_record(
    clip_dir: Path, dataset: str, sequence: str
) -> dict[str, Any] | None:
    path = clip_dir / "clip.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("dataset") != dataset or payload.get("sequence_id") != sequence:
        return None
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return None
    for item in files:
        try:
            recorded = Path(item["path"])
            expected_size = int(item["bytes"])
        except (KeyError, TypeError, ValueError):
            return None
        if not recorded.is_file() or recorded.stat().st_size != expected_size:
            return None
    return payload


def _normalize_incrowd_xyzw_trajectory(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with source.open("r", encoding="utf-8") as input_handle, partial.open(
        "w", encoding="utf-8"
    ) as output_handle:
        output_handle.write(
            "# tracking_timestamp_sec tx_world_device ty_world_device "
            "tz_world_device qw_world_device qx_world_device "
            "qy_world_device qz_world_device\n"
        )
        for line_number, raw_line in enumerate(input_handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise DatasetDownloadError(
                    f"Invalid InCrowd-VI trajectory row {line_number}: {source}"
                )
            output_handle.write(
                " ".join([*fields[:4], fields[7], *fields[4:7]]) + "\n"
            )
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(partial, destination)
    return destination


def _download_incrowd_trajectory(
    clip: dict[str, Any], clip_dir: Path
) -> Path:
    source_name = str(clip.get("trajectory", "trj_gt_sec_wxyz.txt"))
    order = str(clip.get("trajectory_quaternion_order", "wxyz"))
    source = download_https(
        clip["base_url"] + source_name,
        clip_dir / "reference" / source_name,
    )
    if order == "wxyz":
        return source
    if order == "xyzw":
        return _normalize_incrowd_xyzw_trajectory(
            source, clip_dir / "reference" / "trj_gt_sec_wxyz.txt"
        )
    raise DatasetDownloadError(
        f"Unsupported InCrowd-VI quaternion order {order!r}: {clip['sequence']}"
    )


def _timestamp_scale(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    deltas = sorted(
        abs(right - left)
        for left, right in zip(values, values[1:])
        if right != left
    )
    median = deltas[len(deltas) // 2] if deltas else 0
    # Absolute Unix timestamps in seconds are already around 1e9. Infer the
    # unit from frame spacing, which remains unambiguous for video streams.
    if median >= 1e7:
        return 1e9
    if median >= 1e4:
        return 1e6
    if median >= 10:
        return 1e3
    return 1.0


def _parse_camera_csv(payload: bytes) -> list[tuple[float, str]]:
    text = payload.decode("utf-8-sig", "replace")
    rows: list[tuple[float, str]] = []
    for row in csv.reader(text.splitlines()):
        if len(row) < 2 or not row[0].strip() or row[0].lstrip().startswith("#"):
            continue
        try:
            rows.append((float(row[0].strip()), row[1].strip()))
        except ValueError:
            continue
    if not rows:
        raise DatasetDownloadError("Camera CSV contains no timestamped images")
    rows.sort()
    return rows


def _sample_rows(
    rows: list[tuple[float, str]],
    start_s: float,
    duration_s: float,
    fps: int,
) -> list[tuple[float, str]]:
    raw_timestamps = [row[0] for row in rows]
    scale = _timestamp_scale(raw_timestamps)
    origin = raw_timestamps[0]
    seconds = [(value - origin) / scale for value in raw_timestamps]
    if seconds[-1] + 1e-6 < start_s + duration_s:
        raise DatasetDownloadError(
            f"Sequence has only {seconds[-1]:.2f}s, requested end is "
            f"{start_s + duration_s:.2f}s"
        )
    sampled: list[tuple[float, str]] = []
    last_index = -1
    count = int(round(duration_s * fps))
    for frame_index in range(count):
        target = start_s + frame_index / fps
        position = bisect.bisect_left(seconds, target)
        candidates = [min(position, len(rows) - 1)]
        if position:
            candidates.append(position - 1)
        selected = min(candidates, key=lambda index: abs(seconds[index] - target))
        if selected == last_index:
            continue
        sampled.append(rows[selected])
        last_index = selected
    if len(sampled) < count * 0.95:
        raise DatasetDownloadError(
            f"Sampling coverage too low: {len(sampled)}/{count} frames"
        )
    return sampled


def _parse_tum_image_list(payload: bytes | str) -> list[tuple[float, str]]:
    text = payload.decode("utf-8-sig", "replace") if isinstance(payload, bytes) else payload
    rows: list[tuple[float, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise DatasetDownloadError(
                f"Invalid TUM image-list row {line_number}: {raw_line!r}"
            )
        relative = PurePosixPath(fields[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetDownloadError(f"Unsafe image-list path: {fields[1]}")
        try:
            rows.append((float(fields[0]), relative.as_posix()))
        except ValueError as error:
            raise DatasetDownloadError(
                f"Invalid image timestamp on row {line_number}: {fields[0]!r}"
            ) from error
    if not rows:
        raise DatasetDownloadError("TUM image list contains no timestamped images")
    rows.sort()
    return rows


def _write_frame_rows(
    path: Path, sampled: list[tuple[float, str]], images: list[Path]
) -> Path:
    if len(sampled) != len(images):
        raise ValueError("Sampled timestamps and image paths have different lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("output_index", "source_timestamp", "filename"))
        for index, ((timestamp, _), image) in enumerate(zip(sampled, images)):
            writer.writerow((index, f"{timestamp:.9f}", image.name))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return path


def _validate_rgb_images(images: list[Path]) -> tuple[int, int]:
    if not images:
        raise DatasetDownloadError("No RGB images were extracted")
    dimensions: tuple[int, int] | None = None
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                if image.mode not in {"RGB", "RGBA"}:
                    raise DatasetDownloadError(
                        f"Expected color image, got mode {image.mode}: {image_path}"
                    )
                if dimensions is None:
                    dimensions = image.size
                elif image.size != dimensions:
                    raise DatasetDownloadError(
                        f"Inconsistent image dimensions: {image_path} is {image.size}, "
                        f"expected {dimensions}"
                    )
                image.verify()
        except (OSError, ValueError) as error:
            raise DatasetDownloadError(f"Invalid RGB image: {image_path}") from error
    assert dimensions is not None
    return dimensions


def _strict_rgb_record(
    dataset_name: str,
    dataset: dict[str, Any],
    clip: dict[str, Any],
    clip_dir: Path,
    sampled: list[tuple[float, str]],
    images: list[Path],
    references: list[Path],
    camera: dict[str, Any],
    target_fps: int,
    notes: list[str],
) -> dict[str, Any]:
    expected = int(round(float(clip["duration_s"]) * target_fps))
    if len(images) != expected:
        raise DatasetDownloadError(
            f"{clip['sequence']}: only {len(images)}/{expected} sampled RGB frames"
        )
    width, height = _validate_rgb_images(images)
    frames_csv = _write_frame_rows(clip_dir / "frames.csv", sampled, images)
    frame_manifest = _write_image_manifest(clip_dir / "frame_manifest.csv", images)
    camera_path = clip_dir / "reference" / "camera.json"
    camera_payload = {
        **camera,
        "width": width,
        "height": height,
        "native_color_stream": True,
        "fisheye": False,
    }
    _write_json(camera_path, camera_payload)
    record = _clip_record(
        dataset_name,
        clip,
        frames_csv,
        [*references, frame_manifest, camera_path],
        str(clip_dir / "frames"),
        notes,
    )
    record.update(
        {
            "reference_grade": dataset["reference_grade"],
            "reference_type": dataset["reference_type"],
            "frame_count": len(images),
            "expected_frame_count": expected,
            "input_characteristics": {
                "color": "RGB",
                "projection_model": camera_payload.get("projection_model", "pinhole"),
                "distortion_model": camera_payload.get("distortion_model"),
                "native_color_stream": True,
                "fisheye": False,
            },
        }
    )
    _write_json(clip_dir / "clip.json", record)
    return record


def _checked_download(
    url: str, destination: Path, expected_size: int | None = None
) -> Path:
    result = download_https(url, destination)
    if expected_size is not None and result.stat().st_size != expected_size:
        result.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"Downloaded size mismatch for {url}: expected {expected_size} bytes"
        )
    return result


def _tar_members_by_suffix(
    archive: tarfile.TarFile, suffixes: Iterable[str]
) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    result: dict[str, tarfile.TarInfo] = {}
    for suffix in suffixes:
        candidates = [member for member in members if member.name.endswith(suffix)]
        if not candidates:
            raise DatasetDownloadError(f"TGZ member *{suffix} was not found")
        result[suffix] = min(candidates, key=lambda member: len(member.name))
    return result


def _extract_tar_files(
    archive: tarfile.TarFile,
    members: list[tuple[tarfile.TarInfo, Path]],
) -> list[Path]:
    outputs: list[Path] = []
    for member, destination in sorted(members, key=lambda item: item[0].offset_data):
        if not member.isfile():
            raise DatasetDownloadError(f"Expected regular TGZ member: {member.name}")
        source = archive.extractfile(member)
        if source is None:
            raise DatasetDownloadError(f"Unable to read TGZ member: {member.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        with source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if partial.stat().st_size != member.size:
            partial.unlink(missing_ok=True)
            raise DatasetDownloadError(f"Short TGZ extraction: {member.name}")
        os.replace(partial, destination)
        outputs.append(destination)
    return outputs


def _find_suffix(names: Iterable[str], suffix: str, required: bool = True) -> str | None:
    matches = [name for name in names if name.endswith(suffix)]
    if not matches:
        if required:
            raise DatasetDownloadError(f"Archive member *{suffix} was not found")
        return None
    if len(matches) > 1:
        matches.sort(key=len)
    return matches[0]


def _find_any_suffix(
    names: Iterable[str], suffixes: Iterable[str], required: bool = True
) -> str | None:
    materialized = tuple(names)
    for suffix in suffixes:
        match = _find_suffix(materialized, suffix, required=False)
        if match is not None:
            return match
    if required:
        raise DatasetDownloadError(
            f"Archive member matching one of {list(suffixes)} was not found"
        )
    return None


def _extract_euroc_clip(
    archive_url: str,
    archive_cache: Path,
    clip_dir: Path,
    clip: dict[str, Any],
    target_fps: int,
    workers: int,
    require_archive_gt: bool = True,
) -> tuple[Path, list[Path]]:
    archive = RemoteZip(archive_url, archive_cache)
    camera_csv_name = _find_any_suffix(
        archive.names,
        ("/mav0/cam0/data.csv", "/aria/cam0/data.csv"),
    )
    assert camera_csv_name is not None
    camera_rows = _parse_camera_csv(archive.read(camera_csv_name))
    sampled = _sample_rows(
        camera_rows,
        float(clip["start_s"]),
        float(clip["duration_s"]),
        target_fps,
    )
    data_prefix = camera_csv_name.rsplit("/", 1)[0] + "/data/"
    by_basename = {
        Path(name).name: name
        for name in archive.names
        if name.startswith(data_prefix) and not name.endswith("/")
    }
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for timestamp, filename in sampled:
            member = by_basename.get(Path(filename).name)
            if member is None:
                raise DatasetDownloadError(f"Image member missing: {filename}")
            destination = frame_dir / Path(filename).name
            jobs.append(executor.submit(archive.extract, member, destination))
        for future in as_completed(jobs):
            future.result()
    support: list[Path] = []
    support_members: list[tuple[str, str]] = [
        (camera_csv_name, "camera_timestamps.csv")
    ]
    sensor_member = _find_any_suffix(
        archive.names,
        ("/mav0/cam0/sensor.yaml", "/aria/cam0/sensor.yaml"),
        required=False,
    )
    if sensor_member:
        support_members.append((sensor_member, "sensor.yaml"))
    gt_member = _find_suffix(
        archive.names, "/mav0/gt/data.csv", required=require_archive_gt
    )
    if gt_member:
        support_members.append((gt_member, "gt_trajectory.csv"))
    for member, output_name in support_members:
        if member:
            destination = clip_dir / "reference" / output_name
            archive.extract(member, destination)
            support.append(destination)
    rows_path = clip_dir / "frames.csv"
    rows_path.write_text(
        "source_timestamp,filename\n"
        + "".join(f"{timestamp:.0f},{Path(name).name}\n" for timestamp, name in sampled),
        encoding="utf-8",
    )
    support.append(rows_path)
    return frame_dir, support


def download_monado(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["monado"]
    root = ctx.data_root / "monado"
    cache = ctx.data_root / "_cache" / "remote_zip" / "monado"
    records = []
    devices = sorted({clip["device"] for clip in dataset["clips"]})
    calibration_paths: dict[str, list[Path]] = {}
    for device in devices:
        shared = root / "_shared" / device
        paths = []
        for filename in ("calibration.json", "mocap_calibration.json"):
            remote_path = f"M_monado_datasets/{device}/extras/{filename}"
            url = HF_RESOLVE.format(
                repo=dataset["hf_repo"], path=quote(remote_path, safe="/")
            )
            destination = shared / filename
            try:
                download_https(url, destination)
                paths.append(destination)
            except DatasetDownloadError:
                if filename == "calibration.json":
                    raise
        calibration_paths[device] = paths
    for clip in dataset["clips"]:
        print(f"[monado] {clip['sequence']}")
        archive_url = HF_RESOLVE.format(
            repo=dataset["hf_repo"], path=quote(clip["archive"], safe="/")
        )
        clip_dir = root / "clips" / clip["sequence"]
        frame_dir, support = _extract_euroc_clip(
            archive_url,
            cache,
            clip_dir,
            clip,
            ctx.target_fps,
            ctx.workers,
        )
        record = _clip_record(
            "monado",
            clip,
            clip_dir / "frames.csv",
            support + calibration_paths[clip["device"]],
            str(frame_dir),
            ["grayscale images; replicate to three channels at model input"],
        )
        _write_json(clip_dir / "clip.json", record)
        records.append(record)
    return records


def _member_by_suffix(entries: list[TarMember], suffix: str) -> TarMember:
    candidates = [entry for entry in entries if entry.name.endswith(suffix)]
    if not candidates:
        raise DatasetDownloadError(f"TAR member *{suffix} was not found")
    return candidates[0]


def download_princeton365(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["princeton365"]
    root = ctx.data_root / "princeton365"
    records = []
    suffixes = (
        ".json",
        ".mp4",
        ".user_camera_mtx.npy",
        ".user_camera_dist.npy",
        ".relative_transform.npy",
        ".gt_trajectory.txt",
    )
    for clip in dataset["clips"]:
        print(f"[princeton365] {clip['sequence']}")
        clip_dir = root / "clips" / clip["sequence"]
        existing = _existing_clip_record(
            clip_dir, "princeton365", clip["sequence"]
        )
        if existing is not None:
            records.append(existing)
            continue
        shard_path = f"validation/{int(clip['shard']):06d}.tar"
        url = HF_RESOLVE.format(
            repo=dataset["hf_repo"], path=quote(shard_path, safe="/")
        )
        archive = RemoteTar(url, HttpRangeClient())
        entries = archive.scan(stop_suffixes=(".gt_trajectory.txt",), max_members=64)
        reference_dir = clip_dir / "reference"
        source_video = root / "_sources" / f"{clip['sequence']}.mp4"
        video_member = _member_by_suffix(entries, ".mp4")
        archive.extract(video_member, source_video)
        output_video = ffmpeg_clip(
            source_video,
            clip_dir / "video.mp4",
            float(clip["start_s"]),
            float(clip["duration_s"]),
            ctx.target_fps,
            ctx.ffmpeg,
        )
        references = []
        for suffix in suffixes:
            if suffix == ".mp4":
                continue
            member = _member_by_suffix(entries, suffix)
            destination = reference_dir / Path(member.name).name
            archive.extract(member, destination)
            references.append(destination)
        if not ctx.keep_source:
            source_video.unlink(missing_ok=True)
        record = _clip_record(
            "princeton365",
            clip,
            output_video,
            references,
            str(output_video),
            ["handheld/user-camera RGB; not a strict head-mounted sequence"],
        )
        _write_json(clip_dir / "clip.json", record)
        records.append(record)
    return records


def download_incrowd(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["incrowd_vi"]
    root = ctx.data_root / "incrowd_vi"
    calibration = download_https(
        dataset["calibration_url"], root / "_shared" / "Calibration-Data.md"
    )
    records = []
    for clip in dataset["clips"]:
        print(f"[incrowd_vi] {clip['sequence']}")
        clip_dir = root / "clips" / clip["sequence"]
        existing = _existing_clip_record(clip_dir, "incrowd_vi", clip["sequence"])
        if existing is not None:
            records.append(existing)
            continue
        source_video = root / "_sources" / clip["video"]
        download_https(clip["base_url"] + clip["video"], source_video)
        gt = _download_incrowd_trajectory(clip, clip_dir)
        output_video = ffmpeg_clip(
            source_video,
            clip_dir / "video.mp4",
            float(clip["start_s"]),
            float(clip["duration_s"]),
            ctx.target_fps,
            ctx.ffmpeg,
        )
        if not ctx.keep_source:
            source_video.unlink(missing_ok=True)
        record = _clip_record(
            "incrowd_vi",
            clip,
            output_video,
            [gt, calibration],
            str(output_video),
            ["all selected InCrowd-VI sequences are indoor"],
        )
        _write_json(clip_dir / "clip.json", record)
        records.append(record)
    return records


def download_lamaria(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["lamaria"]
    root = ctx.data_root / "lamaria"
    cache = ctx.data_root / "_cache" / "remote_zip" / "lamaria"
    records = []
    for clip in dataset["clips"]:
        sequence = clip["sequence"]
        print(f"[lamaria] {sequence}")
        split = clip["split"]
        archive_url = (
            f"https://cvg-data.inf.ethz.ch/lamaria/asl_folder/{split}/{sequence}.zip"
        )
        clip_dir = root / "clips" / sequence
        frame_dir, support = _extract_euroc_clip(
            archive_url,
            cache,
            clip_dir,
            clip,
            ctx.target_fps,
            ctx.workers,
            require_archive_gt=False,
        )
        gt = download_https(
            "https://cvg-data.inf.ethz.ch/lamaria/ground_truth/"
            f"pseudo_dense/{sequence}.txt",
            clip_dir / "reference" / f"{sequence}_pseudo_dense.txt",
        )
        calibration = download_https(
            "https://cvg-data.inf.ethz.ch/lamaria/pinhole_calibrations/"
            f"{split}/{sequence}.json",
            clip_dir / "reference" / f"{sequence}_pinhole_calibration.json",
        )
        record = _clip_record(
            "lamaria",
            clip,
            clip_dir / "frames.csv",
            support + [gt, calibration],
            str(frame_dir),
            [
                "ASL cam0 is grayscale; replicate to three channels at model input",
                "trajectory is pseudo-dense GT, not independent framewise mocap",
            ],
        )
        _write_json(clip_dir / "clip.json", record)
        records.append(record)
    return records


def _existing_strict_rgb_record(
    clip_dir: Path,
    dataset_name: str,
    sequence: str,
    expected_frames: int,
) -> dict[str, Any] | None:
    record = _existing_clip_record(clip_dir, dataset_name, sequence)
    if record is None or int(record.get("frame_count", -1)) != expected_frames:
        return None
    frames = [
        path
        for path in (clip_dir / "frames").iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ] if (clip_dir / "frames").is_dir() else []
    required = (
        clip_dir / "frames.csv",
        clip_dir / "frame_manifest.csv",
        clip_dir / "reference" / "camera.json",
    )
    if len(frames) != expected_frames or not all(
        path.is_file() and path.stat().st_size > 0 for path in required
    ):
        return None
    return record


def download_tum_rgbd(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["tum_rgbd"]
    root = ctx.data_root / "tum_rgbd"
    source_root = ctx.data_root / "_cache" / "sources" / "tum_rgbd"
    records: list[dict[str, Any]] = []
    for clip in dataset["clips"]:
        sequence = str(clip["sequence"])
        print(f"[tum_rgbd] {sequence}")
        clip_dir = root / "clips" / sequence
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        existing = _existing_strict_rgb_record(
            clip_dir, "tum_rgbd", sequence, expected
        )
        if existing is not None:
            records.append(existing)
            continue
        source = source_root / f"{sequence}.tgz"
        _checked_download(clip["url"], source, int(clip["source_bytes"]))
        try:
            with tarfile.open(source, mode="r:gz") as archive:
                required = _tar_members_by_suffix(
                    archive, ("/rgb.txt", "/groundtruth.txt")
                )
                rgb_member = required["/rgb.txt"]
                rgb_file = archive.extractfile(rgb_member)
                if rgb_file is None:
                    raise DatasetDownloadError(f"Unable to read {rgb_member.name}")
                with rgb_file:
                    rows = _parse_tum_image_list(rgb_file.read())
                sampled = _sample_rows(
                    rows,
                    float(clip["start_s"]),
                    float(clip["duration_s"]),
                    ctx.target_fps,
                )
                prefix = rgb_member.name[: -len("rgb.txt")]
                member_index = {member.name: member for member in archive.getmembers()}
                frame_dir = clip_dir / "frames"
                jobs: list[tuple[tarfile.TarInfo, Path]] = []
                images: list[Path] = []
                for index, (_, relative_name) in enumerate(sampled):
                    source_name = prefix + relative_name
                    try:
                        member = member_index[source_name]
                    except KeyError as error:
                        raise DatasetDownloadError(
                            f"TGZ image member is missing: {source_name}"
                        ) from error
                    destination = frame_dir / f"{index:06d}_{Path(relative_name).name}"
                    jobs.append((member, destination))
                    images.append(destination)
                source_rgb = clip_dir / "reference" / "source_rgb.txt"
                groundtruth = clip_dir / "reference" / "groundtruth.txt"
                jobs.extend(
                    (
                        (rgb_member, source_rgb),
                        (required["/groundtruth.txt"], groundtruth),
                    )
                )
                _extract_tar_files(archive, jobs)
            record = _strict_rgb_record(
                "tum_rgbd",
                dataset,
                clip,
                clip_dir,
                sampled,
                images,
                [source_rgb, groundtruth],
                dataset["cameras"][clip["camera"]],
                ctx.target_fps,
                [
                    "native Kinect RGB stream; Brown-Conrady/pinhole, not fisheye",
                    "groundtruth.txt is external motion-capture camera C2W in TUM format",
                    "depth frames intentionally omitted from the RGB-only evaluation subset",
                ],
            )
            record["source"] = {
                "url": clip["url"],
                "archive_bytes": int(clip["source_bytes"]),
                "retained_full_archive": bool(ctx.keep_source),
            }
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
        finally:
            if not ctx.keep_source:
                source.unlink(missing_ok=True)
    return records


def download_bonn_rgbd_dynamic(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["bonn_rgbd_dynamic"]
    root = ctx.data_root / "bonn_rgbd_dynamic"
    cache_root = ctx.data_root / "_cache" / "remote_zip" / "bonn_rgbd_dynamic"
    records: list[dict[str, Any]] = []
    for clip in dataset["clips"]:
        sequence = str(clip["sequence"])
        print(f"[bonn_rgbd_dynamic] {sequence}")
        clip_dir = root / "clips" / sequence
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        existing = _existing_strict_rgb_record(
            clip_dir, "bonn_rgbd_dynamic", sequence, expected
        )
        if existing is not None:
            records.append(existing)
            continue
        url = dataset["base_url"] + clip["archive"]
        archive = RemoteZip(url, cache_root)
        if archive.remote.size != int(clip["source_bytes"]):
            raise DatasetDownloadError(
                f"Remote Bonn archive size changed for {clip['archive']}: "
                f"{archive.remote.size} != {clip['source_bytes']}"
            )
        prefix = f"{sequence}/"
        rgb_member = prefix + "rgb.txt"
        gt_member = prefix + "groundtruth.txt"
        rows = _parse_tum_image_list(archive.read(rgb_member))
        sampled = _sample_rows(
            rows,
            float(clip["start_s"]),
            float(clip["duration_s"]),
            ctx.target_fps,
        )
        frame_dir = clip_dir / "frames"
        images = [
            frame_dir / f"{index:06d}_{Path(relative_name).name}"
            for index, (_, relative_name) in enumerate(sampled)
        ]
        with ThreadPoolExecutor(max_workers=max(1, ctx.workers)) as executor:
            futures = [
                executor.submit(
                    archive.extract,
                    prefix + relative_name,
                    destination,
                )
                for (_, relative_name), destination in zip(sampled, images)
            ]
            for future in as_completed(futures):
                future.result()
        source_rgb = archive.extract(
            rgb_member, clip_dir / "reference" / "source_rgb.txt"
        )
        groundtruth = archive.extract(
            gt_member, clip_dir / "reference" / "groundtruth.txt"
        )
        record = _strict_rgb_record(
            "bonn_rgbd_dynamic",
            dataset,
            clip,
            clip_dir,
            sampled,
            images,
            [source_rgb, groundtruth],
            dataset["camera"],
            ctx.target_fps,
            [
                "native ASUS Xtion RGB stream; radial-tangential pinhole, not fisheye",
                "groundtruth.txt is the official OptiTrack reference in TUM format",
                "official marker/sensor transforms are preserved in camera.json",
                "depth frames intentionally omitted from the RGB-only evaluation subset",
            ],
        )
        record["source"] = {
            "url": url,
            "archive_bytes": archive.remote.size,
            "remote_identity": {
                "etag": archive.remote.etag,
                "last_modified": archive.remote.last_modified,
            },
            "retained_full_archive": False,
        }
        _write_json(clip_dir / "clip.json", record)
        records.append(record)
    if not ctx.keep_source:
        _remove_owned_eval_cache(cache_root, ctx.data_root)
    return records


def _openloris_extract(
    archive_tool: str,
    source: Path,
    staging: Path,
    members: list[str],
    data_root: Path,
) -> None:
    if staging.exists():
        _remove_owned_eval_cache(staging, data_root)
    staging.mkdir(parents=True)
    subprocess.run(
        [archive_tool, "-xf", str(source), "-C", str(staging), *members],
        check=True,
    )
    missing = [member for member in members if not (staging / member).is_file()]
    if missing:
        raise DatasetDownloadError(
            f"OpenLORIS 7z extraction missed {len(missing)} requested members"
        )


def download_openloris_office(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["openloris_office"]
    root = ctx.data_root / "openloris_office"
    cache_root = ctx.data_root / "_cache" / "openloris_office"
    package = RemoteTar(dataset["package_tar_url"], HttpRangeClient())
    if package.remote.size != int(dataset["package_tar_bytes"]):
        raise DatasetDownloadError(
            f"OpenLORIS package TAR changed size: {package.remote.size} != "
            f"{dataset['package_tar_bytes']}"
        )
    entries = {entry.name: entry for entry in package.scan(max_members=32)}
    records: list[dict[str, Any]] = []
    for clip in dataset["clips"]:
        sequence = str(clip["sequence"])
        print(f"[openloris_office] {sequence}")
        clip_dir = root / "clips" / sequence
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        existing = _existing_strict_rgb_record(
            clip_dir, "openloris_office", sequence, expected
        )
        if existing is not None:
            records.append(existing)
            continue
        archive_name = str(clip["archive_member"])
        try:
            entry = entries[archive_name]
        except KeyError as error:
            raise DatasetDownloadError(
                f"OpenLORIS TAR member is missing: {archive_name}"
            ) from error
        if entry.size != int(clip["source_bytes"]):
            raise DatasetDownloadError(
                f"OpenLORIS member size changed for {archive_name}: "
                f"{entry.size} != {clip['source_bytes']}"
            )
        source = cache_root / archive_name
        package.extract(entry, source)
        staging = cache_root / "staging" / sequence
        try:
            color_member = f"{sequence}/color.txt"
            color_text = subprocess.check_output(
                [ctx.archive_tool, "-xOf", str(source), color_member]
            )
            rows = _parse_tum_image_list(color_text)
            sampled = _sample_rows(
                rows,
                float(clip["start_s"]),
                float(clip["duration_s"]),
                ctx.target_fps,
            )
            reference_members = [
                color_member,
                f"{sequence}/groundtruth.txt",
                f"{sequence}/sensors.yaml",
                f"{sequence}/trans_matrix.yaml",
            ]
            image_members = [
                f"{sequence}/{relative_name}" for _, relative_name in sampled
            ]
            _openloris_extract(
                ctx.archive_tool,
                source,
                staging,
                [*reference_members, *image_members],
                ctx.data_root,
            )
            frame_dir = clip_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            images: list[Path] = []
            for index, member in enumerate(image_members):
                destination = frame_dir / f"{index:06d}_{Path(member).name}"
                os.replace(staging / member, destination)
                images.append(destination)
            reference_dir = clip_dir / "reference"
            references: list[Path] = []
            reference_names = {
                color_member: "source_color.txt",
                f"{sequence}/groundtruth.txt": "groundtruth.txt",
                f"{sequence}/sensors.yaml": "sensors.yaml",
                f"{sequence}/trans_matrix.yaml": "trans_matrix.yaml",
            }
            for member, output_name in reference_names.items():
                destination = reference_dir / output_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / member, destination)
                references.append(destination)
            try:
                camera = {
                    **dataset["camera_selection"],
                    **read_camera_intrinsics(
                        reference_dir / "sensors.yaml",
                        str(dataset["camera_selection"]["camera_key"]),
                    ),
                }
            except (OSError, ValueError) as error:
                raise DatasetDownloadError(
                    f"Unable to read OpenLORIS color intrinsics for {sequence}"
                ) from error
            record = _strict_rgb_record(
                "openloris_office",
                dataset,
                clip,
                clip_dir,
                sampled,
                images,
                references,
                camera,
                ctx.target_fps,
                [
                    "native D435i color stream selected; T265 fisheye streams are excluded",
                    "sensors.yaml identifies d400_color_optical_frame as pinhole",
                    "groundtruth.txt is the official office OptiTrack reference",
                    "depth, fisheye, and IMU samples intentionally omitted from this RGB-only subset",
                ],
            )
            record["source"] = {
                "url": dataset["package_tar_url"],
                "package_tar_bytes": package.remote.size,
                "package_etag": package.remote.etag,
                "member": archive_name,
                "member_bytes": entry.size,
                "retained_full_archive": bool(ctx.keep_source),
            }
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
        finally:
            if staging.exists():
                _remove_owned_eval_cache(staging, ctx.data_root)
            if not ctx.keep_source:
                source.unlink(missing_ok=True)
    return records


def _sample_droid_indices(
    timestamps_ms: Iterable[int], start_s: float, duration_s: float, fps: int
) -> list[tuple[float, int]]:
    values = [int(value) for value in timestamps_ms]
    if len(values) < 2 or any(right < left for left, right in zip(values, values[1:])):
        raise DatasetDownloadError("DROID capture timestamps are missing or non-monotonic")
    sampled = _sample_rows(
        [(float(timestamp), str(index)) for index, timestamp in enumerate(values)],
        start_s,
        duration_s,
        fps,
    )
    return [(timestamp, int(index)) for timestamp, index in sampled]


def _extract_indexed_video_frames(
    source: Path,
    clip_dir: Path,
    sampled: list[tuple[float, int]],
    ffmpeg: str,
    data_root: Path,
) -> list[Path]:
    frame_dir = clip_dir / "frames"
    destinations = [
        frame_dir / f"{output_index:06d}_source_{source_index:06d}.png"
        for output_index, (_, source_index) in enumerate(sampled)
    ]
    if (
        frame_dir.is_dir()
        and len([path for path in frame_dir.iterdir() if path.is_file()])
        == len(destinations)
        and all(path.is_file() and path.stat().st_size > 0 for path in destinations)
    ):
        _validate_rgb_images(destinations)
        return destinations
    staging = clip_dir / ".frames-staging"
    if staging.exists():
        _remove_owned_eval_cache(staging, data_root)
    staging.mkdir(parents=True)
    unique_source_indices = list(dict.fromkeys(source_index for _, source_index in sampled))
    expression = "+".join(
        f"eq(n\\,{source_index})" for source_index in unique_source_indices
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(staging / "decoded_%06d.png"),
        ],
        check=True,
    )
    decoded = sorted(staging.glob("decoded_*.png"))
    if len(decoded) != len(unique_source_indices):
        raise DatasetDownloadError(
            "Indexed video decode returned "
            f"{len(decoded)}/{len(unique_source_indices)} unique source frames"
        )
    decoded_by_index: dict[int, Path] = {}
    for decoded_path, source_index in zip(decoded, unique_source_indices):
        indexed_path = staging / f"source_{source_index:06d}.png"
        os.replace(
            decoded_path,
            indexed_path,
        )
        decoded_by_index[source_index] = indexed_path
    for output_index, (_, source_index) in enumerate(sampled):
        destination = staging / f"{output_index:06d}_source_{source_index:06d}.png"
        shutil.copyfile(decoded_by_index[source_index], destination)
    for indexed_path in decoded_by_index.values():
        indexed_path.unlink()
    staged = [staging / path.name for path in destinations]
    _validate_rgb_images(staged)
    if frame_dir.exists():
        _remove_owned_eval_cache(frame_dir, data_root)
    os.replace(staging, frame_dir)
    return destinations


def _write_droid_trajectory(
    destination: Path,
    sampled: list[tuple[float, int]],
    poses: Iterable[Iterable[float]],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    origin_ms = sampled[0][0]
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "output_index",
                "source_frame_index",
                "estimated_capture_ms",
                "clip_time_s",
                "tx",
                "ty",
                "tz",
                "rx_xyz_rad",
                "ry_xyz_rad",
                "rz_xyz_rad",
            )
        )
        for output_index, (((timestamp, source_index), pose)) in enumerate(
            zip(sampled, poses)
        ):
            values = [float(value) for value in pose]
            if len(values) != 6:
                raise DatasetDownloadError("DROID camera pose does not have 6 values")
            writer.writerow(
                (
                    output_index,
                    source_index,
                    int(timestamp),
                    f"{(timestamp - origin_ms) / 1000.0:.9f}",
                    *(f"{value:.12g}" for value in values),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def download_droid_wrist(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["droid_wrist"]
    root = ctx.data_root / "droid_wrist"
    cache_root = ctx.data_root / "_cache" / "droid_wrist"
    expected_records: list[dict[str, Any] | None] = []
    for clip in dataset["clips"]:
        clip_dir = root / "clips" / str(clip["sequence"])
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        expected_records.append(
            _existing_strict_rgb_record(
                clip_dir, "droid_wrist", str(clip["sequence"]), expected
            )
        )
    if all(record is not None for record in expected_records):
        return [record for record in expected_records if record is not None]
    try:
        import h5py
    except ImportError as error:
        raise DatasetDownloadError(
            "DROID subset extraction requires h5py in the selected Python environment"
        ) from error

    intrinsics_path = _checked_download(
        str(dataset["intrinsics_url"]),
        cache_root / "intrinsics.json",
        int(dataset["intrinsics_bytes"]),
    )
    if _sha256(intrinsics_path) != str(dataset["intrinsics_sha256"]):
        intrinsics_path.unlink(missing_ok=True)
        raise DatasetDownloadError("DROID intrinsics annotation SHA-256 mismatch")
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetDownloadError("Unable to parse DROID intrinsics annotation") from error

    records: list[dict[str, Any]] = []
    for clip, existing in zip(dataset["clips"], expected_records):
        sequence = str(clip["sequence"])
        print(f"[droid_wrist] {sequence}")
        if existing is not None:
            records.append(existing)
            continue
        clip_dir = root / "clips" / sequence
        relative = str(clip["relative_path"])
        base_url = str(dataset["raw_base_url"]).rstrip("/") + "/" + relative
        metadata_url = base_url + "/" + str(clip["metadata_file"])
        trajectory_url = base_url + "/trajectory.h5"
        video_url = (
            base_url
            + "/recordings/MP4/"
            + str(clip["wrist_serial"])
            + ".mp4"
        )
        source_dir = cache_root / sequence
        metadata_path = _checked_download(
            metadata_url,
            clip_dir / "reference" / "metadata.json",
            int(clip["metadata_bytes"]),
        )
        trajectory_path = _checked_download(
            trajectory_url,
            source_dir / "trajectory.h5",
            int(clip["trajectory_bytes"]),
        )
        video_path = _checked_download(
            video_url,
            source_dir / f"{clip['wrist_serial']}.mp4",
            int(clip["video_bytes"]),
        )
        success = False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("uuid") != sequence
                or str(metadata.get("wrist_cam_serial"))
                != str(clip["wrist_serial"])
                or int(metadata.get("trajectory_length", -1))
                != int(clip["trajectory_length"])
                or metadata.get("success") is not True
            ):
                raise DatasetDownloadError(f"DROID metadata gate failed: {sequence}")
            serial = str(clip["wrist_serial"])
            pose_key = f"observation/camera_extrinsics/{serial}_left"
            timestamp_key = (
                f"observation/timestamp/cameras/{serial}_estimated_capture"
            )
            with h5py.File(trajectory_path, "r") as archive:
                if pose_key not in archive or timestamp_key not in archive:
                    raise DatasetDownloadError(
                        f"DROID H5 lacks wrist pose/timestamp datasets: {sequence}"
                    )
                timestamps = archive[timestamp_key][:]
                pose_dataset = archive[pose_key]
                if (
                    len(timestamps) != int(clip["trajectory_length"])
                    or len(pose_dataset) != len(timestamps)
                ):
                    raise DatasetDownloadError(
                        f"DROID H5 trajectory length mismatch: {sequence}"
                    )
                sampled = _sample_droid_indices(
                    timestamps,
                    float(clip["start_s"]),
                    float(clip["duration_s"]),
                    ctx.target_fps,
                )
                poses = pose_dataset[[source_index for _, source_index in sampled]].tolist()
            images = _extract_indexed_video_frames(
                video_path, clip_dir, sampled, ctx.ffmpeg, ctx.data_root
            )
            trajectory = _write_droid_trajectory(
                clip_dir / "reference" / "camera_to_robot_base.csv",
                sampled,
                poses,
            )
            try:
                calibration = intrinsics[sequence][serial]
            except (KeyError, TypeError) as error:
                raise DatasetDownloadError(
                    f"DROID intrinsics missing for {sequence}/{serial}"
                ) from error
            matrix = [float(value) for value in calibration["cameraMatrix"]]
            if len(matrix) != 4:
                raise DatasetDownloadError("Unexpected DROID cameraMatrix format")
            camera = {
                **dataset["camera"],
                "fx": matrix[0],
                "cx": matrix[1],
                "fy": matrix[2],
                "cy": matrix[3],
                "distortion_coefficients": [
                    float(value) for value in calibration.get("distCoeffs", [])
                ],
                "annotation_repo": dataset["annotation_repo"],
                "annotation_revision": dataset["annotation_revision"],
            }
            rows = [(timestamp, str(source_index)) for timestamp, source_index in sampled]
            record = _strict_rgb_record(
                "droid_wrist",
                dataset,
                clip,
                clip_dir,
                rows,
                images,
                [metadata_path, trajectory],
                camera,
                ctx.target_fps,
                [
                    "native rectified left RGB from the robot wrist ZED camera",
                    "frames are selected by H5 observation index, not MP4 presentation time",
                    "estimated_capture timestamps define real acquisition time at about 15 FPS",
                    "camera-to-base is a robot-kinematic reference, not external motion capture",
                ],
            )
            record["source"] = {
                "release_version": dataset["release_version"],
                "metadata_url": metadata_url,
                "trajectory_url": trajectory_url,
                "video_url": video_url,
                "metadata_sha256": _sha256(metadata_path),
                "trajectory_sha256": _sha256(trajectory_path),
                "video_sha256": _sha256(video_path),
                "source_trajectory_frames": int(clip["trajectory_length"]),
                "retained_full_episode_sources": bool(ctx.keep_source),
            }
            record["sampling"] = {
                "basis": "estimated_capture_timestamp_then_h5_frame_index",
                "first_source_frame": sampled[0][1],
                "last_source_frame": sampled[-1][1],
                "sampled_frames": len(sampled),
            }
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
            success = True
        finally:
            if success and not ctx.keep_source and not ctx.robot_with_exo:
                trajectory_path.unlink(missing_ok=True)
                video_path.unlink(missing_ok=True)
    if not ctx.keep_source and not ctx.robot_with_exo:
        intrinsics_path.unlink(missing_ok=True)
        if cache_root.exists():
            _remove_owned_eval_cache(cache_root, ctx.data_root)
    return records


def _normalized_tar_name(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _tar_member_exact(entries: Iterable[TarMember], name: str) -> TarMember:
    matches = [
        entry for entry in entries if _normalized_tar_name(entry.name) == name
    ]
    if len(matches) != 1:
        raise DatasetDownloadError(
            f"Expected exactly one TAR member {name!r}, found {len(matches)}"
        )
    return matches[0]


def _parse_holoassist_poses(
    payload: bytes,
) -> list[tuple[float, int, list[float]]]:
    rows: list[tuple[float, int, list[float]]] = []
    for line_number, raw_line in enumerate(
        payload.decode("utf-8-sig", "replace").splitlines(), start=1
    ):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) != 18:
            raise DatasetDownloadError(
                f"Invalid HoloAssist pose row {line_number}: {len(fields)} fields"
            )
        try:
            video_time = float(fields[0])
            filetime = int(fields[1])
            matrix = [float(value) for value in fields[2:]]
        except ValueError as error:
            raise DatasetDownloadError(
                f"Invalid HoloAssist pose values on row {line_number}"
            ) from error
        if any(abs(value - expected) > 1e-6 for value, expected in zip(matrix[12:], (0, 0, 0, 1))):
            raise DatasetDownloadError(
                f"Invalid HoloAssist homogeneous pose row {line_number}"
            )
        rows.append((video_time, filetime, matrix))
    if len(rows) < 2 or any(
        right[0] <= left[0] or right[1] < left[1]
        for left, right in zip(rows, rows[1:])
    ):
        raise DatasetDownloadError("HoloAssist poses are missing or non-monotonic")
    return rows


def _parse_holoassist_intrinsics(payload: bytes) -> dict[str, Any]:
    lines = [line.strip() for line in payload.decode("utf-8-sig", "replace").splitlines() if line.strip()]
    if len(lines) != 1:
        raise DatasetDownloadError("HoloAssist intrinsics must contain exactly one row")
    try:
        values = [float(value) for value in lines[0].split()]
    except ValueError as error:
        raise DatasetDownloadError("Invalid HoloAssist intrinsics values") from error
    if len(values) != 25:
        raise DatasetDownloadError(
            f"Unexpected HoloAssist intrinsics format: {len(values)} fields"
        )
    width, height = int(values[23]), int(values[24])
    if width <= 0 or height <= 0:
        raise DatasetDownloadError("Invalid HoloAssist calibration dimensions")
    return {
        "source_width": width,
        "source_height": height,
        "intrinsics_matrix": values[:9],
        "radial_distortion": values[9:15],
        "tangential_distortion": values[15:17],
        "focal_length": values[17],
        "fx": values[18],
        "fy": values[19],
        "cx": values[20],
        "cy": values[21],
        "closed_form_distorts": bool(round(values[22])),
    }


def _write_holoassist_trajectory(
    destination: Path,
    sampled: list[tuple[float, int]],
    poses: list[tuple[float, int, list[float]]],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "output_index",
                "source_frame_index",
                "video_time_s",
                "filetime_ticks_100ns",
                *(f"m{row}{column}" for row in range(4) for column in range(4)),
            )
        )
        for output_index, (video_time, source_index) in enumerate(sampled):
            pose_time, filetime, matrix = poses[source_index]
            if abs(pose_time - video_time) > 1e-9:
                raise DatasetDownloadError("HoloAssist pose index/time mismatch")
            writer.writerow(
                (
                    output_index,
                    source_index,
                    f"{video_time:.12g}",
                    filetime,
                    *(f"{value:.12g}" for value in matrix),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def _holoassist_test_split(ctx: DownloadContext, dataset: dict[str, Any]) -> Path:
    cache = ctx.data_root / "_cache" / "holoassist" / "data-splits-v1_2.zip"
    archive_path = download_https(str(dataset["official_split_url"]), cache)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read("test-v1_2.txt")
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise DatasetDownloadError("Unable to read HoloAssist official test split") from error
    destination = ctx.data_root / "holoassist" / "_shared" / "test-v1_2.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(payload)
    os.replace(partial, destination)
    selected = {str(clip["sequence"]) for clip in dataset["clips"]}
    official = {
        line.strip()
        for line in payload.decode("utf-8-sig", "replace").splitlines()
        if line.strip()
    }
    missing = selected - official
    if missing:
        raise DatasetDownloadError(
            f"HoloAssist selected sequences are absent from test-v1_2: {sorted(missing)}"
        )
    return destination


def download_holoassist(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["holoassist"]
    root = ctx.data_root / "holoassist"
    index_root = ctx.data_root / "_cache" / "remote_tar" / "holoassist"
    expected_records: list[dict[str, Any] | None] = []
    for clip in dataset["clips"]:
        clip_dir = root / "clips" / str(clip["sequence"])
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        expected_records.append(
            _existing_strict_rgb_record(
                clip_dir, "holoassist", str(clip["sequence"]), expected
            )
        )
    if all(record is not None for record in expected_records):
        return [record for record in expected_records if record is not None]
    split_path = _holoassist_test_split(ctx, dataset)
    video_archive = RemoteTar(
        str(dataset["video_tar_url"]), HttpRangeClient(timeout_s=120, retries=8)
    )
    calibration_archive = RemoteTar(
        str(dataset["cam_info_tar_url"]), HttpRangeClient(timeout_s=120, retries=8)
    )
    if video_archive.remote.size != int(dataset["video_tar_bytes"]):
        raise DatasetDownloadError("HoloAssist video TAR size changed")
    if calibration_archive.remote.size != int(dataset["cam_info_tar_bytes"]):
        raise DatasetDownloadError("HoloAssist camera-info TAR size changed")
    sequences = [str(clip["sequence"]) for clip in dataset["clips"]]
    video_required = {
        f"{sequence}/Export_py/Video/{filename}"
        for sequence in sequences
        for filename in ("Pose_sync.txt", "VideoMp4Timing.txt")
    }
    video_required.update(
        f"{sequence}/Export_py/Video_compress.mp4"
        for sequence in sequences
    )
    calibration_required = {
        f"{sequence}/Export_py/Video/Intrinsics.txt"
        for sequence in sequences
    }
    video_entries = video_archive.index(
        index_root / "video_compress", required_names=video_required
    )
    calibration_entries = calibration_archive.index(
        index_root / "cam_info", required_names=calibration_required
    )
    records: list[dict[str, Any]] = []
    for clip, existing in zip(dataset["clips"], expected_records):
        sequence = str(clip["sequence"])
        print(f"[holoassist] {sequence}")
        if existing is not None:
            records.append(existing)
            continue
        clip_dir = root / "clips" / sequence
        pose_member = _tar_member_exact(
            video_entries, f"{sequence}/Export_py/Video/Pose_sync.txt"
        )
        timing_member = _tar_member_exact(
            video_entries, f"{sequence}/Export_py/Video/VideoMp4Timing.txt"
        )
        video_member = _tar_member_exact(
            video_entries, f"{sequence}/Export_py/Video_compress.mp4"
        )
        intrinsics_member = _tar_member_exact(
            calibration_entries, f"{sequence}/Export_py/Video/Intrinsics.txt"
        )
        reference_dir = clip_dir / "reference"
        pose_path = video_archive.extract(
            pose_member, reference_dir / "source_pose_sync.txt"
        )
        timing_path = video_archive.extract(
            timing_member, reference_dir / "video_mp4_timing.txt"
        )
        intrinsics_path = calibration_archive.extract(
            intrinsics_member, reference_dir / "source_intrinsics.txt"
        )
        poses = _parse_holoassist_poses(pose_path.read_bytes())
        timing = [
            int(line.strip())
            for line in timing_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if len(timing) != 2 or timing[0] != poses[0][1] or timing[1] < poses[-1][1]:
            raise DatasetDownloadError(f"HoloAssist MP4 timing gate failed: {sequence}")
        sampled_rows = _sample_rows(
            [(video_time, str(index)) for index, (video_time, _, _) in enumerate(poses)],
            float(clip["start_s"]),
            float(clip["duration_s"]),
            ctx.target_fps,
        )
        sampled = [(timestamp, int(index)) for timestamp, index in sampled_rows]
        source_video = (
            ctx.data_root
            / "_cache"
            / "holoassist"
            / sequence
            / "Video_compress.mp4"
        )
        video_archive.extract(video_member, source_video)
        success = False
        try:
            images = _extract_indexed_video_frames(
                source_video, clip_dir, sampled, ctx.ffmpeg, ctx.data_root
            )
            width, height = _validate_rgb_images(images)
            source_calibration = _parse_holoassist_intrinsics(
                intrinsics_path.read_bytes()
            )
            scale_x = width / int(source_calibration["source_width"])
            scale_y = height / int(source_calibration["source_height"])
            trajectory = _write_holoassist_trajectory(
                reference_dir / "camera_to_hololens_world.csv",
                sampled,
                poses,
            )
            camera = {
                **dataset["camera"],
                "distortion_model": "Brown-Conrady",
                "fx": float(source_calibration["fx"]) * scale_x,
                "fy": float(source_calibration["fy"]) * scale_y,
                "cx": float(source_calibration["cx"]) * scale_x,
                "cy": float(source_calibration["cy"]) * scale_y,
                "radial_distortion": source_calibration["radial_distortion"],
                "tangential_distortion": source_calibration["tangential_distortion"],
                "closed_form_distorts": source_calibration["closed_form_distorts"],
                "source_calibration_width": source_calibration["source_width"],
                "source_calibration_height": source_calibration["source_height"],
                "intrinsics_scale_x": scale_x,
                "intrinsics_scale_y": scale_y,
                "pose_direction": "camera_to_hololens_world",
                "exporter_reference_commit": "dd8458dba7bc82015a5b41e67c865a56b3c0eb58",
            }
            record = _strict_rgb_record(
                "holoassist",
                dataset,
                clip,
                clip_dir,
                sampled_rows,
                images,
                [
                    pose_path,
                    timing_path,
                    intrinsics_path,
                    trajectory,
                    split_path,
                ],
                camera,
                ctx.target_fps,
                [
                    "native HoloLens front-facing color video; official compressed width is 256",
                    "frames are selected by Pose_sync row index at the requested video time",
                    "intrinsics are scaled from the official calibration resolution to decoded RGB",
                    "CameraPose is camera-to-world per the official psi CameraView projection code",
                    "device tracking is not independent external motion capture",
                    "ReViV pretraining used HoloAssist, so report this as pretrained-domain evaluation",
                ],
            )
            record["source"] = {
                "video_tar": {
                    "url": video_archive.remote.url,
                    "bytes": video_archive.remote.size,
                    "etag": video_archive.remote.etag,
                    "member": video_member.name,
                    "member_offset": video_member.data_offset,
                    "member_bytes": video_member.size,
                    "member_sha256": _sha256(source_video),
                },
                "cam_info_tar": {
                    "url": calibration_archive.remote.url,
                    "bytes": calibration_archive.remote.size,
                    "etag": calibration_archive.remote.etag,
                    "member": intrinsics_member.name,
                },
                "official_split_revision": dataset["official_split_revision"],
                "retained_full_sequence_video": bool(ctx.keep_source),
            }
            record["sampling"] = {
                "basis": "Pose_sync_video_time_then_source_frame_index",
                "first_source_frame": sampled[0][1],
                "last_source_frame": sampled[-1][1],
                "sampled_frames": len(sampled),
            }
            record["evaluation_role"] = "pretrained_domain_device_reference"
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
            success = True
        finally:
            if success and not ctx.keep_source:
                source_video.unlink(missing_ok=True)
    return records


def _sample_stera_indices(
    source_frames: int,
    start_s: float,
    duration_s: float,
    source_fps: int,
    target_fps: int,
) -> list[tuple[float, int]]:
    if source_frames < 2 or source_fps <= 0 or target_fps <= 0:
        raise DatasetDownloadError("Invalid Stera source video metadata")
    count = int(round(duration_s * target_fps))
    sampled: list[tuple[float, int]] = []
    for output_index in range(count):
        target_s = start_s + output_index / target_fps
        source_index = int(math.floor(target_s * source_fps + 0.5))
        if source_index >= source_frames:
            raise DatasetDownloadError(
                "Stera window exceeds the source video: "
                f"frame {source_index}/{source_frames}"
            )
        sampled.append((source_index / source_fps, source_index))
    if any(
        right[1] <= left[1] for left, right in zip(sampled, sampled[1:])
    ):
        raise DatasetDownloadError(
            "Stera source FPS must produce unique increasing evaluation frames"
        )
    return sampled


def _stera_camera_to_world(
    rotation_link_to_world: Any,
    translation_world: Any,
    rotation_optical_to_link: Any,
):
    import numpy as np

    rotation = np.asarray(rotation_link_to_world, dtype=np.float64)
    translation = np.asarray(translation_world, dtype=np.float64)
    optical_to_link = np.asarray(rotation_optical_to_link, dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or optical_to_link.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(translation))
        or not np.all(np.isfinite(optical_to_link))
    ):
        raise DatasetDownloadError("Invalid Stera camera pose or optical transform")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ optical_to_link
    result[:3, 3] = translation
    if (
        not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5)
        or abs(float(np.linalg.det(result[:3, :3])) - 1.0) > 1e-5
    ):
        raise DatasetDownloadError("Stera camera rotation is not in SO(3)")
    return result


def _stera_source_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DatasetDownloadError(f"Unsafe Stera source path: {relative!r}")
    return path


def _download_stera_file(
    dataset: dict[str, Any],
    sequence: str,
    relative: str,
    spec: dict[str, Any],
    destination: Path,
) -> Path:
    expected_size = int(spec["bytes"])
    expected_sha256 = str(spec["sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise DatasetDownloadError(
            f"Invalid Stera SHA-256 declaration: {sequence}/{relative}"
        )
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and _sha256(destination) == expected_sha256
    ):
        return destination
    destination.unlink(missing_ok=True)
    try:
        from huggingface_hub import get_token
    except ImportError as error:
        raise DatasetDownloadError(
            "Stera download requires the existing huggingface_hub client"
        ) from error
    token = get_token()
    if not token:
        raise DatasetDownloadError(
            "Stera access is gated; log in with `hf auth login` after approval"
        )
    source_path = _stera_source_path(f"{sequence}/{relative}")
    url = (
        "https://huggingface.co/datasets/"
        f"{dataset['repository']}/resolve/{dataset['revision']}/"
        f"{quote(source_path.as_posix(), safe='/')}?download=true"
    )
    for _ in range(2):
        result = download_https(
            url,
            destination,
            timeout_s=120,
            retries=6,
            request_headers={"Authorization": f"Bearer {token}"},
            trust_env=False,
        )
        if result.stat().st_size == expected_size and _sha256(result) == expected_sha256:
            return result
        result.unlink(missing_ok=True)
    raise DatasetDownloadError(
        f"Stera source size/hash mismatch: {sequence}/{relative}"
    )


def _probe_stera_video(
    path: Path,
    ffmpeg: str,
    expected_frames: int,
    source_fps: int,
    width: int,
    height: int,
) -> float:
    sibling = Path(ffmpeg).resolve().with_name("ffprobe")
    ffprobe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
    if not ffprobe:
        raise DatasetDownloadError("ffprobe is required for Stera source validation")
    payload = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise DatasetDownloadError(f"Stera MP4 has {len(streams)} video streams")
    stream = streams[0]
    numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
    rate = float(numerator) / float(denominator)
    if (
        int(stream.get("nb_frames", -1)) != expected_frames
        or int(stream.get("width", -1)) != width
        or int(stream.get("height", -1)) != height
        or abs(rate - source_fps) > 1e-6
    ):
        raise DatasetDownloadError("Stera MP4 metadata does not match the frozen plan")
    return float(payload.get("format", {}).get("duration", 0))


def _nearest_stera_tracking_states(
    pose_timestamps: Any,
    sampled: list[tuple[float, int]],
    tracking_timestamps: Any,
    tracking_states: Any,
) -> list[str]:
    timestamps = [float(value) for value in tracking_timestamps]
    if len(timestamps) < 2 or any(
        right < left for left, right in zip(timestamps, timestamps[1:])
    ):
        raise DatasetDownloadError("Stera tracking-state timestamps are invalid")
    result: list[str] = []
    for _, source_index in sampled:
        target = float(pose_timestamps[source_index])
        position = bisect.bisect_left(timestamps, target)
        candidates = [min(position, len(timestamps) - 1)]
        if position:
            candidates.append(position - 1)
        selected = min(candidates, key=lambda index: abs(timestamps[index] - target))
        value = tracking_states[selected]
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", "replace"))
        else:
            result.append(str(value))
    return result


def _write_stera_trajectory(
    destination: Path,
    sampled: list[tuple[float, int]],
    pose_timestamps: Any,
    rotations: Any,
    translations: Any,
    optical_to_link: Any,
    tracking_states: list[str],
    target_fps: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    origin = float(pose_timestamps[0])
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "output_index",
                "source_frame_index",
                "target_clip_time_s",
                "mp4_time_s",
                "arkit_timestamp_s",
                "arkit_elapsed_s",
                "tracking_state",
                *(f"m{row}{column}" for row in range(4) for column in range(4)),
            )
        )
        for output_index, ((mp4_time, source_index), state) in enumerate(
            zip(sampled, tracking_states)
        ):
            matrix = _stera_camera_to_world(
                rotations[source_index],
                translations[source_index],
                optical_to_link,
            )
            timestamp = float(pose_timestamps[source_index])
            writer.writerow(
                (
                    output_index,
                    f"{source_index}",
                    f"{output_index / target_fps:.9f}",
                    f"{mp4_time:.9f}",
                    f"{timestamp:.12g}",
                    f"{timestamp - origin:.12g}",
                    state,
                    *(f"{value:.12g}" for value in matrix.reshape(-1)),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def _copy_stera_reference(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    shutil.copyfile(source, partial)
    os.replace(partial, destination)
    return destination


def download_stera10m(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["stera10m"]
    root = ctx.data_root / "stera10m"
    cache_root = ctx.data_root / "_cache" / "stera"
    expected_records: list[dict[str, Any] | None] = []
    for clip in dataset["clips"]:
        clip_dir = root / "clips" / str(clip["sequence"])
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        expected_records.append(
            _existing_strict_rgb_record(
                clip_dir, "stera10m", str(clip["sequence"]), expected
            )
        )
    if all(record is not None for record in expected_records):
        if not ctx.keep_source and cache_root.exists():
            _remove_owned_eval_cache(cache_root, ctx.data_root)
        return [record for record in expected_records if record is not None]
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise DatasetDownloadError(
            "Stera subset extraction requires h5py and numpy in the selected environment"
        ) from error

    required_source_files = {
        "annotation.hdf5",
        "hierarchy.json",
        "rgb.mp4",
        "calibrations/meta.json",
        "calibrations/rgb_K.npy",
        "calibrations/rgb_D.npy",
        "calibrations/R_optical_to_link.npy",
    }
    records: list[dict[str, Any]] = []
    for clip, existing in zip(dataset["clips"], expected_records):
        sequence = str(clip["sequence"])
        print(f"[stera10m] {sequence}")
        if existing is not None:
            records.append(existing)
            continue
        source_specs = clip["source_files"]
        if set(source_specs) != required_source_files:
            raise DatasetDownloadError(
                f"Stera source file declaration changed: {sequence}"
            )
        source_dir = cache_root / "source" / sequence
        source_paths = {
            relative: _download_stera_file(
                dataset,
                sequence,
                relative,
                spec,
                source_dir / _stera_source_path(relative),
            )
            for relative, spec in source_specs.items()
        }
        success = False
        try:
            hierarchy = json.loads(
                source_paths["hierarchy.json"].read_text(encoding="utf-8")
            )
            if (
                hierarchy.get("session_id") != sequence
                or float(hierarchy.get("total_duration_s", -1)) + 1e-6
                < float(clip["start_s"]) + float(clip["duration_s"])
            ):
                raise DatasetDownloadError(
                    f"Stera hierarchy/window gate failed: {sequence}"
                )
            calibration_meta = json.loads(
                source_paths["calibrations/meta.json"].read_text(encoding="utf-8")
            )
            rgb_meta = calibration_meta.get("rgb", {})
            if (
                rgb_meta.get("distortion_model") != "plumb_bob"
                or int(rgb_meta.get("width", -1)) != int(dataset["camera"]["source_width"])
                or int(rgb_meta.get("height", -1)) != int(dataset["camera"]["source_height"])
            ):
                raise DatasetDownloadError(
                    f"Stera RGB calibration gate failed: {sequence}"
                )
            source_frames = int(clip["source_frames"])
            source_fps = int(dataset["source_fps"])
            duration = _probe_stera_video(
                source_paths["rgb.mp4"],
                ctx.ffmpeg,
                source_frames,
                source_fps,
                int(dataset["camera"]["source_width"]),
                int(dataset["camera"]["source_height"]),
            )
            if duration + 0.1 < float(clip["start_s"]) + float(clip["duration_s"]):
                raise DatasetDownloadError(f"Stera MP4 is too short: {sequence}")
            sampled = _sample_stera_indices(
                source_frames,
                float(clip["start_s"]),
                float(clip["duration_s"]),
                source_fps,
                ctx.target_fps,
            )
            with h5py.File(source_paths["annotation.hdf5"], "r") as archive:
                required_h5 = {
                    "cam-pose/timestamps",
                    "cam-pose/rotations",
                    "cam-pose/translations",
                    "tracking-state/timestamps",
                    "tracking-state/state_str",
                }
                if not all(name in archive for name in required_h5):
                    raise DatasetDownloadError(
                        f"Stera HDF5 lacks camera/tracking datasets: {sequence}"
                    )
                pose_timestamps = archive["cam-pose/timestamps"][:]
                rotations = archive["cam-pose/rotations"][:]
                translations = archive["cam-pose/translations"][:]
                if (
                    pose_timestamps.shape != (source_frames,)
                    or rotations.shape != (source_frames, 3, 3)
                    or translations.shape != (source_frames, 3)
                    or int(archive["metadata"].attrs.get("num_rgb_frames", -1))
                    != source_frames
                    or int(archive["metadata"].attrs.get("num_pose_samples", -1))
                    != source_frames
                    or not np.all(np.isfinite(pose_timestamps))
                    or not np.all(np.isfinite(rotations))
                    or not np.all(np.isfinite(translations))
                    or np.any(np.diff(pose_timestamps) <= 0)
                ):
                    raise DatasetDownloadError(
                        f"Stera RGB/pose alignment gate failed: {sequence}"
                    )
                selected_timestamps = np.asarray(
                    [pose_timestamps[index] for _, index in sampled],
                    dtype=np.float64,
                )
                maximum_pose_gap = float(np.max(np.diff(selected_timestamps)))
                if maximum_pose_gap > float(dataset["maximum_pose_gap_s"]):
                    raise DatasetDownloadError(
                        f"Stera selected window crosses a pose gap: {sequence} "
                        f"({maximum_pose_gap:.3f}s)"
                    )
                tracking_states = _nearest_stera_tracking_states(
                    pose_timestamps,
                    sampled,
                    archive["tracking-state/timestamps"][:],
                    archive["tracking-state/state_str"][:],
                )
            expected_tracking_state = str(dataset["required_tracking_state"])
            if set(tracking_states) != {expected_tracking_state}:
                raise DatasetDownloadError(
                    f"Stera tracking state gate failed: {sequence} "
                    f"({sorted(set(tracking_states))})"
                )
            intrinsics = np.load(
                source_paths["calibrations/rgb_K.npy"], allow_pickle=False
            )
            distortion = np.load(
                source_paths["calibrations/rgb_D.npy"], allow_pickle=False
            )
            optical_to_link = np.load(
                source_paths["calibrations/R_optical_to_link.npy"],
                allow_pickle=False,
            )
            if (
                intrinsics.shape != (3, 3)
                or distortion.ndim != 1
                or optical_to_link.shape != (3, 3)
                or not np.all(np.isfinite(intrinsics))
                or not np.all(np.isfinite(distortion))
            ):
                raise DatasetDownloadError(
                    f"Stera calibration arrays are invalid: {sequence}"
                )
            clip_dir = root / "clips" / sequence
            images = _extract_indexed_video_frames(
                source_paths["rgb.mp4"],
                clip_dir,
                sampled,
                ctx.ffmpeg,
                ctx.data_root,
            )
            reference_dir = clip_dir / "reference"
            reference_files = [
                _copy_stera_reference(
                    source_paths["hierarchy.json"],
                    reference_dir / "hierarchy.json",
                ),
                _copy_stera_reference(
                    source_paths["calibrations/meta.json"],
                    reference_dir / "source_calibration_meta.json",
                ),
                _copy_stera_reference(
                    source_paths["calibrations/rgb_K.npy"],
                    reference_dir / "rgb_K.npy",
                ),
                _copy_stera_reference(
                    source_paths["calibrations/rgb_D.npy"],
                    reference_dir / "rgb_D.npy",
                ),
                _copy_stera_reference(
                    source_paths["calibrations/R_optical_to_link.npy"],
                    reference_dir / "R_optical_to_link.npy",
                ),
            ]
            trajectory = _write_stera_trajectory(
                reference_dir / "camera_optical_to_arkit_world.csv",
                sampled,
                pose_timestamps,
                rotations,
                translations,
                optical_to_link,
                tracking_states,
                ctx.target_fps,
            )
            reference_files.append(trajectory)
            camera = {
                **dataset["camera"],
                "fx": float(intrinsics[0, 0]),
                "fy": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2]),
                "intrinsics_matrix": intrinsics.tolist(),
                "distortion_coefficients": distortion.tolist(),
                "rotation_optical_to_link": optical_to_link.tolist(),
                "pose_direction": "camera_optical_to_arkit_world",
                "sdk_commit": dataset["sdk_commit"],
                "dataset_revision": dataset["revision"],
            }
            rows = [(time_s, str(index)) for time_s, index in sampled]
            record = _strict_rgb_record(
                "stera10m",
                dataset,
                clip,
                clip_dir,
                rows,
                images,
                reference_files,
                camera,
                ctx.target_fps,
                [
                    "native 1280x720 head-mounted iPhone RGB with plumb_bob calibration",
                    "MP4 frames and HDF5 camera poses are one-to-one by source frame index",
                    "camera-link ARKit pose is composed with R_optical_to_link for RGB optical pose",
                    "selected windows require normal ARKit tracking and reject pose gaps above 250 ms",
                    "ARKit device tracking is not independent external motion capture",
                ],
            )
            record["source"] = {
                "repository": dataset["repository"],
                "revision": dataset["revision"],
                "license": dataset["license"],
                "source_files": source_specs,
                "retained_full_session_sources": bool(ctx.keep_source),
            }
            record["sampling"] = {
                "basis": "constant_15fps_mp4_frame_index_then_same_index_arkit_pose",
                "first_source_frame": sampled[0][1],
                "last_source_frame": sampled[-1][1],
                "sampled_frames": len(sampled),
                "maximum_selected_pose_gap_ms": maximum_pose_gap * 1000.0,
            }
            record["tracking"] = {
                "required_state": expected_tracking_state,
                "sampled_state_counts": {
                    expected_tracking_state: len(tracking_states)
                },
            }
            record["evaluation_role"] = "gated_device_reference"
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
            success = True
        finally:
            if success and not ctx.keep_source and source_dir.exists():
                _remove_owned_eval_cache(source_dir, ctx.data_root)
    if not ctx.keep_source and len(records) == len(dataset["clips"]):
        if cache_root.exists():
            _remove_owned_eval_cache(cache_root, ctx.data_root)
    return records


def _rh20t_scene_members(
    dataset: dict[str, Any], *, include_robot_exo: bool = False
) -> set[str]:
    archive_root = str(dataset["archive_root"])
    members: set[str] = set()
    for clip in dataset["clips"]:
        sequence = str(clip["sequence"])
        base = f"{archive_root}/{sequence}"
        members.update(
            {
                f"{base}/metadata.json",
                f"{base}/cam_{dataset['in_hand_serial']}/color.mp4",
                f"{base}/cam_{dataset['in_hand_serial']}/timestamps.npy",
                f"{base}/transformed/tcp_base.npy",
            }
        )
        if include_robot_exo:
            serial = str(dataset["demo_exo"]["serial"])
            members.update(
                {
                    f"{base}/cam_{serial}/color.mp4",
                    f"{base}/cam_{serial}/timestamps.npy",
                }
            )
    return members


def _rh20t_stage_is_complete(
    stage_root: Path,
    dataset: dict[str, Any],
    *,
    include_robot_exo: bool = False,
) -> bool:
    for name in _rh20t_scene_members(
        dataset, include_robot_exo=include_robot_exo
    ):
        path = stage_root / PurePosixPath(name)
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    archive_root = stage_root / str(dataset["archive_root"])
    try:
        calibration_ids = {
            str(
                json.loads(
                    (archive_root / str(clip["sequence"]) / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )["calib"]
            )
            for clip in dataset["clips"]
        }
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return all(
        (archive_root / "calib" / calibration_id / filename).is_file()
        for calibration_id in calibration_ids
        for filename in ("devices.npy", "extrinsics.npy", "intrinsics.npy", "tcp.npy")
    )


def _stage_rh20t_archive(
    archive_path: Path,
    stage_root: Path,
    dataset: dict[str, Any],
    data_root: Path,
    *,
    include_robot_exo: bool = False,
) -> Path:
    if _rh20t_stage_is_complete(
        stage_root, dataset, include_robot_exo=include_robot_exo
    ):
        return stage_root / str(dataset["archive_root"])
    if stage_root.exists():
        _remove_owned_eval_cache(stage_root, data_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    archive_root = str(dataset["archive_root"])
    required = _rh20t_scene_members(
        dataset, include_robot_exo=include_robot_exo
    )
    found: set[str] = set()
    calibration_pattern = re.compile(
        rf"^{re.escape(archive_root)}/calib/[^/]+/"
        r"(?:devices|extrinsics|intrinsics|tcp)\.npy$"
    )
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                name = _normalized_tar_name(member.name)
                if name not in required and not calibration_pattern.fullmatch(name):
                    continue
                if name in found:
                    raise DatasetDownloadError(f"Duplicate RH20T TAR member: {name}")
                if not member.isfile() or member.size <= 0:
                    raise DatasetDownloadError(
                        f"RH20T member is not a non-empty regular file: {name}"
                    )
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise DatasetDownloadError(f"Unsafe RH20T TAR member: {name}")
                source = archive.extractfile(member)
                if source is None:
                    raise DatasetDownloadError(f"Unable to extract RH20T member: {name}")
                destination = stage_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + ".part")
                with source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if partial.stat().st_size != member.size:
                    partial.unlink(missing_ok=True)
                    raise DatasetDownloadError(f"Short RH20T TAR member: {name}")
                os.replace(partial, destination)
                found.add(name)
    except (OSError, tarfile.TarError) as error:
        raise DatasetDownloadError("Unable to stream the RH20T cfg3 archive") from error
    missing = required - found
    if missing:
        raise DatasetDownloadError(
            f"RH20T archive lacks selected scene members: {sorted(missing)}"
        )
    if not _rh20t_stage_is_complete(
        stage_root, dataset, include_robot_exo=include_robot_exo
    ):
        raise DatasetDownloadError(
            "RH20T selected scenes reference missing calibration arrays"
        )
    return stage_root / archive_root


def _rh20t_load_dict(path: Path, label: str) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise DatasetDownloadError(
            "RH20T subset extraction requires numpy in the selected environment"
        ) from error
    try:
        value = np.load(path, allow_pickle=True)
        payload = value.item()
    except (OSError, TypeError, ValueError) as error:
        raise DatasetDownloadError(f"Unable to read RH20T {label}: {path}") from error
    if not isinstance(payload, dict):
        raise DatasetDownloadError(f"RH20T {label} is not a dictionary: {path}")
    return payload


def _rh20t_pose_matrix(pose: Iterable[float]):
    import numpy as np

    values = np.asarray(list(pose), dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise DatasetDownloadError("RH20T TCP pose must be finite xyz+quaternion-wxyz")
    quaternion = values[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise DatasetDownloadError("RH20T TCP quaternion has zero norm")
    w, x, y, z = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = values[:3]
    return matrix


def _rh20t_camera_to_aligned_base(
    aligned_tcp_pose: Iterable[float], camera: dict[str, Any]
):
    import numpy as np

    tcp_camera = np.asarray(camera["tcp_camera_matrix"], dtype=np.float64)
    align_tcp = np.asarray(camera["align_tcp_matrix"], dtype=np.float64)
    if tcp_camera.shape != (4, 4) or align_tcp.shape != (4, 4):
        raise DatasetDownloadError("RH20T hand-eye matrices must be 4x4")
    result = (
        _rh20t_pose_matrix(aligned_tcp_pose)
        @ np.linalg.inv(align_tcp)
        @ np.linalg.inv(tcp_camera)
    )
    if not np.allclose(result[3], (0, 0, 0, 1), atol=1e-9):
        raise DatasetDownloadError("RH20T camera pose is not homogeneous")
    return result


def _sample_rh20t_indices(
    timestamps_ms: Iterable[int], start_s: float, duration_s: float, fps: int
) -> list[tuple[int, int]]:
    values = [int(value) for value in timestamps_ms]
    if len(values) < 2 or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise DatasetDownloadError(
            "RH20T color timestamps are missing or not strictly increasing"
        )
    origin = values[0]
    end_s = (values[-1] - origin) / 1000.0
    if end_s + 1e-6 < start_s + duration_s:
        raise DatasetDownloadError(
            f"RH20T sequence has only {end_s:.2f}s, requested end is "
            f"{start_s + duration_s:.2f}s"
        )
    sampled: list[tuple[int, int]] = []
    maximum_error_ms = 0.0
    count = int(round(duration_s * fps))
    for output_index in range(count):
        target_ms = origin + 1000.0 * (start_s + output_index / fps)
        position = bisect.bisect_left(values, target_ms)
        candidates = [min(position, len(values) - 1)]
        if position:
            candidates.append(position - 1)
        source_index = min(
            candidates, key=lambda index: abs(values[index] - target_ms)
        )
        maximum_error_ms = max(
            maximum_error_ms, abs(values[source_index] - target_ms)
        )
        sampled.append((values[source_index], source_index))
    if maximum_error_ms > 250:
        raise DatasetDownloadError(
            f"RH20T temporal resampling error is too large: {maximum_error_ms:.1f}ms"
        )
    return sampled


def _rh20t_tcp_rows(path: Path, serial: str) -> list[tuple[int, list[float]]]:
    payload = _rh20t_load_dict(path, "aligned TCP reference")
    if serial in payload:
        groups = [payload[serial]]
    elif "base" in payload:
        groups = [payload["base"]]
    else:
        groups = list(payload.values())
    by_timestamp: dict[int, list[float]] = {}
    try:
        for group in groups:
            for item in group:
                timestamp = int(item["timestamp"])
                pose = [float(value) for value in item["tcp"]]
                if len(pose) != 7:
                    raise ValueError("unexpected TCP width")
                previous = by_timestamp.get(timestamp)
                if previous is not None and any(
                    abs(left - right) > 1e-9 for left, right in zip(previous, pose)
                ):
                    raise ValueError("conflicting duplicate TCP timestamp")
                by_timestamp[timestamp] = pose
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetDownloadError(f"Malformed RH20T TCP reference: {path}") from error
    rows = sorted(by_timestamp.items())
    if len(rows) < 2:
        raise DatasetDownloadError(f"RH20T TCP reference is empty: {path}")
    return rows


def _interpolate_rh20t_tcp(
    rows: list[tuple[int, list[float]]], timestamp: int
) -> tuple[list[float], bool]:
    import numpy as np

    times = [row[0] for row in rows]
    position = bisect.bisect_left(times, timestamp)
    if position < len(rows) and rows[position][0] == timestamp:
        return list(rows[position][1]), True
    if position == 0 or position == len(rows):
        raise DatasetDownloadError(
            f"RH20T TCP reference does not bracket color timestamp {timestamp}"
        )
    left_time, left_pose = rows[position - 1]
    right_time, right_pose = rows[position]
    left = np.asarray(left_pose, dtype=np.float64)
    right = np.asarray(right_pose, dtype=np.float64)
    if float(np.dot(left[3:], right[3:])) < 0:
        right[3:] *= -1
    ratio = (timestamp - left_time) / (right_time - left_time)
    pose = left + ratio * (right - left)
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if quaternion_norm < 1e-12:
        raise DatasetDownloadError("RH20T interpolated TCP quaternion has zero norm")
    pose[3:] /= quaternion_norm
    return pose.tolist(), False


def _rh20t_reference_samples(
    sampled: list[tuple[int, int]],
    tcp_rows: list[tuple[int, list[float]]],
    camera: dict[str, Any],
) -> list[tuple[int, int, list[float], Any, bool]]:
    result = []
    for timestamp, source_index in sampled:
        pose, exact = _interpolate_rh20t_tcp(tcp_rows, timestamp)
        result.append(
            (
                timestamp,
                source_index,
                pose,
                _rh20t_camera_to_aligned_base(pose, camera),
                exact,
            )
        )
    return result


def _write_rh20t_trajectory(
    destination: Path,
    samples: list[tuple[int, int, list[float], Any, bool]],
    target_fps: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    origin_ms = samples[0][0]
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "output_index",
                "source_frame_index",
                "timestamp_ms",
                "target_clip_time_s",
                "source_clip_time_s",
                "tcp_exact_timestamp",
                "tcp_tx",
                "tcp_ty",
                "tcp_tz",
                "tcp_qw",
                "tcp_qx",
                "tcp_qy",
                "tcp_qz",
                *(f"m{row}{column}" for row in range(4) for column in range(4)),
            )
        )
        for output_index, (timestamp, source_index, pose, matrix, exact) in enumerate(
            samples
        ):
            writer.writerow(
                (
                    output_index,
                    source_index,
                    timestamp,
                    f"{output_index / target_fps:.9f}",
                    f"{(timestamp - origin_ms) / 1000.0:.9f}",
                    int(exact),
                    *(f"{value:.12g}" for value in pose),
                    *(f"{float(value):.12g}" for value in matrix.reshape(-1)),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def _atomic_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with source.open("rb") as input_handle, partial.open("wb") as output:
        shutil.copyfileobj(input_handle, output, 4 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)
    return destination


def _rh20t_calibration(
    calibration_dir: Path, serial: str, camera: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    devices = np.load(calibration_dir / "devices.npy", allow_pickle=False)
    if serial not in {str(value) for value in devices.tolist()}:
        raise DatasetDownloadError(f"RH20T calibration omits wrist camera {serial}")
    intrinsics = _rh20t_load_dict(
        calibration_dir / "intrinsics.npy", "camera intrinsics"
    )
    extrinsics = _rh20t_load_dict(
        calibration_dir / "extrinsics.npy", "camera extrinsics"
    )
    try:
        intrinsic = np.asarray(intrinsics[serial], dtype=np.float64)
        extrinsic = np.asarray(extrinsics[serial], dtype=np.float64).squeeze()
        calibration_tcp = np.asarray(
            np.load(calibration_dir / "tcp.npy", allow_pickle=False),
            dtype=np.float64,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise DatasetDownloadError(
            f"Malformed RH20T calibration for wrist camera {serial}"
        ) from error
    if intrinsic.shape != (3, 4) or extrinsic.shape != (4, 4):
        raise DatasetDownloadError("Unexpected RH20T intrinsic/extrinsic shape")
    tcp_camera = np.asarray(camera["tcp_camera_matrix"], dtype=np.float64)
    align_base = np.asarray(camera["align_base_matrix"], dtype=np.float64)
    raw_tcp_matrix = _rh20t_pose_matrix(calibration_tcp)
    base_world = (
        np.linalg.inv(extrinsic)
        @ tcp_camera
        @ np.linalg.inv(raw_tcp_matrix)
    )
    official_aligned_extrinsic = extrinsic @ base_world @ align_base
    official_camera_pose = np.linalg.inv(official_aligned_extrinsic)
    direct_camera_pose = (
        np.linalg.inv(align_base)
        @ raw_tcp_matrix
        @ np.linalg.inv(tcp_camera)
    )
    residual = float(np.max(np.abs(official_camera_pose - direct_camera_pose)))
    if residual > 1e-8:
        raise DatasetDownloadError(
            f"RH20T hand-eye direction gate failed with residual {residual:.3g}"
        )
    return {
        "source_intrinsics_matrix_3x4": intrinsic.tolist(),
        "source_fx": float(intrinsic[0, 0]),
        "source_fy": float(intrinsic[1, 1]),
        "source_cx": float(intrinsic[0, 2]),
        "source_cy": float(intrinsic[1, 2]),
        "source_skew": float(intrinsic[0, 1]),
        "calibration_direction_residual": residual,
    }


def download_rh20t_wrist(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["rh20t_wrist"]
    root = ctx.data_root / "rh20t_wrist"
    cache_root = ctx.data_root / "_cache" / "rh20t"
    stage_root = cache_root / "extracted"
    expected_records: list[dict[str, Any] | None] = []
    for clip in dataset["clips"]:
        clip_dir = root / "clips" / str(clip["sequence"])
        expected = int(round(float(clip["duration_s"]) * ctx.target_fps))
        expected_records.append(
            _existing_strict_rgb_record(
                clip_dir, "rh20t_wrist", str(clip["sequence"]), expected
            )
        )
    if all(record is not None for record in expected_records):
        return [record for record in expected_records if record is not None]

    archive_path: Path | None = None
    if not _rh20t_stage_is_complete(
        stage_root, dataset, include_robot_exo=ctx.robot_with_exo
    ):
        if ctx.rh20t_archive is not None:
            archive_path = ctx.rh20t_archive
            if not archive_path.is_file():
                raise DatasetDownloadError(
                    f"RH20T archive does not exist: {archive_path}"
                )
        else:
            archive_path = download_google_drive_ranges(
                str(dataset["google_drive_id"]),
                cache_root / str(dataset["archive_name"]),
                int(dataset["archive_bytes"]),
                workers=ctx.workers,
                mirror_url=str(dataset["mirror_url"]),
                expected_sha256=str(dataset["archive_sha256"]),
            )
        if archive_path.stat().st_size != int(dataset["archive_bytes"]):
            raise DatasetDownloadError("RH20T cfg3 archive size mismatch")
        actual_archive_sha256 = _sha256(archive_path)
        if actual_archive_sha256 != str(dataset["archive_sha256"]):
            raise DatasetDownloadError(
                f"RH20T cfg3 archive SHA-256 mismatch: {actual_archive_sha256}"
            )
        archive_root = _stage_rh20t_archive(
            archive_path,
            stage_root,
            dataset,
            ctx.data_root,
            include_robot_exo=ctx.robot_with_exo,
        )
    else:
        archive_root = stage_root / str(dataset["archive_root"])

    serial = str(dataset["in_hand_serial"])
    records: list[dict[str, Any]] = []
    completed = False
    try:
        for clip, existing in zip(dataset["clips"], expected_records):
            sequence = str(clip["sequence"])
            print(f"[rh20t_wrist] {sequence}")
            if existing is not None:
                records.append(existing)
                continue
            source_dir = archive_root / sequence
            video_path = source_dir / f"cam_{serial}" / "color.mp4"
            timestamps_path = source_dir / f"cam_{serial}" / "timestamps.npy"
            tcp_path = source_dir / "transformed" / "tcp_base.npy"
            metadata_path = source_dir / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                calibration_id = str(metadata["calib"])
            except (KeyError, OSError, TypeError, ValueError) as error:
                raise DatasetDownloadError(
                    f"Malformed RH20T metadata: {sequence}"
                ) from error
            calibration_dir = archive_root / "calib" / calibration_id
            timestamp_payload = _rh20t_load_dict(
                timestamps_path, "color timestamps"
            )
            try:
                timestamps = [int(value) for value in timestamp_payload["color"]]
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetDownloadError(
                    f"Malformed RH20T color timestamps: {sequence}"
                ) from error
            sampled = _sample_rh20t_indices(
                timestamps,
                float(clip["start_s"]),
                float(clip["duration_s"]),
                ctx.target_fps,
            )
            tcp_rows = _rh20t_tcp_rows(tcp_path, serial)
            samples = _rh20t_reference_samples(
                sampled, tcp_rows, dataset["camera"]
            )
            clip_dir = root / "clips" / sequence
            images = _extract_indexed_video_frames(
                video_path, clip_dir, sampled, ctx.ffmpeg, ctx.data_root
            )
            width, height = _validate_rgb_images(images)
            calibration = _rh20t_calibration(
                calibration_dir, serial, dataset["camera"]
            )
            source_width, source_height = [
                int(value) for value in dataset["camera"]["source_resolution"]
            ]
            scale_x = width / source_width
            scale_y = height / source_height
            reference_dir = clip_dir / "reference"
            reference_metadata = _atomic_copy(
                metadata_path, reference_dir / "metadata.json"
            )
            calibration_references = [
                _atomic_copy(
                    calibration_dir / filename,
                    reference_dir / "calibration" / filename,
                )
                for filename in (
                    "devices.npy",
                    "extrinsics.npy",
                    "intrinsics.npy",
                    "tcp.npy",
                )
            ]
            trajectory = _write_rh20t_trajectory(
                reference_dir / "camera_to_aligned_robot_base.csv",
                samples,
                ctx.target_fps,
            )
            camera = {
                **dataset["camera"],
                **calibration,
                "fx": calibration["source_fx"] * scale_x,
                "fy": calibration["source_fy"] * scale_y,
                "cx": calibration["source_cx"] * scale_x,
                "cy": calibration["source_cy"] * scale_y,
                "skew": calibration["source_skew"] * scale_x,
                "intrinsics_scale_x": scale_x,
                "intrinsics_scale_y": scale_y,
                "calibration_id": calibration_id,
                "api_repo": dataset["api_repo"],
                "api_commit": dataset["api_commit"],
                "pose_formula": (
                    "T_aligned_base_camera = T_aligned_base_aligned_tcp "
                    "@ inv(align_tcp_matrix) @ inv(tcp_camera_matrix)"
                ),
            }
            frame_rows = [
                (float(timestamp), str(source_index))
                for timestamp, source_index in sampled
            ]
            record = _strict_rgb_record(
                "rh20t_wrist",
                dataset,
                clip,
                clip_dir,
                frame_rows,
                images,
                [reference_metadata, *calibration_references, trajectory],
                camera,
                ctx.target_fps,
                [
                    "native released RGB from the robot in-hand RealSense color stream",
                    "released video is 640x360; official 1280x720 intrinsics are scaled by 0.5",
                    "frames are sampled by the official non-uniform millisecond color timestamps",
                    "nearest-frame repeats normalize the native 8-9 Hz stream to the 10 Hz profile",
                    "camera pose is derived from aligned TCP plus the cfg3 hand-eye transform",
                    "B2 robot-kinematic reference includes arm and hand-eye calibration error",
                    "this clip is excluded from the external-mocap A leaderboard",
                ],
            )
            member_base = f"{dataset['archive_root']}/{sequence}"
            record["source"] = {
                "archive": {
                    "google_drive_id": dataset["google_drive_id"],
                    "mirror_repo": dataset["mirror_repo"],
                    "mirror_revision": dataset["mirror_revision"],
                    "bytes": int(dataset["archive_bytes"]),
                    "sha256": dataset["archive_sha256"],
                    "caller_owned_local_archive": ctx.rh20t_archive is not None,
                    "retained_pipeline_archive": bool(ctx.keep_source),
                },
                "members": {
                    "video": {
                        "name": f"{member_base}/cam_{serial}/color.mp4",
                        "bytes": video_path.stat().st_size,
                        "sha256": _sha256(video_path),
                    },
                    "timestamps": {
                        "name": f"{member_base}/cam_{serial}/timestamps.npy",
                        "bytes": timestamps_path.stat().st_size,
                        "sha256": _sha256(timestamps_path),
                    },
                    "tcp_base": {
                        "name": f"{member_base}/transformed/tcp_base.npy",
                        "bytes": tcp_path.stat().st_size,
                        "sha256": _sha256(tcp_path),
                    },
                },
                "calibration_id": calibration_id,
                "retained_full_scene_sources": bool(ctx.keep_source),
            }
            record["sampling"] = {
                "basis": "color_timestamp_ms_then_source_frame_index",
                "first_source_frame": sampled[0][1],
                "last_source_frame": sampled[-1][1],
                "sampled_frames": len(sampled),
                "unique_source_frames": len(
                    {source_index for _, source_index in sampled}
                ),
                "repeated_output_frames": len(sampled)
                - len({source_index for _, source_index in sampled}),
                "maximum_resampling_error_ms": max(
                    abs(
                        timestamp
                        - (
                            timestamps[0]
                            + 1000.0
                            * (float(clip["start_s"]) + index / ctx.target_fps)
                        )
                    )
                    for index, (timestamp, _) in enumerate(sampled)
                ),
                "exact_tcp_timestamp_frames": sum(
                    int(sample[-1]) for sample in samples
                ),
            }
            record["evaluation_role"] = "robot_kinematic_reference"
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
        completed = len(records) == len(dataset["clips"])
    finally:
        if (
            completed
            and not ctx.keep_source
            and not ctx.robot_with_exo
            and cache_root.exists()
        ):
            _remove_owned_eval_cache(cache_root, ctx.data_root)
    return records


def _parts_after_zip_marker(name: str, marker: str) -> tuple[str, ...] | None:
    parts = PurePosixPath(name).parts
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    return parts[index + 1 :]


def _crc32_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _extract_materialized_zip_member(
    archive_path: Path, member: str, destination: Path
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise DatasetDownloadError(f"Refusing EgoBody ZIP symlink: {member}")
        if (
            destination.is_file()
            and destination.stat().st_size == info.file_size
            and _crc32_file(destination) == info.CRC
        ):
            return destination
        partial = destination.with_name(destination.name + ".part")
        partial.unlink(missing_ok=True)
        with archive.open(info) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if (
            partial.stat().st_size != info.file_size
            or _crc32_file(partial) != info.CRC
        ):
            partial.unlink(missing_ok=True)
            raise DatasetDownloadError(
                f"CRC or size mismatch for EgoBody member: {member}"
            )
        os.replace(partial, destination)
    return destination


def _index_egobody_color_archive(
    archive_path: Path,
    selected_pairs: set[tuple[str, str]],
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str, int], tuple[int, str]],
]:
    pv_members: dict[tuple[str, str], str] = {}
    images: dict[tuple[str, str, int], tuple[int, str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            tail = _parts_after_zip_marker(info.filename, "egocentric_color")
            if tail is None or len(tail) < 3:
                continue
            pair = (tail[0], tail[1])
            if pair not in selected_pairs:
                continue
            if tail[-1].endswith("_pv.txt"):
                pv_members[pair] = info.filename
                continue
            if len(tail) < 4 or tail[-2] != "PV":
                continue
            match = PV_IMAGE_RE.match(tail[-1])
            if match:
                images[(pair[0], pair[1], int(match.group("timestamp")))] = (
                    int(match.group("frame")),
                    info.filename,
                )
    missing = selected_pairs - set(pv_members)
    if missing:
        raise DatasetDownloadError(
            f"EgoBody PV metadata is missing for: {sorted(missing)}"
        )
    return pv_members, images


def _index_egobody_gaze_archive(
    archive_path: Path,
    selected_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(
                "_head_hand_eye.csv"
            ):
                continue
            tail = _parts_after_zip_marker(info.filename, "egocentric_gaze")
            if tail is None or len(tail) < 3:
                continue
            pair = (tail[0], tail[1])
            if pair in selected_pairs:
                result[pair] = info.filename
    missing = selected_pairs - set(result)
    if missing:
        raise DatasetDownloadError(
            f"EgoBody head tracking is missing for: {sorted(missing)}"
        )
    return result


def _index_egobody_calibration_archive(
    archive_path: Path,
    clips: list[dict[str, Any]],
) -> dict[str, list[str]]:
    wanted = {str(clip["sequence"]): str(clip["scene"]) for clip in clips}
    result = {recording: [] for recording in wanted}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            tail = _parts_after_zip_marker(info.filename, "calibrations")
            if tail is None or len(tail) < 2 or tail[0] not in wanted:
                continue
            recording = tail[0]
            if tail[-1] == "holo_to_kinect12.json" or (
                len(tail) >= 3
                and tail[-2] == "kinect12_to_world"
                and tail[-1] == f"{wanted[recording]}.json"
            ):
                result[recording].append(info.filename)
    incomplete = {
        recording: members
        for recording, members in result.items()
        if not any(name.endswith("/holo_to_kinect12.json") for name in members)
    }
    if incomplete:
        raise DatasetDownloadError(
            "EgoBody holo-to-Kinect calibration is missing for: "
            f"{sorted(incomplete)}"
        )
    return result


def _index_egobody_exo_archive(
    archive_path: Path,
    selected_recordings: set[str],
) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            tail = _parts_after_zip_marker(info.filename, "kinect_color")
            if (
                tail is None
                or len(tail) < 3
                or tail[0] not in selected_recordings
                or tail[1] != "master"
            ):
                continue
            match = EXO_IMAGE_RE.match(tail[-1])
            if match:
                result[(tail[0], int(match.group("frame")))] = info.filename
    return result


def _sample_egobody_clip(
    pv_path: Path,
    clip: dict[str, Any],
    image_index: dict[tuple[str, str, int], tuple[int, str]],
    target_fps: int,
) -> tuple[PVCalibration, list[PVRecord]]:
    calibration, raw_records = parse_pv_file(pv_path)
    recording = str(clip["sequence"])
    hololens = str(clip["hololens_sequence"])
    attached: list[PVRecord] = []
    for raw in raw_records:
        indexed = image_index.get((recording, hololens, raw.timestamp))
        if indexed is None:
            continue
        frame_id, member = indexed
        if not (
            int(clip["recording_start_frame"])
            <= frame_id
            <= int(clip["recording_end_frame"])
        ):
            continue
        attached.append(
            PVRecord(
                raw.timestamp,
                raw.fx,
                raw.fy,
                raw.T_W_E,
                frame_id,
                Path(member),
            )
        )
    sampled = sample_records(
        attached,
        float(target_fps),
        float(clip["start_s"]),
        float(clip["duration_s"]),
    )
    expected = int(round(float(clip["duration_s"]) * target_fps))
    if len(sampled) != expected:
        raise DatasetDownloadError(
            f"{recording}: EgoBody sampling produced {len(sampled)}/{expected} frames"
        )
    if (
        sampled[0].frame_id != int(clip["expected_start_frame"])
        or sampled[0].timestamp != int(clip["expected_start_timestamp"])
    ):
        raise DatasetDownloadError(
            f"{recording}: frozen EgoBody start frame/timestamp no longer matches"
        )
    return calibration, sampled


def _write_egobody_reference(
    clip_dir: Path,
    calibration: PVCalibration,
    sampled: list[PVRecord],
) -> tuple[Path, Path, Path]:
    reference = clip_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    frames_csv = clip_dir / "frames.csv"
    frames_partial = frames_csv.with_name(frames_csv.name + ".part")
    with frames_partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("output_index", "timestamp", "frame_id", "filename"))
        for index, record in enumerate(sampled):
            writer.writerow(
                (
                    index,
                    record.timestamp,
                    record.frame_id,
                    Path(str(record.image_path)).name,
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(frames_partial, frames_csv)

    trajectory = reference / "pv_trajectory.csv"
    trajectory_partial = trajectory.with_name(trajectory.name + ".part")
    matrix_columns = [f"t{row}{column}" for row in range(4) for column in range(4)]
    with trajectory_partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp",
                "frame_id",
                "fx",
                "fy",
                "filename",
                *matrix_columns,
            )
        )
        for record in sampled:
            writer.writerow(
                (
                    record.timestamp,
                    record.frame_id,
                    f"{record.fx:.12g}",
                    f"{record.fy:.12g}",
                    Path(str(record.image_path)).name,
                    *(f"{value:.12g}" for value in record.T_W_E.reshape(-1)),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(trajectory_partial, trajectory)

    camera = reference / "pv_camera.json"
    _write_json(
        camera,
        {
            "cx": calibration.cx,
            "cy": calibration.cy,
            "width": calibration.width,
            "height": calibration.height,
            "per_frame_focal_lengths": "pv_trajectory.csv:fx,fy",
            "pose_field": "pv2world_transform",
            "pose_direction": "camera_to_hololens_world",
        },
    )
    return frames_csv, trajectory, camera


def _write_image_manifest(path: Path, images: list[Path]) -> Path:
    partial = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("output_index", "filename", "bytes", "sha256"))
        for index, image in enumerate(images):
            writer.writerow(
                (index, image.name, image.stat().st_size, _sha256(image))
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return path


def _crop_egobody_gaze_member(
    archive_path: Path,
    member: str,
    destination: Path,
    start_timestamp: int,
    end_timestamp: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    padding_ticks = 5_000_000
    minimum = start_timestamp - padding_ticks
    maximum = end_timestamp + padding_ticks
    rows = 0
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as source, partial.open("wb") as output:
            for raw_line in source:
                try:
                    timestamp = int(raw_line.split(b",", 1)[0])
                except ValueError:
                    continue
                if minimum <= timestamp <= maximum:
                    output.write(raw_line)
                    rows += 1
            output.flush()
            os.fsync(output.fileno())
    if rows == 0:
        partial.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"EgoBody head tracking has no rows for {start_timestamp}-{end_timestamp}"
        )
    os.replace(partial, destination)
    return destination


def _remove_owned_eval_cache(cache_root: Path, data_root: Path) -> None:
    if not cache_root.exists():
        return
    if cache_root.is_symlink():
        raise DatasetDownloadError(f"Refusing to remove symlink cache: {cache_root}")
    resolved_cache = cache_root.resolve()
    resolved_root = data_root.resolve()
    if resolved_root not in resolved_cache.parents:
        raise DatasetDownloadError(
            f"Refusing to remove cache outside evaluation root: {resolved_cache}"
        )
    shutil.rmtree(resolved_cache)


def download_egobody(ctx: DownloadContext) -> list[dict[str, Any]]:
    if not ctx.accept_egobody_license:
        raise DatasetDownloadError(
            "EgoBody license confirmation is required; review the official terms, "
            "then pass --accept-egobody-license"
        )
    if ctx.egobody_netrc_file is None:
        raise DatasetDownloadError(
            "EgoBody authentication is required; pass --egobody-netrc-file"
        )
    dataset = ctx.plan["datasets"]["egobody"]
    root = ctx.data_root / "egobody"
    cache_root = ctx.data_root / "_cache" / "remote_zip" / "egobody"
    clips = list(dataset["clips"])
    selected_pairs = {
        (str(clip["sequence"]), str(clip["hololens_sequence"]))
        for clip in clips
    }
    try:
        cache = RemoteZipCache(
            root,
            ctx.egobody_netrc_file,
            connections=ctx.workers,
            cache_root=cache_root,
        )
        color_archive = cache.ensure_index("egocentric_color.zip")
        pv_members, image_index = _index_egobody_color_archive(
            color_archive, selected_pairs
        )

        gaze_archive = cache.ensure_index("egocentric_gaze.zip")
        gaze_members = _index_egobody_gaze_archive(
            gaze_archive, selected_pairs
        )

        calibration_archive = cache.ensure_index("calibrations.zip")
        calibration_members = _index_egobody_calibration_archive(
            calibration_archive, clips
        )
        all_calibrations = sorted(
            {
                member
                for members in calibration_members.values()
                for member in members
            }
        )
        calibration_archive = cache.ensure_members(
            "calibrations.zip", all_calibrations, merge_gap_bytes=0
        )

        exo_archive: Path | None = None
        exo_index: dict[tuple[str, int], str] = {}
        exo_camera: Path | None = None
        if ctx.egobody_with_exo:
            exo_archive = cache.ensure_index("kinect_color.zip")
            exo_index = _index_egobody_exo_archive(
                exo_archive, {str(clip["sequence"]) for clip in clips}
            )
            camera_archive = cache.ensure_index("kinect_cam_params.zip")
            with zipfile.ZipFile(camera_archive) as archive:
                camera_member = _find_suffix(
                    (info.filename for info in archive.infolist()),
                    "/kinect_master/Color.json",
                )
            assert camera_member is not None
            camera_archive = cache.ensure_members(
                "kinect_cam_params.zip",
                [camera_member],
                merge_gap_bytes=0,
            )
            exo_camera = _extract_materialized_zip_member(
                camera_archive,
                camera_member,
                root / "_shared" / "kinect_master" / "Color.json",
            )

        records: list[dict[str, Any]] = []
        for clip in clips:
            recording = str(clip["sequence"])
            hololens = str(clip["hololens_sequence"])
            print(f"[egobody] {recording}")
            clip_dir = root / "clips" / recording
            staging_pv = cache_root / "staging" / f"{recording}_pv.txt"
            color_archive = cache.ensure_members(
                "egocentric_color.zip",
                [pv_members[(recording, hololens)]],
                merge_gap_bytes=0,
            )
            _extract_materialized_zip_member(
                color_archive,
                pv_members[(recording, hololens)],
                staging_pv,
            )
            pv_calibration, sampled = _sample_egobody_clip(
                staging_pv, clip, image_index, ctx.target_fps
            )
            selected_ego_members = [
                str(record.image_path) for record in sampled
            ]
            color_archive = cache.ensure_members(
                "egocentric_color.zip",
                selected_ego_members,
                merge_gap_bytes=512 * 1024,
            )
            frame_dir = clip_dir / "frames"
            frame_paths = [
                _extract_materialized_zip_member(
                    color_archive,
                    member,
                    frame_dir / Path(member).name,
                )
                for member in selected_ego_members
            ]
            frames_csv, trajectory, camera = _write_egobody_reference(
                clip_dir, pv_calibration, sampled
            )
            frame_manifest = _write_image_manifest(
                clip_dir / "frame_manifest.csv", frame_paths
            )

            gaze_member = gaze_members[(recording, hololens)]
            gaze_archive = cache.ensure_members(
                "egocentric_gaze.zip",
                [gaze_member],
                merge_gap_bytes=0,
            )
            gaze = _crop_egobody_gaze_member(
                gaze_archive,
                gaze_member,
                clip_dir / "reference" / "head_hand_eye.csv",
                sampled[0].timestamp,
                sampled[-1].timestamp,
            )

            support = [trajectory, camera, frame_manifest, gaze]
            for member in calibration_members[recording]:
                destination = (
                    clip_dir
                    / "reference"
                    / "calibration"
                    / Path(member).name
                )
                support.append(
                    _extract_materialized_zip_member(
                        calibration_archive, member, destination
                    )
                )

            exo_paths: list[Path] = []
            if ctx.egobody_with_exo:
                assert exo_archive is not None
                assert exo_camera is not None
                selected_exo_members = [
                    exo_index[(recording, int(record.frame_id))]
                    for record in sampled
                    if (recording, int(record.frame_id)) in exo_index
                ]
                minimum_exo = int(len(sampled) * 0.95)
                if len(selected_exo_members) < minimum_exo:
                    raise DatasetDownloadError(
                        f"{recording}: only {len(selected_exo_members)}/"
                        f"{len(sampled)} synchronized exo frames"
                    )
                exo_archive = cache.ensure_members(
                    "kinect_color.zip",
                    selected_exo_members,
                    merge_gap_bytes=512 * 1024,
                )
                exo_paths = [
                    _extract_materialized_zip_member(
                        exo_archive,
                        member,
                        clip_dir / "exo_frames" / Path(member).name,
                    )
                    for member in selected_exo_members
                ]
                support.append(
                    _write_image_manifest(
                        clip_dir / "exo_frame_manifest.csv", exo_paths
                    )
                )
                support.append(exo_camera)

            record = _clip_record(
                "egobody",
                clip,
                frames_csv,
                support,
                str(frame_dir),
                [
                    "PV pose is HoloLens device tracking, not independent mocap GT",
                    "head_hand_eye.csv is cropped to the selected window plus 0.5 s padding",
                    (
                        "synchronized master Kinect exo frames included"
                        if ctx.egobody_with_exo
                        else "exo RGB omitted; add --egobody-with-exo if needed"
                    ),
                ],
            )
            record.update(
                {
                    "reference_grade": dataset["reference_grade"],
                    "split": clip["split"],
                    "scene": clip["scene"],
                    "frame_count": len(frame_paths),
                    "expected_frame_count": int(
                        round(float(clip["duration_s"]) * ctx.target_fps)
                    ),
                    "exo_frame_count": len(exo_paths),
                }
            )
            _write_json(clip_dir / "clip.json", record)
            records.append(record)
        if not ctx.keep_source:
            _remove_owned_eval_cache(cache_root, ctx.data_root)
        return records
    except EgoBodyDownloadError as error:
        raise DatasetDownloadError(str(error)) from error
    except (OSError, PermissionError, ValueError, zipfile.BadZipFile) as error:
        raise DatasetDownloadError(f"EgoBody selective download failed: {error}") from error


def _download_explorer_item(
    item: dict[str, Any], destination: Path
) -> Path:
    expected_size = int(item["file_size_bytes"])
    expected_sha1 = str(item["sha1sum"]).lower()
    if destination.is_file() and (
        destination.stat().st_size != expected_size
        or _sha1(destination).lower() != expected_sha1
    ):
        destination.unlink()
    download_https(item["download_url"], destination)
    if destination.stat().st_size != expected_size:
        destination.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"Size mismatch for official file {item['filename']}"
        )
    actual_sha1 = _sha1(destination).lower()
    if actual_sha1 != expected_sha1:
        destination.unlink(missing_ok=True)
        raise DatasetDownloadError(
            f"SHA1 mismatch for official file {item['filename']}"
        )
    return destination


def _explorer_manifest(url: str) -> dict[str, Any]:
    response = requests.get(
        url,
        headers={"User-Agent": "ego-video-camera-eval-data/1.0"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "sequences" not in payload or "sequence_config" not in payload:
        raise DatasetDownloadError("Invalid Aria Dataset Explorer manifest")
    return payload


def _download_aria_preview_subset(
    ctx: DownloadContext, dataset_name: str
) -> list[dict[str, Any]]:
    if not ctx.accept_aria_licenses:
        raise DatasetDownloadError(
            "Aria/HOT3D license confirmation is required; review the official "
            "dataset terms, then pass --accept-aria-licenses"
        )
    dataset = ctx.plan["datasets"][dataset_name]
    manifest = _explorer_manifest(dataset["explorer_manifest"])
    root = ctx.data_root / dataset_name
    records = []
    for clip in dataset["clips"]:
        sequence = str(clip["sequence"])
        print(f"[{dataset_name}:preview] {sequence}")
        if sequence not in manifest["sequences"]:
            raise DatasetDownloadError(
                f"Sequence is absent from current official manifest: {sequence}"
            )
        available = manifest["sequences"][sequence]
        missing = set(dataset["minimal_groups"]) - set(available)
        if missing:
            raise DatasetDownloadError(
                f"{sequence} lacks official data groups: {sorted(missing)}"
            )
        clip_dir = root / "clips" / sequence
        source_dir = root / "_sources" / sequence
        reference_files: list[Path] = []
        source_items = []
        source_video: Path | None = None
        for group in dataset["minimal_groups"]:
            item = available[group]
            artifact = _download_explorer_item(
                item, source_dir / str(item["filename"])
            )
            source_items.append(
                {
                    "group": group,
                    "filename": item["filename"],
                    "bytes": int(item["file_size_bytes"]),
                    "sha1": item["sha1sum"],
                }
            )
            if group == "video_main_rgb":
                source_video = artifact
            elif zipfile.is_zipfile(artifact):
                reference_files.extend(
                    _safe_extract_zip(artifact, clip_dir / "reference" / group)
                )
                if not ctx.keep_source:
                    artifact.unlink(missing_ok=True)
            else:
                destination = clip_dir / "reference" / group / artifact.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    destination.unlink()
                shutil.move(str(artifact), destination)
                reference_files.append(destination)
        if source_video is None:
            raise DatasetDownloadError(f"No RGB preview was provided for {sequence}")
        output_video = ffmpeg_clip(
            source_video,
            clip_dir / "video.mp4",
            float(clip["start_s"]),
            float(clip["duration_s"]),
            ctx.target_fps,
            ctx.ffmpeg,
        )
        if not ctx.keep_source:
            source_video.unlink(missing_ok=True)
        record = _clip_record(
            dataset_name,
            clip,
            output_video,
            reference_files,
            str(output_video),
            [
                "input is the official H.264 RGB preview, not raw VRS",
                "use --aria-mode raw for strict sensor timestamps and raw imagery",
            ],
        )
        record["official_source_items"] = source_items
        record["aria_mode"] = "preview"
        _write_json(clip_dir / "clip.json", record)
        records.append(record)
    return records


def _download_adt_raw(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["adt"]
    if not ctx.accept_aria_licenses:
        raise DatasetDownloadError(
            "Review the ADT license, then pass --accept-aria-licenses"
        )
    if ctx.adt_cdn_file is None:
        raise DatasetDownloadError(
            "ADT needs --adt-cdn-file (official links JSON, valid for 14 days)"
        )
    if shutil.which("aria_dataset_downloader") is None:
        raise DatasetDownloadError(
            "aria_dataset_downloader is unavailable; install official projectaria-tools"
        )
    sequences = [str(clip["sequence"]) for clip in dataset["clips"]]
    output = ctx.data_root / "adt" / "_sources"
    command = [
        "aria_dataset_downloader",
        "--cdn_file",
        str(ctx.adt_cdn_file),
        "--output_folder",
        str(output),
        "--data_types",
        *[str(value) for value in dataset["official_data_types"]],
        "--sequence_names",
        *sequences,
    ]
    subprocess.run(command, check=True)
    records = []
    for clip in dataset["clips"]:
        records.append(
            {
                "dataset": "adt",
                "sequence_id": clip["sequence"],
                "start_s": float(clip["start_s"]),
                "duration_s": float(clip["duration_s"]),
                "stratum": clip.get("stratum"),
                "source_directory": str(output / clip["sequence"]),
                "status": "source_downloaded_needs_vrs_export",
            }
        )
    return records


def download_adt(ctx: DownloadContext) -> list[dict[str, Any]]:
    if ctx.aria_mode == "preview":
        return _download_aria_preview_subset(ctx, "adt")
    return _download_adt_raw(ctx)


def _hot3d_data_groups(cdn_file: Path) -> list[str]:
    payload = json.loads(cdn_file.read_text(encoding="utf-8"))
    config = payload["sequence_config"]
    groups: list[str] = []
    if config["main"].get("recording") != "None":
        groups.append("main_vrs")
    if config["main"].get("mps") != "None":
        groups.extend(
            [
                "mps_slam_trajectories",
                "mps_slam_calibration",
                "mps_slam_points",
                "mps_slam_summary",
                "mps_eye_gaze",
            ]
        )
    groups.extend(config.get("data_groups", {}).keys())
    return groups


def _download_hot3d_raw(ctx: DownloadContext) -> list[dict[str, Any]]:
    dataset = ctx.plan["datasets"]["hot3d"]
    if not ctx.accept_aria_licenses:
        raise DatasetDownloadError(
            "Review the HOT3D license, then pass --accept-aria-licenses"
        )
    if ctx.hot3d_cdn_file is None:
        raise DatasetDownloadError(
            "HOT3D needs --hot3d-cdn-file (official Aria links JSON)"
        )
    if ctx.hot3d_downloader is None or not ctx.hot3d_downloader.is_file():
        raise DatasetDownloadError(
            "HOT3D needs --hot3d-downloader pointing to "
            "hot3d/data_downloader/dataset_downloader_base_main.py"
        )
    groups = _hot3d_data_groups(ctx.hot3d_cdn_file)
    cli_groups = [group for group in groups if group != "mps_slam_summary"]
    selected = set(dataset["requested_data_groups"])
    for group in cli_groups:
        if any(
            re.search(pattern, group, re.IGNORECASE)
            for pattern in dataset["optional_group_patterns"]
        ):
            selected.add(group)
    missing = set(dataset["requested_data_groups"]) - set(cli_groups)
    if missing:
        raise DatasetDownloadError(
            f"HOT3D CDN manifest lacks required groups: {sorted(missing)}"
        )
    indices = [str(index) for index, group in enumerate(cli_groups) if group in selected]
    sequences = [str(clip["sequence"]) for clip in dataset["clips"]]
    output = ctx.data_root / "hot3d" / "_sources"
    print(f"[hot3d] selected data groups: {sorted(selected)}")
    command = [
        sys.executable,
        str(ctx.hot3d_downloader),
        "--cdn_file",
        str(ctx.hot3d_cdn_file),
        "--output_folder",
        str(output),
        "--data_types",
        *indices,
        "--sequence_names",
        *sequences,
    ]
    subprocess.run(command, check=True)
    return [
        {
            "dataset": "hot3d",
            "sequence_id": clip["sequence"],
            "start_s": float(clip["start_s"]),
            "duration_s": float(clip["duration_s"]),
            "stratum": clip.get("stratum"),
            "source_directory": str(output / clip["sequence"]),
            "downloaded_groups": sorted(selected),
            "status": "source_downloaded_needs_vrs_export",
        }
        for clip in dataset["clips"]
    ]


def download_hot3d(ctx: DownloadContext) -> list[dict[str, Any]]:
    if ctx.aria_mode == "preview":
        return _download_aria_preview_subset(ctx, "hot3d")
    return _download_hot3d_raw(ctx)


DOWNLOADERS = {
    "adt": download_adt,
    "egobody": download_egobody,
    "monado": download_monado,
    "princeton365": download_princeton365,
    "hot3d": download_hot3d,
    "incrowd_vi": download_incrowd,
    "lamaria": download_lamaria,
    "tum_rgbd": download_tum_rgbd,
    "bonn_rgbd_dynamic": download_bonn_rgbd_dynamic,
    "openloris_office": download_openloris_office,
    "droid_wrist": download_droid_wrist,
    "holoassist": download_holoassist,
    "rh20t_wrist": download_rh20t_wrist,
    "stera10m": download_stera10m,
}


def execute_download(ctx: DownloadContext, names: Iterable[str]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "profile": ctx.plan["profile"]["id"],
        "plan_path": ctx.plan["_plan_path"],
        "target_fps": ctx.target_fps,
        "data_root": str(ctx.data_root),
        "datasets": {},
    }
    ctx.data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = ctx.data_root / "evaluation_manifest.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if (
        previous.get("profile") == manifest["profile"]
        and previous.get("data_root") == manifest["data_root"]
        and isinstance(previous.get("datasets"), dict)
    ):
        manifest["datasets"].update(previous["datasets"])
    for name in names:
        started = time.time()
        try:
            records = DOWNLOADERS[name](ctx)
            if ctx.robot_with_exo and name in {"droid_wrist", "rh20t_wrist"}:
                # Lazy import keeps the generic downloader independent of the
                # optional robot demo preparation layer.
                from .robot_exo import prepare_robot_exo

                prepare_robot_exo(
                    ctx.plan,
                    ctx.data_root,
                    ctx.ffmpeg,
                    workers=ctx.workers,
                    keep_source=ctx.keep_source,
                    rh20t_archive=ctx.rh20t_archive,
                    datasets=(name,),
                )
            state = {
                "status": "complete",
                "elapsed_s": round(time.time() - started, 3),
                "clips": records,
            }
        except (DatasetDownloadError, subprocess.CalledProcessError) as error:
            state = {
                "status": "blocked",
                "elapsed_s": round(time.time() - started, 3),
                "reason": str(error),
                "clips": [],
            }
            print(f"[{name}] BLOCKED: {error}", file=sys.stderr)
        manifest["datasets"][name] = state
        _write_json(manifest_path, manifest)
    return manifest


def _verify_strict_rgb_clip(
    clip_dir: Path,
    dataset_name: str,
    sequence: str,
    expected_frames: int,
) -> str | None:
    record = _existing_strict_rgb_record(
        clip_dir, dataset_name, sequence, expected_frames
    )
    if record is None:
        return "missing or incomplete strict RGB clip record"
    characteristics = record.get("input_characteristics", {})
    if (
        characteristics.get("color") != "RGB"
        or characteristics.get("projection_model") != "pinhole"
        or characteristics.get("native_color_stream") is not True
        or characteristics.get("fisheye") is not False
    ):
        return "clip input-characteristics gate failed"
    for item in record.get("files", []):
        try:
            path = Path(item["path"])
            expected_size = int(item["bytes"])
            expected_hash = str(item["sha256"])
        except (KeyError, TypeError, ValueError):
            return "malformed clip file record"
        if not path.is_file() or path.stat().st_size != expected_size:
            return f"recorded file is missing or changed: {path}"
        if _sha256(path) != expected_hash:
            return f"recorded file SHA-256 mismatch: {path}"
    manifest_path = clip_dir / "frame_manifest.csv"
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError) as error:
        return f"unable to read frame manifest: {error}"
    if len(rows) != expected_frames:
        return f"frame manifest count mismatch: {len(rows)} != {expected_frames}"
    images: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        filename = str(row.get("filename", ""))
        if not filename or Path(filename).name != filename or filename in seen:
            return f"unsafe or duplicate frame filename: {filename!r}"
        seen.add(filename)
        image = clip_dir / "frames" / filename
        try:
            expected_size = int(row["bytes"])
            expected_hash = str(row["sha256"])
        except (KeyError, TypeError, ValueError):
            return f"malformed frame manifest row: {filename}"
        if not image.is_file() or image.stat().st_size != expected_size:
            return f"frame is missing or changed: {image}"
        if _sha256(image) != expected_hash:
            return f"frame SHA-256 mismatch: {image}"
        images.append(image)
    try:
        width, height = _validate_rgb_images(images)
        camera = json.loads(
            (clip_dir / "reference" / "camera.json").read_text(encoding="utf-8")
        )
    except (DatasetDownloadError, OSError, ValueError) as error:
        return f"RGB/camera validation failed: {error}"
    if (
        camera.get("projection_model") != "pinhole"
        or camera.get("native_color_stream") is not True
        or camera.get("fisheye") is not False
        or int(camera.get("width", -1)) != width
        or int(camera.get("height", -1)) != height
    ):
        return "camera metadata gate failed"
    return None


def verify_download(
    plan: dict[str, Any],
    data_root: Path,
    names: Iterable[str],
    *,
    robot_with_exo: bool = False,
) -> dict[str, Any]:
    names = tuple(names)
    result: dict[str, Any] = {"datasets": {}, "ok": True}
    for name in names:
        dataset = plan["datasets"][name]
        missing = []
        ready = 0
        invalid_reasons: dict[str, str] = {}
        for clip in dataset["clips"]:
            clip_dir = data_root / name / "clips" / clip["sequence"]
            clip_json = clip_dir / "clip.json"
            if name in {
                "tum_rgbd",
                "bonn_rgbd_dynamic",
                "openloris_office",
                "droid_wrist",
                "holoassist",
                "rh20t_wrist",
                "stera10m",
            }:
                expected = int(
                    round(
                        float(clip["duration_s"])
                        * int(plan["profile"]["target_fps"])
                    )
                )
                reason = _verify_strict_rgb_clip(
                    clip_dir, name, str(clip["sequence"]), expected
                )
                if reason is None:
                    ready += 1
                else:
                    missing.append(clip["sequence"])
                    invalid_reasons[str(clip["sequence"])] = reason
            elif name == "egobody" and clip_json.is_file():
                try:
                    payload = json.loads(clip_json.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    payload = {}
                expected = int(
                    round(
                        float(clip["duration_s"])
                        * int(plan["profile"]["target_fps"])
                    )
                )
                images = [
                    path
                    for path in (clip_dir / "frames").glob("*")
                    if path.is_file() and PV_IMAGE_RE.match(path.name)
                ]
                required = (
                    clip_dir / "frames.csv",
                    clip_dir / "frame_manifest.csv",
                    clip_dir / "reference" / "pv_trajectory.csv",
                    clip_dir / "reference" / "pv_camera.json",
                    clip_dir / "reference" / "head_hand_eye.csv",
                )
                if (
                    len(images) == expected
                    and int(payload.get("frame_count", -1)) == expected
                    and all(path.is_file() and path.stat().st_size > 0 for path in required)
                ):
                    ready += 1
                else:
                    missing.append(clip["sequence"])
            elif clip_json.is_file():
                ready += 1
            elif name in {"adt", "hot3d"}:
                source = data_root / name / "_sources" / clip["sequence"]
                if source.exists():
                    ready += 1
                else:
                    missing.append(clip["sequence"])
            else:
                missing.append(clip["sequence"])
        status = {
            "expected": len(dataset["clips"]),
            "ready_or_source_downloaded": ready,
            "missing": missing,
            "invalid_reasons": invalid_reasons,
        }
        result["datasets"][name] = status
        if missing:
            result["ok"] = False
    if robot_with_exo:
        robot_names = tuple(
            name for name in names if name in {"droid_wrist", "rh20t_wrist"}
        )
        if robot_names:
            from .robot_exo import verify_robot_exo

            robot_report = verify_robot_exo(plan, data_root, robot_names)
            result["robot_exo"] = robot_report
            result["ok"] = bool(result["ok"] and robot_report["ok"])
    return result


def build_parser(default_plan: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download only the fixed ego-pose evaluation subset."
    )
    parser.add_argument(
        "action", choices=("plan", "download", "verify"), nargs="?", default="plan"
    )
    parser.add_argument("--plan", type=Path, default=default_plan)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("EGO_POSE_EVAL_DATA", "data/ego_pose_eval_core65")),
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated names, or all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--archive-tool",
        default="bsdtar",
        help="Archive reader with 7z support, required for OpenLORIS packages",
    )
    parser.add_argument(
        "--aria-mode",
        choices=("preview", "raw"),
        default="preview",
        help="preview downloads official H.264 RGB plus pose; raw downloads VRS",
    )
    parser.add_argument(
        "--accept-aria-licenses",
        action="store_true",
        help="Confirm that you reviewed and accept the official ADT/HOT3D terms",
    )
    parser.add_argument(
        "--egobody-netrc-file",
        type=Path,
        default=(
            Path(os.environ["EGOBODY_NETRC"])
            if os.environ.get("EGOBODY_NETRC")
            else None
        ),
        help="Owner-readable (mode 600) netrc containing official EgoBody access",
    )
    parser.add_argument(
        "--accept-egobody-license",
        action="store_true",
        help="Confirm that you reviewed and accept the official EgoBody terms",
    )
    parser.add_argument(
        "--egobody-with-exo",
        action="store_true",
        help="Also fetch synchronized master-Kinect RGB for selected EgoBody frames",
    )
    parser.add_argument(
        "--robot-with-exo",
        action="store_true",
        help=(
            "Also prepare the fixed synchronized exterior-camera streams for the "
            "DROID/RH20T robot demo"
        ),
    )
    parser.add_argument("--adt-cdn-file", type=Path)
    parser.add_argument("--hot3d-cdn-file", type=Path)
    parser.add_argument("--hot3d-downloader", type=Path)
    parser.add_argument(
        "--rh20t-archive",
        type=Path,
        help=(
            "Use an already downloaded RH20T_cfg3.tar.gz; the caller-owned file "
            "is verified but never deleted"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without network or filesystem writes",
    )
    return parser


def _resolve_executable(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved:
        return resolved
    sibling = Path(sys.executable).resolve().with_name(value)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def main(argv: list[str] | None = None) -> int:
    default_plan = (
        Path(__file__).resolve().parents[2] / "configs" / "ego_pose_eval_core65.yaml"
    )
    args = build_parser(default_plan).parse_args(argv)
    plan = load_plan(args.plan)
    names = selected_datasets(plan, args.datasets)
    summary = plan_summary(plan, names)
    if args.action == "plan" or args.dry_run:
        print_summary(summary)
        if args.dry_run and args.action == "download":
            print("Dry-run: no network requests or filesystem writes were made.")
        return 0
    if args.action == "verify":
        report = verify_download(
            plan,
            args.data_root.resolve(),
            names,
            robot_with_exo=args.robot_with_exo,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 2
    ffmpeg = _resolve_executable(args.ffmpeg)
    needs_ffmpeg = any(
        name
        in {
            "princeton365",
            "incrowd_vi",
            "droid_wrist",
            "holoassist",
            "rh20t_wrist",
            "stera10m",
        }
        for name in names
    ) or (
        args.aria_mode == "preview"
        and any(name in {"adt", "hot3d"} for name in names)
    )
    if ffmpeg is None and needs_ffmpeg:
        raise SystemExit(f"ffmpeg executable was not found: {args.ffmpeg}")
    archive_tool = _resolve_executable(args.archive_tool)
    if archive_tool is None and "openloris_office" in names:
        raise SystemExit(
            f"7z-capable archive executable was not found: {args.archive_tool}"
        )
    context = DownloadContext(
        plan=plan,
        data_root=args.data_root.resolve(),
        target_fps=int(plan["profile"]["target_fps"]),
        workers=max(1, args.workers),
        keep_source=args.keep_source,
        ffmpeg=ffmpeg or args.ffmpeg,
        aria_mode=args.aria_mode,
        accept_aria_licenses=args.accept_aria_licenses,
        egobody_netrc_file=(
            args.egobody_netrc_file.resolve()
            if args.egobody_netrc_file
            else None
        ),
        accept_egobody_license=args.accept_egobody_license,
        egobody_with_exo=args.egobody_with_exo,
        robot_with_exo=args.robot_with_exo,
        adt_cdn_file=args.adt_cdn_file.resolve() if args.adt_cdn_file else None,
        hot3d_cdn_file=(
            args.hot3d_cdn_file.resolve() if args.hot3d_cdn_file else None
        ),
        hot3d_downloader=(
            args.hot3d_downloader.resolve() if args.hot3d_downloader else None
        ),
        rh20t_archive=(
            args.rh20t_archive.resolve() if args.rh20t_archive else None
        ),
        archive_tool=archive_tool or args.archive_tool,
    )
    manifest = execute_download(context, names)
    blocked = [
        name
        for name, state in manifest["datasets"].items()
        if state["status"] == "blocked"
    ]
    print_summary(summary)
    if blocked:
        print(f"Blocked datasets: {', '.join(blocked)}")
        print(
            "Public datasets that completed remain usable; rerun with the missing "
            "official access files to resume."
        )
    return 0
