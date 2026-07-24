from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import requests
import yaml

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
from .remote_zip import RemoteZipCache


DATASET_ORDER = (
    "adt",
    "egobody",
    "monado",
    "princeton365",
    "hot3d",
    "incrowd_vi",
    "lamaria",
)
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


class DatasetDownloadError(RuntimeError):
    pass


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
    missing = set(DATASET_ORDER) - set(datasets)
    if missing:
        raise ValueError(f"Plan is missing datasets: {sorted(missing)}")
    unexpected = set(datasets) - set(DATASET_ORDER)
    if unexpected:
        raise ValueError(f"Plan contains unsupported datasets: {sorted(unexpected)}")
    count = 0
    duration = 0.0
    sequence_keys: set[tuple[str, str]] = set()
    for dataset_name in DATASET_ORDER:
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
    payload["_plan_path"] = str(path.resolve())
    return payload


def selected_datasets(plan: dict[str, Any], value: str | None) -> tuple[str, ...]:
    if not value or value == "all":
        return DATASET_ORDER
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested) - set(plan["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return tuple(name for name in DATASET_ORDER if name in requested)


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
) -> Path:
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS sources are accepted: {url}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_name(destination.name + ".part")
    session = requests.Session()
    session.headers["User-Agent"] = "ego-video-camera-eval-data/1.0"
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
    magnitude = max(abs(values[0]), abs(values[-1]))
    if magnitude > 1e12 or median > 1e6:
        return 1e9
    if magnitude > 1e9 or median > 1e3:
        return 1e6
    if magnitude > 1e6 or median > 1:
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


def verify_download(
    plan: dict[str, Any], data_root: Path, names: Iterable[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {"datasets": {}, "ok": True}
    for name in names:
        dataset = plan["datasets"][name]
        missing = []
        ready = 0
        for clip in dataset["clips"]:
            clip_dir = data_root / name / "clips" / clip["sequence"]
            clip_json = clip_dir / "clip.json"
            if name == "egobody" and clip_json.is_file():
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
        }
        result["datasets"][name] = status
        if missing:
            result["ok"] = False
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
    parser.add_argument("--adt-cdn-file", type=Path)
    parser.add_argument("--hot3d-cdn-file", type=Path)
    parser.add_argument("--hot3d-downloader", type=Path)
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
        report = verify_download(plan, args.data_root.resolve(), names)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 2
    ffmpeg = _resolve_executable(args.ffmpeg)
    needs_ffmpeg = any(
        name in {"princeton365", "incrowd_vi"} for name in names
    ) or (
        args.aria_mode == "preview"
        and any(name in {"adt", "hot3d"} for name in names)
    )
    if ffmpeg is None and needs_ffmpeg:
        raise SystemExit(f"ffmpeg executable was not found: {args.ffmpeg}")
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
        adt_cdn_file=args.adt_cdn_file.resolve() if args.adt_cdn_file else None,
        hot3d_cdn_file=(
            args.hot3d_cdn_file.resolve() if args.hot3d_cdn_file else None
        ),
        hot3d_downloader=(
            args.hot3d_downloader.resolve() if args.hot3d_downloader else None
        ),
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
