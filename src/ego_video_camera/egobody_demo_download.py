"""Selective downloader for the metadata-only EgoBody 80-clip manifest.

The official RGB archives are very large.  This module uses :class:`RemoteZipCache`
to fetch only ZIP central directories, PV metadata, and the image members named by
the manifest.  A local ZIP can be supplied in tests (or when an archive has already
been downloaded); remote URLs are restricted by ``RemoteZipCache`` to the official
ETH Zürich host.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import shutil
import stat
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .download import EgoBodyDownloadError, validate_netrc
from .egobody_io import EXO_IMAGE_RE, PV_IMAGE_RE
from .remote_zip import RemoteZipCache
from .serialization import write_json


MANIFEST_SCHEMA = "egobody_demo_selection_v1"
SOURCE_FPS = 30
DEFAULT_SAMPLE_FPS = 8


@dataclass(frozen=True)
class ImageMember:
    frame_id: int
    timestamp: int | None
    name: str


@dataclass(frozen=True)
class SequenceMembers:
    recording: str
    sequence: str
    pv_member: str
    images: tuple[ImageMember, ...]


def _parts_after_marker(name: str, marker: str) -> tuple[str, ...] | None:
    parts = PurePosixPath(name).parts
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    return parts[index + 1 :]


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe ZIP member path: {name}")
    return path


def _safe_component(value: str, field: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe {field}: {value!r}")
    return value


def _crc32_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def index_color_archive(archive_path: str | Path) -> dict[str, list[SequenceMembers]]:
    """Index PV sequences and image members by recording name.

    Only central-directory metadata is read.  Image payloads are not touched.
    """

    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            tail = _parts_after_marker(info.filename, "egocentric_color")
            if tail is None or len(tail) < 3:
                continue
            recording, sequence = tail[0], tail[1]
            key = (recording, sequence)
            state = by_pair.setdefault(key, {"pv": None, "images": []})
            if tail[-1].endswith("_pv.txt"):
                state["pv"] = info.filename
                continue
            if len(tail) < 4 or tail[-2].upper() != "PV":
                continue
            match = PV_IMAGE_RE.match(tail[-1])
            if match:
                state["images"].append(
                    ImageMember(
                        frame_id=int(match.group("frame")),
                        timestamp=int(match.group("timestamp")),
                        name=info.filename,
                    )
                )
    result: dict[str, list[SequenceMembers]] = {}
    for (recording, sequence), state in sorted(by_pair.items()):
        pv = state["pv"]
        images = tuple(sorted(state["images"], key=lambda item: (item.frame_id, item.name)))
        if not isinstance(pv, str) or not images:
            continue
        # A malformed archive can contain duplicate central-directory entries.
        # Keep one deterministic member for each frame ID.
        unique: dict[int, ImageMember] = {}
        for image in images:
            unique.setdefault(image.frame_id, image)
        result.setdefault(recording, []).append(
            SequenceMembers(recording, sequence, pv, tuple(unique.values()))
        )
    return result


def index_exo_archive(archive_path: str | Path) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            tail = _parts_after_marker(info.filename, "kinect_color")
            if tail is None or len(tail) < 3 or tail[1].lower() != "master":
                continue
            match = EXO_IMAGE_RE.match(tail[-1])
            if match:
                result.setdefault((tail[0], int(match.group("frame"))), info.filename)
    return result


def validate_manifest(
    payload: dict, desktop_count: int | None = None, walking_count: int | None = None
) -> list[dict]:
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported EgoBody manifest schema: {payload.get('schema_version')!r}")
    categories = payload.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("Manifest categories are missing")
    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    def _clip_list(category: str) -> list:
        value = categories.get(category)
        return value.get("clips", []) if isinstance(value, dict) else []

    expected = {
        "desktop_head_motion": (
            desktop_count
            if desktop_count is not None
            else int(selection.get("desktop_count", len(_clip_list("desktop_head_motion"))))
        ),
        "walking_person": (
            walking_count
            if walking_count is not None
            else int(selection.get("walking_count", len(_clip_list("walking_person"))))
        ),
    }
    clips: list[dict] = []
    seen_ids: set[str] = set()
    seen_windows: set[tuple[str, int, int]] = set()
    for category, count in expected.items():
        item = categories.get(category)
        if not isinstance(item, dict) or len(item.get("clips", [])) != count:
            actual = len(item.get("clips", [])) if isinstance(item, dict) else 0
            raise ValueError(f"{category}: expected {count} primary clips, found {actual}")
        for clip in item["clips"]:
            clip_id = str(clip.get("clip_id", ""))
            if not clip_id or clip_id in seen_ids:
                raise ValueError(f"Duplicate or empty clip_id: {clip_id!r}")
            seen_ids.add(clip_id)
            recording = str(clip.get("recording_name", ""))
            start = int(clip["frame_start_inclusive"])
            end = int(clip["frame_end_exclusive"])
            count_frames = int(clip["frame_count"])
            if not recording or start < 0 or end <= start or end - start != count_frames:
                raise ValueError(f"Invalid frame window in {clip_id}")
            key = (recording, start, end)
            if key in seen_windows:
                raise ValueError(f"Duplicate RGB time window in {clip_id}")
            seen_windows.add(key)
            clip = dict(clip)
            clip["category"] = category
            clips.append(clip)
    for first_index, first in enumerate(clips):
        for second in clips[first_index + 1 :]:
            if (
                first["recording_name"] == second["recording_name"]
                and int(first["frame_start_inclusive"]) < int(second["frame_end_exclusive"])
                and int(second["frame_start_inclusive"]) < int(first["frame_end_exclusive"])
            ):
                raise ValueError(f"Overlapping primary RGB windows: {first['clip_id']} / {second['clip_id']}")
    return clips


def _choose_sequence(candidates: list[SequenceMembers], start: int, end: int) -> SequenceMembers:
    viable = [item for item in candidates if bisect.bisect_left([x.frame_id for x in item.images], start) < len(item.images)]
    if not viable:
        raise ValueError(f"No PV sequence contains requested frame range {start}:{end}")
    # Prefer the sequence with the most frames in the requested interval.
    return max(
        viable,
        key=lambda item: sum(start <= image.frame_id < end for image in item.images),
    )


def _sample_images(sequence: SequenceMembers, start: int, end: int, sample_fps: int) -> list[ImageMember]:
    if sample_fps <= 0 or sample_fps > SOURCE_FPS:
        raise ValueError(f"sample_fps must be in 1..{SOURCE_FPS}")
    expected = int(round((end - start) * sample_fps / SOURCE_FPS))
    available = [image for image in sequence.images if start <= image.frame_id < end]
    if len(available) < expected:
        raise ValueError(
            f"{sequence.recording}/{sequence.sequence}: only {len(available)} source frames in {start}:{end}, expected {expected}"
        )
    frame_ids = [image.frame_id for image in available]
    selected: list[ImageMember] = []
    # Keep the assignment monotonic.  A sparse archive must never cause a
    # later target to reuse an earlier frame, because that silently scrambles
    # the temporal order of the emitted clip.
    last_index = -1
    max_error = SOURCE_FPS / sample_fps
    for index in range(expected):
        target = start + (index * SOURCE_FPS) / sample_fps
        lower_bound = last_index + 1
        # Reserve one source frame for each remaining target.  This avoids
        # consuming the tail too early and gives a deterministic coverage error.
        upper_bound = len(available) - (expected - index) + 1
        position = bisect.bisect_left(frame_ids, target, lo=lower_bound, hi=upper_bound)
        options = [value for value in (position - 1, position) if lower_bound <= value < upper_bound]
        if not options:
            raise ValueError(
                f"{sequence.recording}/{sequence.sequence}: sampling coverage ended "
                f"before target frame {target:g} in {start}:{end}"
            )
        best = min(options, key=lambda value: (abs(frame_ids[value] - target), frame_ids[value]))
        error = abs(frame_ids[best] - target)
        if error > max_error:
            raise ValueError(
                f"{sequence.recording}/{sequence.sequence}: sampling gap near target "
                f"frame {target:g} (nearest source frame error {error:g} > {max_error:g})"
            )
        selected.append(available[best])
        last_index = best
    return selected


def _extract_members(archive_path: Path, jobs: Iterable[tuple[str, Path]]) -> list[Path]:
    """Extract several members while opening the (possibly huge) ZIP once."""

    materialized = list(jobs)
    outputs: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member, destination in materialized:
            _safe_member_name(member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            info = archive.getinfo(member)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Refusing ZIP symlink: {member}")
            if (
                destination.is_file()
                and destination.stat().st_size == info.file_size
                and _crc32_file(destination) == info.CRC
            ):
                outputs.append(destination)
                continue
            partial = destination.with_name(destination.name + ".part")
            partial.unlink(missing_ok=True)
            with archive.open(info) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if partial.stat().st_size != info.file_size or _crc32_file(partial) != info.CRC:
                partial.unlink(missing_ok=True)
                raise zipfile.BadZipFile(f"Short or corrupt ZIP member: {member}")
            os.replace(partial, destination)
            outputs.append(destination)
    return outputs


def _extract_member(archive_path: Path, member: str, destination: Path) -> Path:
    return _extract_members(archive_path, [(member, destination)])[0]


def _destination_for_image(frame_dir: Path, output_index: int, image: ImageMember) -> Path:
    suffix = Path(image.name).suffix.lower() or ".jpg"
    return frame_dir / f"{output_index:06d}_{image.frame_id:06d}{suffix}"


def _destination_for_exo(exo_dir: Path, output_index: int, member: str, frame_id: int) -> Path:
    suffix = Path(member).suffix.lower() or ".jpg"
    return exo_dir / f"{output_index:06d}_{frame_id:06d}{suffix}"


def _write_frames_csv(path: Path, selected: list[ImageMember], paths: list[Path]) -> None:
    partial = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("output_index", "source_frame_id", "source_timestamp", "filename"))
        for index, (image, output) in enumerate(zip(selected, paths)):
            writer.writerow((index, image.frame_id, image.timestamp, output.name))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _local_or_remote_archive(
    *, name: str, local_path: Path | None, cache: RemoteZipCache | None
) -> Path:
    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        return local_path
    if cache is None:
        raise ValueError("Either a local archive or a remote cache is required")
    return cache.ensure_index(name)


def download_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    netrc_file: str | Path | None = None,
    *,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    workers: int = 8,
    with_exo: bool = False,
    ego_archive: str | Path | None = None,
    exo_archive: str | Path | None = None,
    cache_root: str | Path | None = None,
    accept_license: bool = False,
) -> dict:
    if not accept_license:
        raise ValueError("EgoBody license confirmation is required (--accept-egobody-license)")
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    clips = validate_manifest(payload)
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    local_ego = Path(ego_archive).resolve() if ego_archive else None
    local_exo = Path(exo_archive).resolve() if exo_archive else None
    cache: RemoteZipCache | None = None
    if local_ego is None or (with_exo and local_exo is None):
        if netrc_file is None:
            raise ValueError("Official EgoBody access requires --netrc-file")
        cache = RemoteZipCache(root, validate_netrc(netrc_file), connections=workers, cache_root=cache_root)
    try:
        ego_path = _local_or_remote_archive(name="egocentric_color.zip", local_path=local_ego, cache=cache)
        color_index = index_color_archive(ego_path)
        exo_path: Path | None = None
        exo_index: dict[tuple[str, int], str] = {}
        if with_exo:
            exo_path = _local_or_remote_archive(name="kinect_color.zip", local_path=local_exo, cache=cache)
            exo_index = index_exo_archive(exo_path)
        # RemoteZipCache initially contains only the central directory.  Plan
        # all payload ranges before opening the sparse ZIP with ``zipfile``.
        selected_members: list[str] = []
        selected_by_clip: dict[str, tuple[SequenceMembers, list[ImageMember]]] = {}
        for clip in clips:
            recording = str(clip["recording_name"])
            _safe_component(str(clip["clip_id"]), "clip_id")
            start = int(clip["frame_start_inclusive"])
            end = int(clip["frame_end_exclusive"])
            sequence = _choose_sequence(color_index.get(recording, []), start, end)
            selected = _sample_images(sequence, start, end, sample_fps)
            selected_by_clip[str(clip["clip_id"])] = (sequence, selected)
            selected_members.extend([sequence.pv_member, *(image.name for image in selected)])
        if cache is not None and local_ego is None:
            ego_path = cache.ensure_members("egocentric_color.zip", selected_members)
        if cache is not None and with_exo and local_exo is None:
            exo_members = [
                member
                for recording, frame_id in {
                    (str(clip["recording_name"]), image.frame_id)
                    for clip in clips
                    for image in selected_by_clip[str(clip["clip_id"])][1]
                }
                if (member := exo_index.get((recording, frame_id))) is not None
            ]
            assert exo_path is not None
            exo_path = cache.ensure_members("kinect_color.zip", exo_members)
        records = []
        for clip in clips:
            recording = str(clip["recording_name"])
            start = int(clip["frame_start_inclusive"])
            end = int(clip["frame_end_exclusive"])
            sequence, selected = selected_by_clip[str(clip["clip_id"])]
            clip_root = root / "clips" / _safe_component(str(clip["clip_id"]), "clip_id")
            frame_dir = clip_root / "frames"
            frame_jobs = [
                (image.name, _destination_for_image(frame_dir, output_index, image))
                for output_index, image in enumerate(selected)
            ]
            frame_paths = _extract_members(ego_path, frame_jobs)
            sequence_name = _safe_component(sequence.sequence, "hololens_sequence")
            _extract_member(ego_path, sequence.pv_member, clip_root / "reference" / f"{sequence_name}_pv.txt")
            _write_frames_csv(clip_root / "frames.csv", selected, frame_paths)
            exo_paths = []
            if with_exo and exo_path is not None:
                exo_dir = clip_root / "exo_frames"
                exo_jobs = []
                for output_index, image in enumerate(selected):
                    member = exo_index.get((recording, image.frame_id))
                    if member is None:
                        continue
                    exo_jobs.append((member, _destination_for_exo(exo_dir, output_index, member, image.frame_id)))
                exo_paths = _extract_members(exo_path, exo_jobs)
            record = {
                "clip_id": clip["clip_id"],
                "category": clip["category"],
                "recording_name": recording,
                "hololens_sequence": sequence.sequence,
                "frame_start_inclusive": start,
                "frame_end_exclusive": end,
                "source_fps": SOURCE_FPS,
                "sample_fps": sample_fps,
                "frame_count": len(frame_paths),
                "exo_frame_count": len(exo_paths),
                "frames_csv": str((clip_root / "frames.csv").relative_to(root)),
                "head_motion_metric_status": clip.get("head_motion_metric_status"),
                "source_action_labels": clip.get("source_action_labels", []),
            }
            write_json(clip_root / "clip.json", record)
            records.append(record)
        manifest_file = Path(manifest_path).resolve()
        manifest_digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        result = {
            "schema_version": "egobody_demo_download_v1",
            "manifest": str(manifest_file),
            "manifest_sha256": manifest_digest,
            "sample_fps": sample_fps,
            "clips": records,
        }
        write_json(root / "download_manifest.json", result)
        # A prior authentication failure marker is no longer authoritative
        # once a complete successful manifest has been written.
        (root / "download_blocked.json").unlink(missing_ok=True)
        return result
    except EgoBodyDownloadError as error:
        blocked = {"status": "blocked", "name": error.name, "http_status": error.http_status, "reason": error.reason}
        write_json(root / "download_blocked.json", blocked)
        raise


def download_exo_only(
    data_root: str | Path,
    netrc_file: str | Path | None = None,
    *,
    workers: int = 8,
    exo_archive: str | Path | None = None,
    cache_root: str | Path | None = None,
    require_all: bool = False,
) -> dict:
    """Materialize only the Kinect master frames for an existing demo80 tree.

    The ego images and PV references are already present after the first
    selective download.  Reusing them avoids reopening the large ego archive;
    only the exact source frame IDs listed by each clip are requested from the
    official Kinect archive (or an explicitly supplied local archive).
    """

    root = Path(data_root).resolve()
    download_manifest_path = root / "download_manifest.json"
    if not download_manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed demo manifest: {download_manifest_path}")
    payload = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("download_manifest.json contains no clips")
    local_exo = Path(exo_archive).resolve() if exo_archive else None
    cache: RemoteZipCache | None = None
    if local_exo is None:
        if netrc_file is None:
            raise ValueError("--netrc-file is required to fetch Kinect exo frames")
        effective_cache_root = (
            Path(cache_root).expanduser()
            if cache_root is not None
            else Path("/tmp/ego_video_camera_remote_zip_cache")
            / hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        )
        cache = RemoteZipCache(
            root,
            validate_netrc(netrc_file),
            connections=workers,
            cache_root=effective_cache_root,
        )
    archive = local_exo if local_exo is not None else cache.ensure_index("kinect_color.zip")
    exo_index = index_exo_archive(archive)
    requested: list[tuple[dict, list[tuple[int, int, str | None]]]] = []
    members: list[str] = []
    requested_total = 0
    for clip in clips:
        clip_id = _safe_component(str(clip.get("clip_id", "")), "clip_id")
        clip_root = root / "clips" / clip_id
        csv_path = clip_root / "frames.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing frames.csv for {clip_id}: {csv_path}")
        rows: list[tuple[int, int, int]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"output_index", "source_frame_id"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"{csv_path} must contain output_index and source_frame_id")
            for row in reader:
                rows.append(
                    (
                        int(row["output_index"]),
                        int(row["source_frame_id"]),
                        int(row.get("source_timestamp") or 0),
                    )
                )
        jobs: list[tuple[int, int, str | None]] = []
        recording = str(clip["recording_name"])
        for output_index, frame_id, timestamp in rows:
            member = exo_index.get((recording, frame_id))
            jobs.append((output_index, frame_id, member))
            if member is not None:
                members.append(member)
        requested.append((clip, jobs))
        requested_total += len(jobs)
    unique_members = list(dict.fromkeys(members))
    if cache is not None:
        archive = cache.ensure_members("kinect_color.zip", unique_members)
    updated: list[dict] = []
    missing_total = 0
    for clip, jobs in requested:
        clip_id = _safe_component(str(clip["clip_id"]), "clip_id")
        exo_dir = root / "clips" / clip_id / "exo_frames"
        extraction_jobs: list[tuple[str, Path]] = []
        for output_index, frame_id, member in jobs:
            if member is None:
                missing_total += 1
                continue
            suffix = Path(member).suffix.lower() or ".jpg"
            destination = exo_dir / f"{output_index:06d}_{frame_id:06d}{suffix}"
            extraction_jobs.append((member, destination))
        _extract_members(archive, extraction_jobs)
        record_path = root / "clips" / clip_id / "clip.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else dict(clip)
        record["exo_frame_count"] = len(extraction_jobs)
        record["exo_frame_missing_count"] = len(jobs) - len(extraction_jobs)
        write_json(record_path, record)
        updated.append(record)
    result = {
        "schema_version": "egobody_demo_exo_download_v1",
        "data_root": str(root),
        "archive": str(archive),
        "clip_count": len(updated),
        "requested_frame_count": requested_total,
        "materialized_frame_count": sum(int(item.get("exo_frame_count", 0)) for item in updated),
        "missing_frame_count": missing_total,
        "clips": updated,
    }
    write_json(root / "exo_download_manifest.json", result)
    if require_all and missing_total:
        raise RuntimeError(
            f"Kinect archive is missing {missing_total} requested frames; "
            "run with a complete official kinect_color.zip or replace the affected clips"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--netrc-file", type=Path)
    parser.add_argument("--sample-fps", type=int, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--with-exo", action="store_true")
    parser.add_argument("--ego-archive", type=Path, help="Local official egocentric_color.zip (testing/resume only)")
    parser.add_argument("--exo-archive", type=Path, help="Local official kinect_color.zip")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--accept-egobody-license", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = download_manifest(
        args.manifest, args.data_root, args.netrc_file,
        sample_fps=args.sample_fps, workers=args.workers,
        with_exo=args.with_exo, ego_archive=args.ego_archive,
        exo_archive=args.exo_archive, cache_root=args.cache_root,
        accept_license=args.accept_egobody_license,
    )
    print(f"Downloaded {len(result['clips'])} clips at {args.sample_fps} FPS")
    return 0


__all__ = [
    "download_manifest", "index_color_archive", "index_exo_archive",
    "download_exo_only", "validate_manifest", "main",
]
