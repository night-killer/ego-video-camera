import os
import zipfile
from pathlib import Path

from ego_video_camera.download import RemoteFile
from ego_video_camera import remote_zip
from ego_video_camera.remote_zip import _merge_intervals, parse_zip_directory


def test_parse_normal_zip_central_directory(tmp_path: Path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("one.txt", b"one")
        archive.writestr("two.txt", b"two")
    data = path.read_bytes()
    info = parse_zip_directory(data, 0)
    assert info.entry_count == 2
    assert info.offset > 0
    assert info.size > 0


def test_merge_intervals_includes_small_archive_gaps():
    assert _merge_intervals([(0, 9), (15, 20), (100, 110)], gap_bytes=5) == [
        (0, 20),
        (100, 110),
    ]


def test_sparse_materialization_replaces_invalid_resume_and_parallelizes(
    tmp_path: Path, monkeypatch
):
    netrc = tmp_path / "dummy.netrc"
    netrc.write_text("machine example.invalid\n", encoding="utf-8")
    netrc.chmod(0o600)
    cache = remote_zip.RemoteZipCache(
        tmp_path / "data",
        netrc,
        connections=4,
        cache_root=tmp_path / "cache",
    )
    remote = RemoteFile(
        name="egocentric_color.zip",
        url="https://egobody.ethz.ch/data/dataset/egocentric_color.zip",
        content_length=20 * 1024**2,
        etag=None,
        accept_ranges="bytes",
    )
    target, _, segment_root = cache._paths(remote.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    os.truncate(target, remote.content_length)

    start, inclusive_end = 1024, 10 * 1024**2 + 1023
    # A 10 MiB range with four workers is split into four 2.5 MiB pieces.
    segment_size = (inclusive_end - start + 1) // 4
    first_end = start + segment_size - 1
    segment_root.mkdir(parents=True, exist_ok=True)
    invalid = segment_root / f"{start:012d}_{first_end:012d}.part"
    invalid.touch()
    os.truncate(invalid, first_end + 2)
    observed = []

    def fake_download(remote_file, netrc_file, path, piece_start, piece_end):
        observed.append((piece_start, piece_end, path.exists()))
        path.touch()
        os.truncate(path, piece_end - piece_start + 1)
        return piece_start, piece_end, path

    monkeypatch.setattr(remote_zip, "_download_segment", fake_download)
    cache._materialize_ranges(remote.name, remote, target, [(start, inclusive_end)])

    assert len(observed) == 4
    assert (start, first_end, False) in observed
    assert not list(segment_root.glob("*.part"))
