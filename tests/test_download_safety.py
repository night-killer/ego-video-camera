import json
import zipfile
from pathlib import Path

import pytest

from ego_video_camera.download import (
    EgoBodyDownloadError,
    RemoteFile,
    _download_segment,
    _bind_or_validate_remote_identity,
    _record_download_blocked,
    _record_download_result,
    _segment_ranges,
    extract_members,
    official_url,
    validate_downloaded_payload,
    validate_netrc,
)


def test_official_manifest_rejects_unapproved_urls():
    with pytest.raises(ValueError):
        official_url("../secret")


def test_zip_extraction_canonicalizes_modality_root_and_blocks_traversal(tmp_path: Path):
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("release/egocentric_color/recording/sequence/PV/frame.jpg", b"ok")
        archive.writestr("../escape.txt", b"bad")
    paths = extract_members(
        archive_path,
        tmp_path / "data",
        ["release/egocentric_color/recording/sequence/PV/frame.jpg"],
    )
    assert paths == [
        tmp_path / "data/egocentric_color/recording/sequence/PV/frame.jpg"
    ]
    with pytest.raises(ValueError):
        extract_members(archive_path, tmp_path / "data", ["../escape.txt"])


def test_netrc_rejects_group_or_other_permissions(tmp_path: Path):
    path = tmp_path / "credentials"
    path.write_text("placeholder", encoding="utf-8")
    path.chmod(0o600)
    assert validate_netrc(path) == path.resolve()
    path.chmod(0o640)
    with pytest.raises(PermissionError):
        validate_netrc(path)


def test_parallel_ranges_are_fixed_size_and_contiguous():
    ranges = _segment_ranges(7, 107, segment_size=32)
    assert ranges == [(7, 38), (39, 70), (71, 102), (103, 106)]


def test_segment_retry_recomputes_bounded_range_offset(tmp_path: Path, monkeypatch):
    calls = []

    def fake_curl(name, netrc, arguments, require_status=None):
        byte_range = arguments[arguments.index("--range") + 1]
        calls.append((byte_range, require_status))
        output = Path(arguments[arguments.index("--output") + 1])
        if len(calls) == 1:
            with output.open("wb") as handle:
                handle.write(b"12345")
            raise EgoBodyDownloadError(name, 206, "network_or_server_error")
        with output.open("wb") as handle:
            handle.write(b"x" * 27)
        return ""

    monkeypatch.setattr("ego_video_camera.download._run_curl", fake_curl)
    monkeypatch.setattr("ego_video_camera.download.time.sleep", lambda _: None)
    remote = RemoteFile("calibrations.zip", "https://egobody.ethz.ch/data/dataset/calibrations.zip", 39, None, "bytes")
    result = _download_segment(remote, tmp_path / "unused", tmp_path / "segment", 7, 38)
    assert result[0:2] == (7, 38)
    assert (tmp_path / "segment").stat().st_size == 32
    assert calls == [("7-38", {206}), ("12-38", {206})]


def test_segment_accepts_complete_206_body_when_curl_reports_late_error(
    tmp_path: Path, monkeypatch
):
    calls = 0

    def fake_curl(name, netrc, arguments, require_status=None):
        nonlocal calls
        calls += 1
        output = Path(arguments[arguments.index("--output") + 1])
        output.write_bytes(b"x" * 32)
        raise EgoBodyDownloadError(name, 206, "network_or_server_error")

    monkeypatch.setattr("ego_video_camera.download._run_curl", fake_curl)
    remote = RemoteFile(
        "calibrations.zip",
        "https://egobody.ethz.ch/data/dataset/calibrations.zip",
        39,
        None,
        "bytes",
    )
    _download_segment(remote, tmp_path / "unused", tmp_path / "segment", 7, 38)
    assert calls == 1
    assert (tmp_path / "segment").stat().st_size == 32


def test_segment_recovers_validated_transfer_after_process_interruption(
    tmp_path: Path, monkeypatch
):
    segment = tmp_path / "segment"
    remote = RemoteFile(
        "calibrations.zip",
        "https://egobody.ethz.ch/data/dataset/calibrations.zip",
        39,
        '"stable-etag"',
        "bytes",
    )

    def interrupted_curl(name, netrc, arguments, require_status=None):
        output = Path(arguments[arguments.index("--output") + 1])
        headers = Path(arguments[arguments.index("--dump-header") + 1])
        output.write_bytes(b"12345")
        headers.write_text(
            "HTTP/1.1 206 Partial Content\r\n"
            "Content-Range: bytes 7-38/39\r\n"
            "Content-Length: 32\r\n"
            'ETag: "stable-etag"\r\n\r\n',
            encoding="iso-8859-1",
        )
        raise KeyboardInterrupt

    monkeypatch.setattr("ego_video_camera.download._run_curl", interrupted_curl)
    with pytest.raises(KeyboardInterrupt):
        _download_segment(remote, tmp_path / "unused", segment, 7, 38)

    calls = []

    def resumed_curl(name, netrc, arguments, require_status=None):
        byte_range = arguments[arguments.index("--range") + 1]
        calls.append(byte_range)
        output = Path(arguments[arguments.index("--output") + 1])
        output.write_bytes(b"x" * 27)
        return ""

    monkeypatch.setattr("ego_video_camera.download._run_curl", resumed_curl)
    _download_segment(remote, tmp_path / "unused", segment, 7, 38)
    assert calls == ["12-38"]
    assert segment.read_bytes() == b"12345" + b"x" * 27
    assert not segment.with_suffix(".transfer").exists()
    assert not segment.with_suffix(".transfer.headers").exists()


def test_segment_discards_unverifiable_stale_transfer(tmp_path: Path, monkeypatch):
    segment = tmp_path / "segment"
    segment.write_bytes(b"12345")
    segment.with_suffix(".transfer").write_bytes(b"untrusted")
    calls = []

    def fake_curl(name, netrc, arguments, require_status=None):
        byte_range = arguments[arguments.index("--range") + 1]
        calls.append(byte_range)
        output = Path(arguments[arguments.index("--output") + 1])
        output.write_bytes(b"x" * 27)
        return ""

    monkeypatch.setattr("ego_video_camera.download._run_curl", fake_curl)
    remote = RemoteFile(
        "calibrations.zip",
        "https://egobody.ethz.ch/data/dataset/calibrations.zip",
        39,
        None,
        "bytes",
    )
    _download_segment(remote, tmp_path / "unused", segment, 7, 38)
    assert calls == ["12-38"]
    assert segment.read_bytes() == b"12345" + b"x" * 27


def test_segment_migrates_legacy_absolute_offset_file(tmp_path: Path, monkeypatch):
    segment = tmp_path / "segment"
    expected = b"z" * 32
    with segment.open("wb") as handle:
        handle.truncate(39)
        handle.seek(7)
        handle.write(expected)

    def fake_curl(name, netrc, arguments, require_status=None):
        raise AssertionError("a complete legacy segment must not be downloaded again")

    monkeypatch.setattr("ego_video_camera.download._run_curl", fake_curl)
    remote = RemoteFile("calibrations.zip", "https://egobody.ethz.ch/data/dataset/calibrations.zip", 39, None, "bytes")
    _download_segment(remote, tmp_path / "unused", segment, 7, 38)
    assert segment.read_bytes() == expected


def test_download_manifest_merges_independent_results(tmp_path: Path):
    first = {"name": "egocentric_color.zip", "status": "downloaded"}
    second = {"name": "kinect_color.zip", "status": "downloaded"}
    _record_download_result(tmp_path, second)
    _record_download_result(tmp_path, first)
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["credentials_recorded"] is False
    assert [item["name"] for item in manifest["files"]] == [
        "egocentric_color.zip",
        "kinect_color.zip",
    ]


def test_successful_result_clears_only_matching_stale_block(tmp_path: Path):
    blocked = tmp_path / "download_blocked.json"
    blocked.write_text(
        json.dumps({"file": "egocentric_gaze.zip", "reason": "network_or_server_error"}),
        encoding="utf-8",
    )
    _record_download_result(tmp_path, {"name": "calibrations.zip"})
    assert blocked.exists()
    _record_download_result(tmp_path, {"name": "egocentric_gaze.zip"})
    assert not blocked.exists()


def test_blocked_report_is_sanitized(tmp_path: Path):
    error = EgoBodyDownloadError(
        "egocentric_color.zip", 403, "authentication_expired_or_rejected"
    )
    _record_download_blocked(tmp_path, error)
    report = json.loads((tmp_path / "download_blocked.json").read_text())
    assert report == {
        "status": "blocked",
        "reason": "authentication_expired_or_rejected",
        "http_status": 403,
        "file": "egocentric_color.zip",
        "source": "https://egobody.ethz.ch/data/dataset/",
        "credentials_recorded": False,
        "action": "Renew EgoBody official authentication and resume the same command",
    }


def test_completed_zip_is_crc_checked_before_promotion(tmp_path: Path):
    archive = tmp_path / "payload.zip.part"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("egocentric_color/recording/sequence/PV/frame.jpg", b"pixels")
    result = validate_downloaded_payload(archive)
    assert result["zip_member_count"] == 1
    assert result["zip_crc_all_members"] is True
    assert len(result["sha256"]) == 64


def test_partial_identity_sidecar_detects_etag_change(tmp_path: Path):
    partial = tmp_path / "egocentric_color.zip.part"
    partial.write_bytes(b"prefix")
    first = RemoteFile(
        "egocentric_color.zip",
        "https://egobody.ethz.ch/data/dataset/egocentric_color.zip",
        100,
        '"etag-one"',
        "bytes",
    )
    changed = RemoteFile(first.name, first.url, first.content_length, '"etag-two"', "bytes")
    sidecar = _bind_or_validate_remote_identity(first, partial)
    recorded = json.loads(sidecar.read_text())
    assert recorded["credentials_recorded"] is False
    assert recorded["adopted_existing_payload"] is True
    with pytest.raises(EgoBodyDownloadError, match="remote_identity_changed"):
        _bind_or_validate_remote_identity(changed, partial)


def test_completed_zip_rejects_unsafe_member_before_promotion(tmp_path: Path):
    archive = tmp_path / "unsafe.zip.part"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", b"bad")
    with pytest.raises(ValueError):
        validate_downloaded_payload(archive)
