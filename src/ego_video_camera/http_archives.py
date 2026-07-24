from __future__ import annotations

import binascii
import hashlib
import json
import os
import struct
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from .remote_zip import parse_zip_directory


USER_AGENT = "ego-video-camera-eval-data/1.0"
LOCAL_ZIP_HEADER = struct.Struct("<4s5H3L2H")


class RangeRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpObject:
    url: str
    size: int
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class TarMember:
    name: str
    size: int
    data_offset: int
    typeflag: str


class HttpRangeClient:
    """Strict HTTPS byte-range reader with bounded retries and identity checks."""

    def __init__(self, timeout_s: float = 60.0, retries: int = 5) -> None:
        self.timeout_s = timeout_s
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def inspect(self, url: str) -> HttpObject:
        if not url.startswith("https://"):
            raise ValueError(f"Only HTTPS sources are accepted: {url}")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self.session.get(
                    url,
                    headers={"Range": "bytes=0-0"},
                    allow_redirects=True,
                    stream=True,
                    timeout=self.timeout_s,
                ) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code == 206 and "/" in content_range:
                        size = int(content_range.rsplit("/", 1)[1])
                    elif response.status_code == 200:
                        value = response.headers.get("Content-Length")
                        if value is None:
                            raise RangeRequestError(
                                "Server returned neither Content-Range nor Content-Length"
                            )
                        size = int(value)
                    else:
                        raise RangeRequestError(
                            f"Unexpected status while inspecting object: {response.status_code}"
                        )
                    return HttpObject(
                        # Keep the stable origin URL. Hugging Face can redirect each
                        # requested range to a range-specific signed CDN URL; reusing
                        # that final URL for a different range returns HTTP 403.
                        url=url,
                        size=size,
                        etag=response.headers.get("ETag"),
                        last_modified=response.headers.get("Last-Modified"),
                    )
            except (OSError, requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RangeRequestError(f"Unable to inspect remote object: {url}") from last_error

    def read(self, remote: HttpObject, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= remote.size:
            raise ValueError(f"Invalid byte range {start}-{end} for {remote.size} bytes")
        expected = end - start + 1
        headers = {"Range": f"bytes={start}-{end}"}
        if remote.etag:
            headers["If-Match"] = remote.etag
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    remote.url,
                    headers=headers,
                    allow_redirects=True,
                    timeout=self.timeout_s,
                )
                if response.status_code != 206:
                    raise RangeRequestError(
                        f"Server did not honor byte range ({response.status_code})"
                    )
                response.raise_for_status()
                payload = response.content
                if len(payload) != expected:
                    raise RangeRequestError(
                        f"Short byte range: expected {expected}, received {len(payload)}"
                    )
                return payload
            except (OSError, requests.RequestException, RangeRequestError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RangeRequestError(
            f"Unable to read bytes {start}-{end} from remote object"
        ) from last_error

    def copy_range(
        self,
        remote: HttpObject,
        start: int,
        size: int,
        destination: str | Path,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        if destination.is_file() and destination.stat().st_size == size:
            return destination
        if partial.exists() and partial.stat().st_size > size:
            partial.unlink()
        completed = partial.stat().st_size if partial.exists() else 0
        if completed == size:
            os.replace(partial, destination)
            return destination
        end = start + size - 1
        headers = {"Range": f"bytes={start + completed}-{end}"}
        if remote.etag:
            headers["If-Match"] = remote.etag
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self.session.get(
                    remote.url,
                    headers=headers,
                    allow_redirects=True,
                    stream=True,
                    timeout=self.timeout_s,
                ) as response:
                    if response.status_code != 206:
                        raise RangeRequestError(
                            f"Server did not honor byte range ({response.status_code})"
                        )
                    response.raise_for_status()
                    with partial.open("ab") as handle:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                if partial.stat().st_size != size:
                    raise RangeRequestError(
                        f"Short range transfer: {partial.stat().st_size} != {size}"
                    )
                os.replace(partial, destination)
                return destination
            except (OSError, requests.RequestException, RangeRequestError) as error:
                last_error = error
                completed = partial.stat().st_size if partial.exists() else 0
                if completed > size:
                    partial.unlink()
                    completed = 0
                headers["Range"] = f"bytes={start + completed}-{end}"
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RangeRequestError(f"Unable to copy range into {destination}") from last_error


class RemoteZip:
    """Read selected members of a remote ZIP without downloading the archive."""

    def __init__(
        self,
        url: str,
        cache_dir: str | Path,
        client: HttpRangeClient | None = None,
    ) -> None:
        self.client = client or HttpRangeClient()
        self.remote = self.client.inspect(url)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        self.sparse_path = cache_dir / f"{key}.zip.index"
        self.meta_path = cache_dir / f"{key}.json"
        self._entries = self._load_index()

    def _identity(self) -> dict[str, object]:
        return {
            "url": self.remote.url,
            "size": self.remote.size,
            "etag": self.remote.etag,
            "last_modified": self.remote.last_modified,
        }

    def _load_index(self) -> dict[str, zipfile.ZipInfo]:
        identity = self._identity()
        valid = False
        if self.sparse_path.is_file() and self.meta_path.is_file():
            try:
                metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
                valid = metadata.get("identity") == identity
            except (OSError, ValueError):
                valid = False
        if not valid:
            self._build_index(identity)
        try:
            with zipfile.ZipFile(self.sparse_path) as archive:
                return {entry.filename: entry for entry in archive.infolist()}
        except zipfile.BadZipFile:
            self._build_index(identity)
            with zipfile.ZipFile(self.sparse_path) as archive:
                return {entry.filename: entry for entry in archive.infolist()}

    def _build_index(self, identity: dict[str, object]) -> None:
        tail_size = min(self.remote.size, 256 * 1024)
        tail_start = self.remote.size - tail_size
        tail = self.client.read(self.remote, tail_start, self.remote.size - 1)
        directory = parse_zip_directory(tail, tail_start)
        directory_bytes = self.client.read(
            self.remote,
            directory.offset,
            directory.offset + directory.size - 1,
        )
        partial = self.sparse_path.with_name(self.sparse_path.name + ".part")
        partial.unlink(missing_ok=True)
        with partial.open("w+b") as handle:
            handle.truncate(self.remote.size)
            handle.seek(directory.offset)
            handle.write(directory_bytes)
            handle.seek(tail_start)
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, self.sparse_path)
        metadata = {
            "identity": identity,
            "entry_count": directory.entry_count,
            "central_directory_offset": directory.offset,
            "central_directory_size": directory.size,
            "cache_kind": "sparse_zip_index",
        }
        self.meta_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def info(self, name: str) -> zipfile.ZipInfo:
        try:
            return self._entries[name]
        except KeyError as error:
            raise KeyError(f"ZIP member not found: {name}") from error

    def read(self, name: str) -> bytes:
        entry = self.info(name)
        if entry.flag_bits & 0x1:
            raise NotImplementedError(f"Encrypted ZIP member is unsupported: {name}")
        header = self.client.read(
            self.remote, entry.header_offset, entry.header_offset + 29
        )
        values = LOCAL_ZIP_HEADER.unpack(header)
        if values[0] != b"PK\x03\x04":
            raise zipfile.BadZipFile(f"Invalid local header for {name}")
        filename_length, extra_length = values[-2:]
        payload_start = entry.header_offset + 30 + filename_length + extra_length
        if entry.compress_size:
            compressed = self.client.read(
                self.remote,
                payload_start,
                payload_start + entry.compress_size - 1,
            )
        else:
            compressed = b""
        if entry.compress_type == zipfile.ZIP_STORED:
            payload = compressed
        elif entry.compress_type == zipfile.ZIP_DEFLATED:
            payload = zlib.decompress(compressed, -15)
        else:
            raise NotImplementedError(
                f"Unsupported ZIP compression method {entry.compress_type}: {name}"
            )
        if len(payload) != entry.file_size:
            raise zipfile.BadZipFile(f"Uncompressed size mismatch for {name}")
        if (binascii.crc32(payload) & 0xFFFFFFFF) != entry.CRC:
            raise zipfile.BadZipFile(f"CRC mismatch for {name}")
        return payload

    def extract(self, name: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == self.info(name).file_size:
            return destination
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(self.read(name))
        os.replace(partial, destination)
        return destination


def _tar_number(raw: bytes) -> int:
    if raw and raw[0] & 0x80:
        value = int.from_bytes(raw, byteorder="big", signed=True)
        return value & ((1 << (len(raw) * 8 - 1)) - 1)
    text = raw.rstrip(b"\0 ").lstrip(b" ")
    return int(text or b"0", 8)


def _pax_fields(payload: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    position = 0
    while position < len(payload):
        space = payload.find(b" ", position)
        if space < 0:
            break
        try:
            length = int(payload[position:space])
        except ValueError:
            break
        record = payload[space + 1 : position + length].rstrip(b"\n")
        if b"=" in record:
            key, value = record.split(b"=", 1)
            fields[key.decode("utf-8", "replace")] = value.decode(
                "utf-8", "replace"
            )
        position += length
    return fields


class RemoteTar:
    """Index and extract early members from a remote uncompressed TAR."""

    def __init__(
        self,
        url: str,
        client: HttpRangeClient | None = None,
    ) -> None:
        self.client = client or HttpRangeClient()
        self.remote = self.client.inspect(url)

    def scan(
        self,
        stop_suffixes: Iterable[str] = (),
        max_members: int = 256,
    ) -> list[TarMember]:
        wanted = tuple(stop_suffixes)
        found: set[str] = set()
        entries: list[TarMember] = []
        offset = 0
        pax: dict[str, str] = {}
        long_name: str | None = None
        for _ in range(max_members):
            if offset + 512 > self.remote.size:
                break
            header = self.client.read(self.remote, offset, offset + 511)
            if not any(header):
                break
            stored_checksum = _tar_number(header[148:156])
            checksum_header = bytearray(header)
            checksum_header[148:156] = b"        "
            if sum(checksum_header) != stored_checksum:
                raise ValueError(f"Invalid TAR header checksum at offset {offset}")
            raw_name = header[:100].split(b"\0", 1)[0].decode(
                "utf-8", "replace"
            )
            prefix = header[345:500].split(b"\0", 1)[0].decode(
                "utf-8", "replace"
            )
            name = f"{prefix}/{raw_name}" if prefix else raw_name
            size = _tar_number(header[124:136])
            typeflag = header[156:157].decode("ascii", "replace") or "0"
            data_offset = offset + 512
            if typeflag in {"x", "g"}:
                payload = (
                    self.client.read(
                        self.remote, data_offset, data_offset + size - 1
                    )
                    if size
                    else b""
                )
                fields = _pax_fields(payload)
                if typeflag == "x":
                    pax = fields
                offset = data_offset + ((size + 511) // 512) * 512
                continue
            if typeflag == "L":
                payload = self.client.read(
                    self.remote, data_offset, data_offset + size - 1
                )
                long_name = payload.rstrip(b"\0\n").decode("utf-8", "replace")
                offset = data_offset + ((size + 511) // 512) * 512
                continue
            effective_name = pax.get("path") or long_name or name
            entry = TarMember(effective_name, size, data_offset, typeflag)
            entries.append(entry)
            for suffix in wanted:
                if effective_name.endswith(suffix):
                    found.add(suffix)
            pax = {}
            long_name = None
            offset = data_offset + ((size + 511) // 512) * 512
            if wanted and found == set(wanted):
                break
        return entries

    def read(self, member: TarMember) -> bytes:
        if member.size == 0:
            return b""
        return self.client.read(
            self.remote,
            member.data_offset,
            member.data_offset + member.size - 1,
        )

    def extract(self, member: TarMember, destination: str | Path) -> Path:
        return self.client.copy_range(
            self.remote, member.data_offset, member.size, destination
        )
