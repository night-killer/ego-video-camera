#!/usr/bin/env python3
"""Resumable parallel HTTP range downloader for slow direct-download servers."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)$")
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--timeout", type=int, default=240)
    return parser.parse_args()


def session() -> requests.Session:
    value = getattr(THREAD_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        value.trust_env = False
        value.headers["User-Agent"] = "ego-video-camera-checkpoint-downloader/1.0"
        THREAD_LOCAL.session = value
    return value


def probe(url: str, timeout: int) -> tuple[str, int, str | None]:
    response = session().head(
        url,
        allow_redirects=True,
        timeout=(30, timeout),
    )
    response.raise_for_status()
    content_length = int(response.headers.get("Content-Length", "-1"))
    return response.url, content_length, response.headers.get("ETag")


def load_completed(path: Path, total_chunks: int) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line:
            continue
        index = int(raw_line)
        if not 0 <= index < total_chunks:
            raise RuntimeError(f"Invalid chunk index {index} in {path}")
        completed.add(index)
    return completed


def download_chunk(
    *,
    url: str,
    index: int,
    start: int,
    end: int,
    total_size: int,
    output_fd: int,
    timeout: int,
) -> int:
    expected_length = end - start + 1
    for attempt in range(1, 21):
        try:
            with session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(30, timeout),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"chunk {index}: expected HTTP 206, got "
                        f"{response.status_code}"
                    )

                match = CONTENT_RANGE_RE.fullmatch(
                    response.headers.get("Content-Range", "")
                )
                if match is None:
                    raise RuntimeError(
                        f"chunk {index}: missing/invalid Content-Range"
                    )
                actual_start, actual_end, actual_total = match.groups()
                if (
                    int(actual_start) != start
                    or int(actual_end) != end
                    or actual_total == "*"
                    or int(actual_total) != total_size
                ):
                    raise RuntimeError(
                        f"chunk {index}: unexpected Content-Range "
                        f"{response.headers.get('Content-Range')!r}"
                    )

                offset = start
                for block in response.iter_content(chunk_size=256 * 1024):
                    if not block:
                        continue
                    os.pwrite(output_fd, block, offset)
                    offset += len(block)
            if offset - start != expected_length:
                raise RuntimeError(
                    f"chunk {index}: expected {expected_length} bytes, "
                    f"received {offset - start}"
                )
            return expected_length
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 20:
                raise RuntimeError(
                    f"chunk {index} failed after {attempt} attempts"
                ) from error
            time.sleep(min(30, attempt * 2))
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    if args.expected_size <= 0:
        raise ValueError("--expected-size must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    aria2_control = Path(f"{output}.aria2")
    partial = Path(f"{output}.range.part")
    journal = Path(f"{output}.range.done")
    metadata = Path(f"{output}.range.json")

    if (
        output.is_file()
        and output.stat().st_size == args.expected_size
        and not aria2_control.exists()
    ):
        print(f"Already complete: {output} ({args.expected_size} bytes)", flush=True)
        return 0

    final_url, remote_size, etag = probe(args.url, args.timeout)
    if remote_size != args.expected_size:
        raise RuntimeError(
            f"Remote size mismatch: expected {args.expected_size}, got {remote_size}"
        )

    expected_metadata = {
        "url": args.url,
        "final_url": final_url,
        "expected_size": args.expected_size,
        "chunk_size": args.chunk_size,
        "etag": etag,
    }
    if metadata.exists():
        existing_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if existing_metadata != expected_metadata:
            raise RuntimeError(
                f"Resume metadata mismatch in {metadata}; refusing to mix files"
            )
    else:
        metadata.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    total_chunks = (args.expected_size + args.chunk_size - 1) // args.chunk_size
    completed = load_completed(journal, total_chunks)
    completed_bytes = sum(
        min(args.chunk_size, args.expected_size - index * args.chunk_size)
        for index in completed
    )

    output_fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(output_fd, args.expected_size)
    journal_handle = journal.open("a", encoding="utf-8", buffering=1)
    journal_lock = threading.Lock()
    start_time = time.monotonic()
    initial_bytes = completed_bytes
    last_report = 0.0

    pending: list[tuple[int, int, int]] = []
    for index in range(total_chunks):
        if index in completed:
            continue
        start = index * args.chunk_size
        end = min(args.expected_size, start + args.chunk_size) - 1
        pending.append((index, start, end))

    print(
        f"Range download: {output} | {len(completed)}/{total_chunks} chunks "
        f"already complete | workers={args.workers}",
        flush=True,
    )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_chunk,
                    url=final_url,
                    index=index,
                    start=start,
                    end=end,
                    total_size=args.expected_size,
                    output_fd=output_fd,
                    timeout=args.timeout,
                ): (index, end - start + 1)
                for index, start, end in pending
            }
            for future in as_completed(futures):
                index, expected_length = futures[future]
                received = future.result()
                if received != expected_length:
                    raise RuntimeError(
                        f"chunk {index}: expected {expected_length}, got {received}"
                    )

                os.fsync(output_fd)
                with journal_lock:
                    journal_handle.write(f"{index}\n")
                    journal_handle.flush()
                    os.fsync(journal_handle.fileno())
                completed_bytes += received

                now = time.monotonic()
                if now - last_report >= 30 or completed_bytes == args.expected_size:
                    elapsed = max(0.001, now - start_time)
                    transferred = completed_bytes - initial_bytes
                    speed = transferred / elapsed
                    remaining = args.expected_size - completed_bytes
                    eta = remaining / speed if speed > 0 else float("inf")
                    print(
                        f"Progress: {completed_bytes / args.expected_size:6.2%} | "
                        f"{speed / 2**20:6.2f} MiB/s | ETA {eta / 60:6.1f} min",
                        flush=True,
                    )
                    last_report = now
    finally:
        journal_handle.close()
        os.close(output_fd)

    if completed_bytes != args.expected_size:
        raise RuntimeError(
            f"Incomplete download: expected {args.expected_size}, got {completed_bytes}"
        )

    os.replace(partial, output)
    for state_path in (journal, metadata, aria2_control):
        state_path.unlink(missing_ok=True)
    print(f"Complete: {output} ({args.expected_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
