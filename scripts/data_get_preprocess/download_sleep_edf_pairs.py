#!/usr/bin/env python
"""
Download a small subject-wise subset of Sleep-EDF sleep-cassette files.

Default behavior:
  - Select 21 distinct subjects from the Sleep-EDF Expanded sleep-cassette set.
  - Download one PSG file and its matching Hypnogram file for each subject.
  - Save files into ../dataset relative to this script.
  - Show a per-file progress bar.
  - Resume partial downloads when the server supports HTTP Range requests.

Example:
  python scripts/data_get_preprocess/download_sleep_edf_pairs.py

Useful options:
  python scripts/data_get_preprocess/download_sleep_edf_pairs.py --num-subjects 21
  python scripts/data_get_preprocess/download_sleep_edf_pairs.py --dest dataset
  python scripts/data_get_preprocess/download_sleep_edf_pairs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
CHUNK_SIZE = 1024 * 512


@dataclass(frozen=True)
class FilePair:
    subject_id: str
    recording_id: str
    psg: str
    hypnogram: str


def read_url_text(url: str, timeout: int) -> str:
    req = Request(url, headers={"User-Agent": "sleep-edf-course-project-downloader/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def list_edf_files(base_url: str, timeout: int) -> list[str]:
    html = read_url_text(base_url, timeout)
    files = sorted(set(re.findall(r'href="([^"]+\.edf)"', html)))
    if not files:
        raise RuntimeError(f"No .edf files found at {base_url}")
    return files


def select_pairs(files: Iterable[str], num_subjects: int) -> list[FilePair]:
    psg_files = sorted(name for name in files if name.endswith("-PSG.edf"))
    hyp_files = sorted(name for name in files if name.endswith("-Hypnogram.edf"))

    pairs: list[FilePair] = []
    seen_subjects: set[str] = set()
    for psg in psg_files:
        # Sleep-cassette examples:
        #   SC4001E0-PSG.edf
        #   SC4001EC-Hypnogram.edf
        # Subject id is SC400; recording id is SC4001.
        subject_id = psg[:5]
        recording_id = psg[:6]
        if subject_id in seen_subjects:
            continue

        matching_hyp = next((name for name in hyp_files if name.startswith(recording_id)), None)
        if matching_hyp is None:
            continue

        pairs.append(
            FilePair(
                subject_id=subject_id,
                recording_id=recording_id,
                psg=psg,
                hypnogram=matching_hyp,
            )
        )
        seen_subjects.add(subject_id)
        if len(pairs) >= num_subjects:
            break

    if len(pairs) < num_subjects:
        raise RuntimeError(f"Only found {len(pairs)} matched distinct-subject pairs.")
    return pairs


def remote_file_size(url: str, timeout: int) -> int | None:
    req = Request(url, method="HEAD", headers={"User-Agent": "sleep-edf-course-project-downloader/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        return None


def progress_line(name: str, downloaded: int, total: int | None, started_at: float) -> str:
    elapsed = max(time.time() - started_at, 1e-6)
    speed = downloaded / elapsed
    speed_mb = speed / (1024 * 1024)

    if total:
        width = 32
        frac = min(downloaded / total, 1.0)
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        pct = 100 * frac
        done_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        return f"\r{name:28s} [{bar}] {pct:6.2f}% {done_mb:7.1f}/{total_mb:7.1f} MB {speed_mb:5.1f} MB/s"

    done_mb = downloaded / (1024 * 1024)
    return f"\r{name:28s} {done_mb:7.1f} MB {speed_mb:5.1f} MB/s"


def download_one(base_url: str, filename: str, dest: Path, timeout: int, retries: int) -> None:
    url = urljoin(base_url, filename)
    final_path = dest / filename
    part_path = dest / f"{filename}.part"
    remote_size = remote_file_size(url, timeout)

    if final_path.exists() and remote_size is not None and final_path.stat().st_size == remote_size:
        print(f"SKIP complete {filename} ({remote_size / (1024 * 1024):.1f} MB)")
        return

    if final_path.exists() and remote_size is not None and final_path.stat().st_size != remote_size:
        # Treat an incomplete final file from a previous interrupted download as a partial file.
        if part_path.exists():
            part_path.unlink()
        final_path.replace(part_path)

    for attempt in range(1, retries + 1):
        existing = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": "sleep-edf-course-project-downloader/1.0"}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", None)
                if existing > 0 and status != 206:
                    # Server did not honor Range; restart safely.
                    existing = 0
                    part_path.unlink(missing_ok=True)

                mode = "ab" if existing > 0 else "wb"
                downloaded = existing
                total = remote_size
                started_at = time.time()

                with part_path.open(mode) as out:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        print(progress_line(filename, downloaded, total, started_at), end="", flush=True)
                print()

            if remote_size is not None and part_path.stat().st_size != remote_size:
                raise RuntimeError(
                    f"downloaded {part_path.stat().st_size} bytes but expected {remote_size} bytes"
                )

            part_path.replace(final_path)
            print(f"DONE {filename}")
            return
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"\nAttempt {attempt}/{retries} failed for {filename}: {exc}")
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent
    default_dest = project_dir / "dataset"
    parser = argparse.ArgumentParser(description="Download a small Sleep-EDF subset with progress bars.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="PhysioNet sleep-cassette directory URL.")
    parser.add_argument("--dest", default=str(default_dest), help="Destination folder for EDF files.")
    parser.add_argument("--num-subjects", type=int, default=21, help="Number of distinct subjects to download.")
    parser.add_argument("--timeout", type=int, default=60, help="Network timeout in seconds.")
    parser.add_argument("--retries", type=int, default=5, help="Retries per file.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected files; do not download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Reading file list from {args.base_url}")
    files = list_edf_files(args.base_url, args.timeout)
    pairs = select_pairs(files, args.num_subjects)

    print(f"\nSelected {len(pairs)} distinct-subject PSG/Hypnogram pairs:")
    for i, pair in enumerate(pairs, start=1):
        print(f"{i:02d}. {pair.subject_id} {pair.recording_id}: {pair.psg} + {pair.hypnogram}")

    manifest_path = dest / "sleep_edf_download_manifest.json"
    manifest_path.write_text(json.dumps([asdict(pair) for pair in pairs], indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run complete. No files downloaded.")
        return 0

    print(f"\nDownloading to: {dest}")
    for pair in pairs:
        download_one(args.base_url, pair.psg, dest, args.timeout, args.retries)
        download_one(args.base_url, pair.hypnogram, dest, args.timeout, args.retries)

    print("\nAll downloads completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
