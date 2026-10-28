"""Build and download the project's reproducible 440-track corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "dataset_manifest.csv"
AI_REVISION = "cf62db6dc87ca2cbab2ed5e1170bfa727a632013"
AI_DATASET = "ai-music4you/ai-generated-songs"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-music-detector/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def append_rows(rows: list[dict[str, str]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing = MANIFEST.exists()
    fields = ["track_id", "label", "audio_url", "source", "license", "source_revision"]
    if existing:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            known_ids = {row["track_id"] for row in csv.DictReader(handle)}
        rows = [row for row in rows if row["track_id"] not in known_ids]
    if not rows:
        return
    with MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def build_ai_manifest(count: int) -> None:
    rows: list[dict[str, str]] = []
    offset = 0
    while len(rows) < count:
        query = urllib.parse.urlencode(
            {
                "dataset": AI_DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": min(50, count - len(rows)),
            }
        )
        payload = fetch_json(f"{DATASET_SERVER}?{query}")
        source_rows = payload.get("rows", [])
        if not source_rows:
            raise RuntimeError("The Hugging Face dataset returned no more rows.")
        for item in source_rows:
            row = item["row"]
            track_id = row["__key__"]
            cached_audio = row["mp3"][0]["src"]
            rows.append(
                {
                    "track_id": track_id,
                    "label": "ai_generated",
                    "audio_url": cached_audio,
                    "source": f"https://huggingface.co/datasets/{AI_DATASET}",
                    "license": "See dataset card and Suno source terms; local research use only",
                    "source_revision": AI_REVISION,
                }
            )
        offset += len(source_rows)
    append_rows(rows[:count])


def build_human_manifest(count: int, start_offset: int = 0) -> None:
    client_id = os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        raise RuntimeError(
            "Set JAMENDO_CLIENT_ID before building the MTG-Jamendo manifest. "
            "Create an API client through Jamendo's developer portal."
        )
    rows = []
    seen_ids: set[str] = set()
    for order in ("album_name", "artist_name", "relevance"):
        offset = start_offset
        empty_pages = 0
        while len(rows) < count:
            query = urllib.parse.urlencode(
                {
                    "client_id": client_id,
                    "format": "json",
                    "limit": min(50, count - len(rows)),
                    "offset": offset,
                    "order": order,
                    "audiodownload_allowed": "true",
                    "audioformat": "mp32",
                }
            )
            payload = fetch_json(f"https://api.jamendo.com/v3.0/tracks/?{query}")
            tracks = payload.get("results", [])
            if not tracks:
                empty_pages += 1
                if empty_pages >= 3:
                    break
                offset += 50
                continue
            empty_pages = 0
            for track in tracks:
                track_id = str(track["id"])
                if track_id in seen_ids:
                    continue
                seen_ids.add(track_id)
                rows.append(
                    {
                        "track_id": f"jamendo_{track_id}",
                        "label": "human",
                        "audio_url": track["audio"],
                        "source": track.get("shareurl", "https://www.jamendo.com/"),
                        "license": track.get("license_ccurl", "Creative Commons; verify the track page"),
                        "source_revision": "Jamendo API response",
                    }
                )
                if len(rows) == count:
                    break
            offset += len(tracks)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"Jamendo returned {len(rows)} tracks; expected {count}.")
    append_rows(rows)


def download_row(row: dict[str, str], jamendo_client_id: str | None) -> dict[str, str]:
    output_dir = ROOT / "data" / "raw" / row["label"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{row['track_id']}.mp3"
    if not destination.exists():
        audio_url = row["audio_url"]
        if row["label"] == "human":
            if not jamendo_client_id:
                raise RuntimeError("Set JAMENDO_CLIENT_ID before downloading human tracks.")
            separator = "&" if "?" in audio_url else "?"
            audio_url = f"{audio_url}{separator}{urllib.parse.urlencode({'client_id': jamendo_client_id})}"
        request = urllib.request.Request(audio_url, headers={"User-Agent": "ai-music-detector/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response, destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    row["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return row


def download_manifest() -> None:
    if not MANIFEST.exists():
        raise RuntimeError(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    jamendo_client_id = os.environ.get("JAMENDO_CLIENT_ID")
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(download_row, row, jamendo_client_id) for row in rows]
        completed = []
        for row, future in zip(rows, futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                row["download_error"] = str(exc)
                completed.append(row)
    fields = list(completed[0])
    if "sha256" not in fields:
        fields.append("sha256")
    if "download_error" not in fields:
        fields.append("download_error")
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=int, default=0, help="Build this many Jamendo rows.")
    parser.add_argument("--human-offset", type=int, default=0, help="Start offset for Jamendo rows.")
    parser.add_argument("--ai", type=int, default=0, help="Build this many Hugging Face rows.")
    parser.add_argument("--download", action="store_true", help="Download all manifest rows.")
    args = parser.parse_args()
    if args.human:
        build_human_manifest(args.human, args.human_offset)
    if args.ai:
        build_ai_manifest(args.ai)
    if args.download:
        download_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
