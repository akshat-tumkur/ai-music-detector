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


def build_human_manifest(count: int) -> None:
    client_id = os.environ.get("JAMENDO_CLIENT_ID")
    if not client_id:
        raise RuntimeError(
            "Set JAMENDO_CLIENT_ID before building the MTG-Jamendo manifest. "
            "Create an API client through Jamendo's developer portal."
        )
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "format": "json",
            "limit": count,
            "offset": 0,
            "audioformat": "mp32",
            "include": "licenses",
            "order": "popularity_total",
        }
    )
    payload = fetch_json(f"https://api.jamendo.com/v3.0/tracks/?{query}")
    rows = []
    for track in payload.get("results", []):
        rows.append(
            {
                "track_id": f"jamendo_{track['id']}",
                "label": "human",
                "audio_url": track["audio"],
                "source": track.get("shareurl", "https://www.jamendo.com/"),
                "license": track.get("license_ccurl", "Creative Commons; see track metadata"),
                "source_revision": "Jamendo API response",
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"Jamendo returned {len(rows)} tracks; expected {count}.")
    append_rows(rows)


def download_manifest() -> None:
    if not MANIFEST.exists():
        raise RuntimeError(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        output_dir = ROOT / "data" / "raw" / row["label"]
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{row['track_id']}.mp3"
        if destination.exists():
            continue
        request = urllib.request.Request(
            row["audio_url"], headers={"User-Agent": "ai-music-detector/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        row["sha256"] = digest
    fields = list(rows[0])
    if "sha256" not in fields:
        fields.append("sha256")
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=int, default=0, help="Build this many Jamendo rows.")
    parser.add_argument("--ai", type=int, default=0, help="Build this many Hugging Face rows.")
    parser.add_argument("--download", action="store_true", help="Download all manifest rows.")
    args = parser.parse_args()
    if args.human:
        build_human_manifest(args.human)
    if args.ai:
        build_ai_manifest(args.ai)
    if args.download:
        download_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
