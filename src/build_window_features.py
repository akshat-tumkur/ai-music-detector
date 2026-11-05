"""Build five-second window features from the dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_extraction import DEFAULT_SAMPLE_RATE, extract_features_from_samples


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "window_features.csv"
METADATA_PATH = ROOT / "data" / "processed" / "window_feature_metadata.json"
WINDOW_SECONDS = 5.0
HOP_SECONDS = 2.5


def _starts(sample_count: int, window_size: int, hop_size: int) -> list[int]:
    if sample_count <= window_size:
        return [0]
    starts = list(range(0, sample_count - window_size + 1, hop_size))
    final_start = sample_count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _extract_track_windows(
    row: dict[str, str],
    window_size: int,
    hop_size: int,
) -> tuple[list[dict[str, object]], str | None]:
    audio_path = ROOT / "data" / "raw" / row["label"] / f"{row['track_id']}.mp3"
    try:
        y, sample_rate = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
        if y.size == 0:
            raise ValueError("audio contains no samples")
        records: list[dict[str, object]] = []
        for start in _starts(len(y), window_size, hop_size):
            window = y[start:start + window_size]
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            features = extract_features_from_samples(window, sample_rate, fast_tempo=True)
            records.append(
                {
                    "track_id": row["track_id"],
                    "label": row["label"],
                    "window_start_seconds": start / sample_rate,
                    **features,
                }
            )
        return records, None
    except (OSError, ValueError, RuntimeError) as exc:
        return [], f"{row['track_id']}: {exc}"


def build_window_feature_table(
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Extract one feature row per five-second window and save it."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")

    window_size = round(WINDOW_SECONDS * DEFAULT_SAMPLE_RATE)
    hop_size = round(HOP_SECONDS * DEFAULT_SAMPLE_RATE)
    records: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_extract_track_windows, row, window_size, hop_size)
            for row in manifest
        ]
        for future in tqdm(futures, desc="Extracting five-second windows"):
            track_records, failure = future.result()
            records.extend(track_records)
            if failure:
                failures.append(failure)

    if failures:
        raise RuntimeError(f"Feature extraction failed for {len(failures)} files:\n" + "\n".join(failures[:10]))
    table = pd.DataFrame(records)
    if table.empty or table.isna().any().any():
        raise RuntimeError("No valid window feature rows were produced.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "window_seconds": WINDOW_SECONDS,
                "hop_seconds": HOP_SECONDS,
                "track_count": int(table["track_id"].nunique()),
                "window_count": len(table),
                "feature_count": len(table.columns) - 3,
                "feature_columns": list(table.columns[3:]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    args = parser.parse_args()
    table = build_window_feature_table(args.manifest, args.output, args.metadata)
    print(f"Saved {len(table)} windows from {table['track_id'].nunique()} tracks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
