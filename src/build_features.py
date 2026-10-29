"""Build a CSV feature table from the downloaded dataset manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .feature_extraction import DEFAULT_SAMPLE_RATE, extract_features


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "features.csv"
METADATA_PATH = ROOT / "data" / "processed" / "feature_metadata.json"


def build_feature_table(
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
    metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Extract features for every downloaded row and save the table."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")

    records: list[dict[str, object]] = []
    failures: list[str] = []
    for row in tqdm(manifest, desc="Extracting features"):
        audio_path = ROOT / "data" / "raw" / row["label"] / f"{row['track_id']}.mp3"
        try:
            features = extract_features(audio_path)
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append(f"{row['track_id']}: {exc}")
            continue
        records.append({"track_id": row["track_id"], "label": row["label"], **features})

    if failures:
        error_text = "\n".join(failures[:10])
        raise RuntimeError(f"Feature extraction failed for {len(failures)} files:\n{error_text}")
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("No feature rows were produced.")
    if table.isna().any().any():
        raise RuntimeError("Feature table contains missing values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    metadata = {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "track_count": len(table),
        "feature_count": len(table.columns) - 2,
        "feature_columns": list(table.columns[2:]),
        "source_manifest": str(manifest_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    args = parser.parse_args()
    table = build_feature_table(args.manifest, args.output, args.metadata)
    print(f"Saved {len(table)} rows and {len(table.columns) - 2} features to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
