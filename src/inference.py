"""Five-second sliding-window inference for variable-length audio."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import librosa
import numpy as np

from src.feature_extraction import DEFAULT_SAMPLE_RATE, extract_features_from_samples


DEFAULT_WINDOW_SECONDS: Final = 5.0
DEFAULT_HOP_SECONDS: Final = 2.5


def _window_starts(sample_count: int, window_size: int, hop_size: int) -> list[int]:
    """Return starts that cover the beginning and end of an audio recording."""
    if sample_count <= window_size:
        return [0]
    starts = list(range(0, sample_count - window_size + 1, hop_size))
    final_start = sample_count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_audio(
    audio_path: str | Path,
    model_path: str | Path,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> dict[str, object]:
    """Predict an audio file and return aggregate and per-window probabilities."""
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive.")
    if hop_seconds > window_seconds:
        raise ValueError("hop_seconds cannot be greater than window_seconds.")

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    y, sample_rate = librosa.load(path, sr=DEFAULT_SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file contains no samples: {path}")

    window_size = round(window_seconds * sample_rate)
    hop_size = round(hop_seconds * sample_rate)
    probabilities: list[float] = []
    details: list[dict[str, float]] = []
    for start in _window_starts(len(y), window_size, hop_size):
        window = y[start:start + window_size]
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))
        features = extract_features_from_samples(window, sample_rate)
        vector = np.array([[features[name] for name in feature_columns]], dtype=np.float64)
        probability = float(model.predict_proba(vector)[0, 1])
        probabilities.append(probability)
        details.append({
            "start_seconds": start / sample_rate,
            "end_seconds": min((start + window_size) / sample_rate, len(y) / sample_rate),
            "ai_probability": probability,
        })

    aggregate_probability = float(np.mean(probabilities))
    return {
        "audio_path": str(path),
        "duration_seconds": len(y) / sample_rate,
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "window_count": len(details),
        "ai_probability": aggregate_probability,
        "human_probability": 1.0 - aggregate_probability,
        "predicted_label": "ai_generated" if aggregate_probability >= 0.5 else "human",
        "windows": details,
    }
