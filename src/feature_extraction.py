"""Librosa feature extraction for one audio file."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import librosa
import numpy as np


DEFAULT_SAMPLE_RATE: Final = 22_050
DEFAULT_DURATION: float | None = 30.0
N_FFT: Final = 2_048
HOP_LENGTH: Final = 512
N_MFCC: Final = 20
N_MELS: Final = 128
N_CHROMA: Final = 12


def _statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return stable summary statistics for a frame-level feature matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    features: dict[str, float] = {}
    for index, row in enumerate(array):
        features[f"{prefix}_{index + 1}_mean"] = float(np.mean(row))
        features[f"{prefix}_{index + 1}_std"] = float(np.std(row))
        features[f"{prefix}_{index + 1}_min"] = float(np.min(row))
        features[f"{prefix}_{index + 1}_max"] = float(np.max(row))
    return features


def extract_features(
    audio_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float | None = DEFAULT_DURATION,
) -> dict[str, float]:
    """Extract a fixed-length, approximately 200-feature vector from audio."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    y, sr = librosa.load(path, sr=sample_rate, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError(f"Audio file contains no samples: {path}")

    features: dict[str, float] = {}
    features.update(_statistics(librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    ), "mfcc"))
    features.update(_statistics(librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_chroma=N_CHROMA
    ), "chroma"))

    spectral_features = (
        ("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)),
        ("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)),
        ("spectral_contrast", librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)),
        ("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)),
        ("zero_crossing_rate", librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)),
        ("rms", librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)),
        ("spectral_flatness", librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)),
    )
    for name, values in spectral_features:
        features.update(_statistics(values, name))

    features["tempo"] = float(librosa.feature.tempo(y=y, sr=sr, hop_length=HOP_LENGTH)[0])
    features["duration_seconds"] = float(librosa.get_duration(y=y, sr=sr))
    features["sample_rate"] = float(sr)
    features["rms_energy"] = float(np.mean(y**2))
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError(f"Feature extraction produced non-finite values: {path}")
    return features


def feature_names() -> list[str]:
    """Return feature names in the same order used by the extractor."""
    names: list[str] = []
    for prefix, count in (
        ("mfcc", N_MFCC),
        ("chroma", N_CHROMA),
        ("spectral_centroid", 1),
        ("spectral_bandwidth", 1),
        ("spectral_contrast", 7),
        ("spectral_rolloff", 1),
        ("zero_crossing_rate", 1),
        ("rms", 1),
        ("spectral_flatness", 1),
    ):
        for index in range(1, count + 1):
            names.extend(
                f"{prefix}_{index}_{statistic}"
                for statistic in ("mean", "std", "min", "max")
            )
    return names + ["tempo", "duration_seconds", "sample_rate", "rms_energy"]
