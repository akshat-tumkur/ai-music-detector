from pathlib import Path

import numpy as np
import soundfile as sf

from src.feature_extraction import extract_features, feature_names


def test_extractor_returns_more_than_100_finite_features(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    samples = np.sin(2 * np.pi * 440 * np.arange(22_050) / 22_050)
    sf.write(path, samples, 22_050)

    features = extract_features(path)

    assert len(features) > 100
    assert list(features) == feature_names()
    assert all(np.isfinite(value) for value in features.values())


def test_extractor_handles_short_audio(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(100), 22_050)

    features = extract_features(path)

    assert len(features) > 100


def test_extractor_rejects_missing_audio(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"

    try:
        extract_features(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Missing audio should raise FileNotFoundError")
