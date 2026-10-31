from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from src.feature_extraction import feature_names
from src.inference import predict_audio


class AlwaysAiModel:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        assert values.shape == (1, 184)
        return np.array([[0.1, 0.9]])


def _create_model(path: Path) -> None:
    joblib.dump({"model": AlwaysAiModel(), "feature_columns": feature_names()}, path)


def test_short_audio_is_padded_to_one_window(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"
    model_path = tmp_path / "model.joblib"
    sf.write(audio_path, np.zeros(22_050), 22_050)
    _create_model(model_path)

    result = predict_audio(audio_path, model_path)

    assert result["window_count"] == 1
    assert result["predicted_label"] == "ai_generated"
    assert result["ai_probability"] == 0.9


def test_exact_five_second_audio_uses_one_window(tmp_path: Path) -> None:
    audio_path = tmp_path / "exact.wav"
    model_path = tmp_path / "model.joblib"
    sf.write(audio_path, np.zeros(5 * 22_050), 22_050)
    _create_model(model_path)

    result = predict_audio(audio_path, model_path)

    assert result["window_count"] == 1
    assert result["windows"][0]["start_seconds"] == 0.0


def test_variable_length_audio_uses_overlapping_windows(tmp_path: Path) -> None:
    audio_path = tmp_path / "long.wav"
    model_path = tmp_path / "model.joblib"
    sf.write(audio_path, np.zeros(8 * 22_050), 22_050)
    _create_model(model_path)

    result = predict_audio(audio_path, model_path)

    assert result["window_count"] == 3
    assert len(result["windows"]) == 3
