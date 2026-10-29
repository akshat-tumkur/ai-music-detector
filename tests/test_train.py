import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import train_and_evaluate


def test_training_saves_model_and_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    human = rng.normal(0, 1, size=(20, 4))
    ai = rng.normal(2, 1, size=(20, 4))
    table = pd.DataFrame(
        np.vstack([human, ai]),
        columns=["feature_1", "feature_2", "feature_3", "feature_4"],
    )
    table.insert(0, "label", ["human"] * 20 + ["ai_generated"] * 20)
    table.insert(0, "track_id", [f"track_{i}" for i in range(40)])
    features_path = tmp_path / "features.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    table.to_csv(features_path, index=False)

    metrics = train_and_evaluate(features_path, model_path, metrics_path)

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["feature_count"] == 4
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    assert json.loads(metrics_path.read_text())["test_samples"] == 8
