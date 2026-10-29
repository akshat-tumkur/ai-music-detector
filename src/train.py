"""Train and evaluate the AI-music detection ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "models" / "music_detector_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "training_metrics.json"
RANDOM_STATE: Final = 42


def build_ensemble(n_features: int) -> VotingClassifier:
    """Create the standardized soft-voting ensemble."""
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=2.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]
    )
    random_forest = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )
    return VotingClassifier(
        estimators=[
            ("xgboost", Pipeline([("scaler", StandardScaler()), ("classifier", xgb)])),
            ("svm", svm),
            ("random_forest", random_forest),
        ],
        voting="soft",
        weights=[2, 1, 1],
        n_jobs=-1,
    )


def train_and_evaluate(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    """Train the ensemble, evaluate held-out tracks, and save artifacts."""
    table = pd.read_csv(features_path)
    if "label" not in table or "track_id" not in table:
        raise ValueError("Feature table must contain track_id and label columns.")
    if table["label"].isna().any() or table["track_id"].duplicated().any():
        raise ValueError("Feature table contains missing labels or duplicate tracks.")

    feature_columns = [column for column in table.columns if column not in {"track_id", "label"}]
    X = table[feature_columns].to_numpy(dtype=np.float64)
    y = (table["label"] == "ai_generated").astype(np.int8).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training requires both human and ai_generated labels.")
    if not np.isfinite(X).all():
        raise ValueError("Feature table contains non-finite values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = build_ensemble(X.shape[1])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    metrics: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "feature_count": len(feature_columns),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, predictions, target_names=["human", "ai_generated"], output_dict=True
        ),
        "feature_columns": feature_columns,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "label_mapping": {"human": 0, "ai_generated": 1},
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    metrics = train_and_evaluate(args.features, args.model, args.metrics)
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"Saved model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
