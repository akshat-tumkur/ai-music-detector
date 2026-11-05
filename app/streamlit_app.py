"""Streamlit dashboard for AI-generated music detection."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import predict_audio


MODEL_PATH = ROOT / "models" / "music_detector_pipeline.joblib"
SUPPORTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif"]


@st.cache_resource
def load_model_bundle() -> dict[str, object]:
    """Load the persisted model once per Streamlit process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _save_upload(uploaded_file: BinaryIO) -> Path:
    """Save an uploaded audio object to a temporary file for Librosa."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".audio"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(uploaded_file.getvalue())
    finally:
        temporary.close()
    return Path(temporary.name)


def _predict_upload(uploaded_file: BinaryIO) -> dict[str, object]:
    """Predict one uploaded file and remove its temporary copy."""
    temporary_path = _save_upload(uploaded_file)
    try:
        return predict_audio(temporary_path, MODEL_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def _window_table(result: dict[str, object]) -> pd.DataFrame:
    """Convert inference window details into a display-friendly table."""
    table = pd.DataFrame(result["windows"])
    table["predicted_label"] = table["ai_probability"].ge(0.5).map(
        {True: "ai_generated", False: "human"}
    )
    table["confidence"] = table["ai_probability"].where(
        table["ai_probability"].ge(0.5),
        1.0 - table["ai_probability"],
    )
    return table


def _feature_importance_table(bundle: dict[str, object]) -> pd.DataFrame:
    """Return global XGBoost feature importance for the saved model."""
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    xgboost_pipeline = model.named_estimators_["xgboost"]
    classifier = xgboost_pipeline.named_steps["classifier"]
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )


def _show_prediction(filename: str, result: dict[str, object]) -> None:
    """Render the aggregate and per-window result for one audio file."""
    ai_probability = float(result["ai_probability"])
    label = str(result["predicted_label"])
    st.subheader(filename)
    first, second, third = st.columns(3)
    first.metric("Prediction", label)
    second.metric("AI probability", f"{ai_probability:.1%}")
    third.metric("Windows analyzed", int(result["window_count"]))

    windows = _window_table(result)
    chart = px.line(
        windows,
        x="start_seconds",
        y="ai_probability",
        markers=True,
        range_y=[0, 1],
        labels={"start_seconds": "Window start (seconds)", "ai_probability": "AI probability"},
        title="AI probability by five-second window",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(windows, use_container_width=True, hide_index=True)


st.set_page_config(page_title="AI Music Detector", page_icon="🎵", layout="wide")
st.title("AI-Generated Music Detector")
st.caption("Five-second sliding-window inference with a XGBoost, SVM, and Random Forest ensemble.")

try:
    model_bundle = load_model_bundle()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

single_tab, batch_tab, explain_tab = st.tabs(["Single prediction", "Batch predictions", "Feature importance"])

with single_tab:
    uploaded_file = st.file_uploader(
        "Upload one audio file",
        type=SUPPORTED_TYPES,
        key="single_audio",
    )
    if uploaded_file is not None:
        try:
            _show_prediction(uploaded_file.name, _predict_upload(uploaded_file))
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not analyze {uploaded_file.name}: {error}")

with batch_tab:
    uploaded_files = st.file_uploader(
        "Upload multiple audio files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="batch_audio",
    )
    if uploaded_files:
        rows: list[dict[str, object]] = []
        for uploaded_file in uploaded_files:
            try:
                result = _predict_upload(uploaded_file)
                rows.append(
                    {
                        "file": uploaded_file.name,
                        "predicted_label": result["predicted_label"],
                        "ai_probability": result["ai_probability"],
                        "human_probability": result["human_probability"],
                        "window_count": result["window_count"],
                        "duration_seconds": result["duration_seconds"],
                    }
                )
            except (OSError, ValueError, RuntimeError) as error:
                st.warning(f"Skipped {uploaded_file.name}: {error}")
        if rows:
            batch_table = pd.DataFrame(rows).sort_values("ai_probability", ascending=False)
            st.dataframe(batch_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch results as CSV",
                batch_table.to_csv(index=False).encode("utf-8"),
                "music_predictions.csv",
                "text/csv",
            )

with explain_tab:
    st.write("These are global feature importances from the XGBoost component of the saved ensemble.")
    importance = _feature_importance_table(model_bundle)
    st.plotly_chart(
        px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 XGBoost feature importances",
        ),
        use_container_width=True,
    )
    st.dataframe(importance, use_container_width=True, hide_index=True)
