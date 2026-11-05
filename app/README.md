# Streamlit dashboard

From the repository root, activate the virtual environment and start the app:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app\streamlit_app.py
```

The trained model must exist at
`models/music_detector_pipeline.joblib`. The dashboard accepts WAV, MP3,
FLAC, OGG, M4A, AIFF, and AIF files. Uploaded files are processed through the
five-second sliding-window inference pipeline and removed after prediction.

The dashboard provides:

- Single-file prediction with aggregate and per-window probabilities.
- Batch prediction with downloadable CSV results.
- Global XGBoost feature importance for the saved ensemble.
