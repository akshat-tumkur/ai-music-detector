# Jupyter notebooks

Open these notebooks from the repository root after activating `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

If the `jupyter` command is unavailable, install it inside the virtual
environment:

```powershell
python -m pip install notebook
```

## Notebook order

1. `01_dataset_exploration.ipynb`
   - Inspect the manifest and class balance.
   - Check which local audio files exist.
   - View a sample waveform.
2. `02_feature_exploration.ipynb`
   - Inspect the 184 extracted features.
   - Compare human and AI-generated feature distributions.
   - View summary differences and correlations.
3. `03_model_training_and_evaluation.ipynb`
   - Load the saved ensemble and metrics.
   - Inspect probabilities and predictions in a DataFrame.
   - View a confusion matrix and ROC curve.

The notebooks read generated files from `data/processed/` and the saved model
from `models/`; run dataset acquisition, feature extraction, and model
training first if those files do not exist.
