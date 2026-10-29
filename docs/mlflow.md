# Compare models with MLflow

The dashboard is local: http://localhost:5000. It is not the Vercel frontend or
the Hugging Face image backend. No model is automatically promoted by MLflow.
The JSON model registry remains the deployment authority.

For this workspace, open the imported experiment directly at
http://localhost:5000/#/experiments/1/runs. If you open the root dashboard instead,
select **Model training**, then **Pixel Landscape Training → Runs**. MLflow's
default GenAI overview will show no traces because these are image-model
training runs. Dismiss the first-use tip with **Got it** if it covers navigation.

Start or reopen the dashboard from the project root:

```powershell
python -m mlops.mlflow_server
```

Install `mlflow==3.12.0` if it is unavailable. The local SQLite database and
image artifacts are under ignored `artifacts/mlflow/`; they stay on this laptop.
Back up that directory separately if you want to preserve dashboard history.

## What is already imported

Experiment: **Pixel Landscape Training**

| Run | Epochs recorded | Best validation loss | Dataset |
|---|---:|---:|---|
| landscape-small-v1 | 44 | 0.597527 | Original |
| landscape-large-v1 | 26 | 0.598136 | Original |
| landscape-clean-candidate-v1 | 8 | 0.590874 | Mixed original/clean |

These are real historical records, not new training. Run timestamps reflect the
import date. `training_seconds_total` records the saved training duration.

In MLflow, select runs and choose **Compare**. Plot `train_loss`, `val_loss`,
`generalization_gap`, and `lr` against **Step** (the epoch). The gap is validation
loss minus training loss; it is a diagnostic, not an image quality metric.
Filter on parameter `dataset_manifest_sha256` before comparing validation loss.
The cleaner-data run uses a different validation set and cannot be ranked by
its lower loss against the other two.

Open a run's **Artifacts** to browse:

- `samples/`: native PNGs and preview grids.
- `evaluations/`: matched small/large comparison sheets and saved reports,
  available for the original small and larger models.
- `metadata/`: histories, tokenizer check when available, and registry provenance.

Exact training-token match counts are limited copying checks. They do not
establish absence of partial memorization. There are no invented feature-accuracy,
FID, or human-quality scores. Inspect the images for missing requested features
and malformed shapes; add a validated evaluation before using automated quality
thresholds to approve a release.

## Future training

Add **`--mlflow`** to `training/train_compositional.py` or its supervisor
`training/run_compositional_training.py`. Every saved epoch logs its metrics,
and supervisor completion uploads preview images. Without this flag, existing
training behavior stays unchanged and MLflow is not required.

Tracking reuses `runs/<run>/mlflow.json` on resume and skips repeated epoch
values. Immutable architecture, dataset and tokenizer parameters cannot change
inside the same tracked run. Exceptions mark the run failed; checkpoints are
saved before epoch tracking, so a tracking error does not corrupt saved weights.
Set `MLFLOW_TRACKING_URI` in your shell to use an external tracking server;
the local dashboard command always opens the local database.

Refresh imported metadata, samples and comparison reports without training:

```powershell
python -m mlops.import_mlflow
```

This reuses the existing run IDs and does not duplicate identical epoch metrics.
It only imports known JSON, PNG and Markdown files, never `.env`, tokens, or
weight binaries. Existing model releases and dataset versions remain in their
dedicated registry, Hugging Face and DVC workflows.
