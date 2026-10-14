# bace-alzheimers-classification
Classification task to determine whether a small molecule would inhibit beta-secretase 1.

## MLflow tracking

The preprocessing and model scripts create local MLflow runs in `mlflow.db` by
default. Install the project dependency and launch the MLflow UI with:

```powershell
pip install -r requirements.txt
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Use `--mlflow-tracking-uri` or the `MLFLOW_TRACKING_URI` environment variable
to send runs to a shared MLflow server. Each script also accepts
`--mlflow-experiment` to select the experiment.

## Inference

Run inference with an MLflow model URI. Results are written under the supplied
model type and include `confusion_matrix.png` and `confusion_matrix.txt`.

```powershell
python src/inference.py `
  --model-uri "runs:/<run-id>/model" `
  --model-flavor sklearn `
  --model-type RandomForestClassifier `
  --fingerprint-dir data/MoleculeNet/bace/fingerprints
```

For the GNN logged by `train.py`, use `--model-flavor pytorch`,
`--dataset-root data/MoleculeNet/bace`, and the model's logged artifact path
(for example, `runs:/<run-id>/GCN-Model`).
