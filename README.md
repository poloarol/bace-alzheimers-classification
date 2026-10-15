# BACE-1 Molecular Activity Classification.

Classification of small molecules to determine whether they inhibit
beta-secretase 1 (BACE1).

## Question

Can we identify when a GNN's molecular activity prediction is reliable, quantify the uncertainty of that prediction, and explain which molecular substructures contributed to it?

```
Experiment	Main question
GCN vs. GAT vs. GATGraphTransformer	Which architecture discriminates best?
Single model vs. ensemble	Does ensemble disagreement help identify errors?
Raw vs. calibrated probabilities	Are the probabilities more reliable?
Standard classification vs. conformal prediction	What coverage and prediction-set size can be achieved?
Seen vs. unseen scaffolds	How robust are the methods to molecular novelty?
XAI across models and seeds	Are the explanations stable and informative?

```

## What was implemented

- BACE molecule graph processing with DeepChem and PyTorch Geometric.
- Scaffold-aware, stratified train/test splitting.
- Morgan fingerprint generation for classical machine-learning models.
- Baseline classifiers:
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
- Graph convolutional neural network (GCN) training.
- Graph attention network (GAT) training.
- GAT combined with graph Transformer convolution training.
- MLflow tracking for preprocessing and model experiments.
- Test-set inference for scikit-learn and PyTorch models.
- Confusion-matrix export as both PNG and TXT.
- A pipeline runner for complete or individual workflow stages.

## Project scaffolding

```text
.
├── app.py                         # Pipeline runner
├── requirements.txt               # Python dependencies
├── README.md
├── data/
│   └── MoleculeNet/bace/
│       ├── raw/                   # bace.csv and bace_processed.csv
│       ├── clean/                 # train_test_split.npz
│       ├── processed/             # PyG graph files
│       ├── fingerprints/          # Generated fingerprint CSV files
│       └── output/                # Confusion matrices and inference output
├── src/
│   ├── dataset.py                 # MoleculeDataset
│   ├── model.py                   # GCN, GAT, and hybrid definitions
│   ├── train.py                   # GNN training and metrics
│   ├── baseline_model.py          # Classical model training
│   ├── build_fingerprints.py      # Morgan fingerprints
│   ├── stratified_scaffold_split.py
│   ├── inference.py               # Test-set inference
│   └── mlflow_utils.py            # Shared MLflow configuration
└── mlflow.db                      # Local MLflow tracking database
```

The graph model factory currently supports `GCN`, `GAT`, and
`GATGraphTransformer`. The GAT models use the molecule edge features produced
by `MoleculeDataset`; the hybrid applies GAT message passing followed by
Transformer convolution layers.

Generated data, processed graphs, models, and MLflow database files are local
artifacts and should not be committed.

## Installation

The project uses Python and the dependencies listed in
[requirements.txt](requirements.txt).

```powershell
pip install -r requirements.txt
```

The repository's existing environment also includes the scientific,
cheminformatics, PyTorch, and PyTorch Geometric dependencies required by the
source scripts.

## MLflow tracking

Runs use SQLite at `mlflow.db` by default. To inspect experiments locally:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the displayed local MLflow URL in a browser. To use a shared
tracking server, set `MLFLOW_TRACKING_URI` or pass
`--mlflow-tracking-uri` to the relevant command.

The default experiments are:

| Experiment | Contents |
|---|---|
| `bace-preprocessing` | Scaffold splits and Morgan fingerprints |
| `bace-baseline` | Random Forest, Logistic Regression, and Gradient Boosting |
| `bace-gnn` | GCN, GAT, and GATGraphTransformer training, metrics, and logged models |

## Results

The GNN training workflow reports the following held-out test metrics from the
best restored checkpoint:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.885628851032983 |
| PR-AUC | 0.8915840796588492 |
| Balanced accuracy | 0.8178144255164914 |
| Inhibitor F1 | 0.8242074927953891 |

The same metrics are logged to MLflow using the exact metric names above.
Training also logs accuracy by epoch and saves each PyTorch model under its
model-specific artifact path (`GCN-Model`, `GAT-Model`, or
`GATGraphTransformer-Model`). Local state-dict checkpoints default to
`model\GCN.pth`, `model\GAT.pth`, and `model\GATGraphTransformer.pth`.

Baseline evaluation logs ROC-AUC, PR-AUC, balanced accuracy, and inhibitor F1
for each estimator. Baseline confusion matrices are written under
`data/MoleculeNet/bace/output/<model-type>/`.

## Run individual pipeline steps

The root-level runner uses `data/MoleculeNet/bace` by default. Override it
with `--dataset-root` when using another dataset location.

### 1. Create the scaffold split

```powershell
python app.py split
```

This creates:

```text
data/MoleculeNet/bace/clean/train_test_split.npz
```

Optional parameters include `--test-size` and `--n-trials`.

### 2. Build Morgan fingerprints

```powershell
python app.py fingerprints
```

This creates train/test fingerprint and label CSV files under:

```text
data/MoleculeNet/bace/fingerprints/
```

Use `--radius` and `--n-bits` to change the fingerprint configuration.

### 3. Train the baseline models

```powershell
python app.py baseline
```

This trains all three scikit-learn estimators and writes their confusion
matrices under:

```text
data/MoleculeNet/bace/output/
```

### 4. Train the GNN

```powershell
python app.py gnn --model GCN --epochs 100
python app.py gnn --model GAT --epochs 100
python app.py gnn --model GATGraphTransformer --epochs 100 --batch-size 32
```

The GNN command logs epoch-level metrics and the final test metrics to MLflow.
Use `--hidden-channels` and `--learning-rate` to change model settings.

### 5. Run inference

Inference requires an MLflow model URI. For a baseline model:

```powershell
python app.py inference `
  --model-uri "runs:/<run-id>/model" `
  --model-flavor sklearn `
  --model-type RandomForestClassifier
```

For a GNN:

```powershell
python app.py inference `
  --model-uri "runs:/<run-id>/GAT-Model" `
  --model-flavor pytorch `
  --model-type GAT
```

Inference loads the held-out test set and writes:

```text
data/MoleculeNet/bace/output/<model-type>/confusion_matrix.png
data/MoleculeNet/bace/output/<model-type>/confusion_matrix.txt
```

The TXT file uses rows for actual classes and columns for predicted classes,
with class order `0, 1`.

## Run the complete pipeline

Run split creation, fingerprint generation, baseline training, and one selected
GNN architecture in order:

```
python app.py all
```

The complete workflow can be configured with the same options exposed by the
individual stages:

```powershell
python app.py all `
  --test-size 0.2 `
  --n-trials 100 `
  --radius 2 `
  --n-bits 2048 `
  --epochs 100 `
  --model GAT `
  --hidden-channels 64 `
  --learning-rate 0.01 `
  --batch-size 32
```

The `--model` option selects the GNN architecture for the `all` command:

```powershell
python app.py all --model GCN
python app.py all --model GAT
python app.py all --model GATGraphTransformer
```

Use `--skip-gnn` to run only preprocessing and baseline training. Use
`--dry-run` to print every command without executing it:

```powershell
python app.py all --dry-run
```

Individual source scripts remain available if direct script-level options are
needed.

## Snakemake pipeline

The [Snakefile](Snakefile) provides a reproducible workflow for:

1. Scaffold split generation.
2. Morgan fingerprint generation.
3. Baseline model training.
4. Training each configured graph model (`GCN`, `GAT`, and
   `GATGraphTransformer`).

The workflow is configured in [config.yaml](config.yaml). Install Snakemake
from [requirements.txt](requirements.txt), inspect the planned DAG, and then
run it with:

```powershell
pip install -r requirements.txt
snakemake --cores 1 -n
snakemake --cores 1
```

The resulting graph checkpoints are written to:

```text
model/GCN.pth
model/GAT.pth
model/GATGraphTransformer.pth
```

Baseline confusion matrices are written under
`data/MoleculeNet/bace/output/`. Snakemake does not infer an MLflow model URI
for the subsequent inference step; after selecting a run in MLflow, use the
individual `inference` command documented above.

## Next Steps:

01. Correct Experimental Protocol:
   Introduce training, validation, calibration, and test roles. Stop selecting checkpoints based on test accuracy. Save fixed split indices and establish a reproducible baseline.

02. Calibrate the baseline test:
   Implement temperature scaling, Brier score, negative log-likelihood, ECE, and reliability diagrams. This creates the foundation for trustworthy probabilities.

03. Add conformal preditions:
   Implement split conformal classification and evaluate coverage, set size, singleton rate, and coverage across scaffold novelty.

04. Add ensemble uncertainty:
   Train multiple seeds and quantify predictive uncertainty, model disagreement, and molecular novelty. Test whether these measures identify errors and unreliable predictions.

05. Add molecular explanation:
   Generate GNNExplainer and atom-level attribution visualizations. Evaluate fidelity and stability rather than relying on visual appeal alone.


