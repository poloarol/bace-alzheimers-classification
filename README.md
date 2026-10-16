# BACE-1 Molecular Activity Classification

A molecular machine learning project comparing classical machine learning and graph neural networks for predicting whether a small molecule inhibits beta-secretase 1 (BACE-1).

The project combines molecular graph learning, cheminformatics, scaffold-aware evaluation, probability calibration, split conformal prediction, experiment tracking, and reproducible workflow automation. The conformal stage reports prediction sets and empirical coverage; future work will focus on explainability and broader uncertainty analysis.

- BACE molecule graph processing with DeepChem and PyTorch Geometric.
- Scaffold-aware, stratified train/validation/calibration/test splitting.
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
- Temperature scaling and split conformal prediction sets.
- Confusion-matrix export as both PNG and TXT.
- A pipeline runner for complete or individual workflow stages.

BACE-1 is an enzyme involved in amyloid-beta production and has been investigated as a therapeutic target in Alzheimer's disease research. This project addresses a molecular activity prediction task: given a molecule's structure, predict whether it inhibits BACE-1.

**This is a molecular property classification task, not a model for diagnosing Alzheimer's disease in patients.**

The project uses the BACE dataset from MoleculeNet, containing approximately 1,530 molecular observations.

## Objectives

- Compare classical machine learning models with graph neural networks for molecular activity classification.
- Represent molecules using Morgan fingerprints and graph-based representations.
- Evaluate generalization using scaffold-aware data splitting.
- Track experiments, metrics, and model artifacts with MLflow.
- Automate the modeling workflow using Snakemake.
- Establish a foundation for explainable and uncertainty-aware molecular predictions.

## Methods

### 1. Molecular representation

Molecules are processed using RDKit, DeepChem, and PyTorch Geometric.

Two representations are used:

- **Morgan fingerprints:** Fixed-length molecular descriptors for classical machine learning.
- **Molecular graphs:** Atoms represented as nodes and chemical bonds represented as edges, with atom and bond features used by graph neural networks.

### 2. Classical machine learning baselines

Three classifiers are implemented using Morgan fingerprints:

- Logistic Regression
- Random Forest
- Gradient Boosting

These models provide reference points for assessing whether graph-based representations offer an advantage over conventional molecular fingerprints.

### 3. Graph neural networks

The graph learning pipeline currently supports three architectures:

| Architecture | Description |
|---|---|
| GCN | Graph convolutional network for learning molecular representations through neighborhood aggregation. |
| GAT | Graph attention network that learns attention-weighted neighborhood representations. |
| GATGraphTransformer | Hybrid architecture combining graph attention with Transformer-style graph convolution. |

The graph models learn molecular representations from atom and bond features and produce binary activity predictions.

### 4. Evaluation metrics

The pipeline reports:

- ROC-AUC
- Precision–Recall AUC (PR-AUC)
- Balanced accuracy
- F1-score for the inhibitor class
- Confusion matrices

ROC-AUC and PR-AUC measure ranking performance, while balanced accuracy and inhibitor-class F1 provide complementary views of classification performance.

### 5. Reproducibility and experiment tracking

The project uses:

- **MLflow** to track experiments, metrics, and model artifacts.
- **Snakemake** to orchestrate preprocessing and model training.
- A configurable command-line runner (`app.py`) to execute individual stages or the complete workflow.

## Current results

The existing GNN training workflow reports the following held-out metrics:

| Metric | Reported score |
|---|---:|
| ROC-AUC | 0.886 |
| PR-AUC | 0.892 |
| Balanced accuracy | 0.818 |
| Inhibitor-class F1 | 0.824 |

These results provide an initial performance baseline for the graph learning pipeline.

**Evaluation caveat:** The table above was produced by an earlier training implementation in which test accuracy influenced checkpoint selection, so these numbers are preliminary and optimistic. The current code selects checkpoints on the validation split only and uses the test split for final evaluation; re-run the pipeline to regenerate the results.

The classical baselines are also implemented and evaluated by the pipeline. A consolidated comparison of the baseline models and all GNN architectures, using a consistent and independently evaluated split, remains a next step.

## Repository structure

```text
bace-alzheimers-classification/
├── app.py
├── config.yaml
├── requirements.txt
├── Snakefile
├── README.md
├── notebooks/
├── model/                       # checkpoints and <Model>.calibration.json
├── tests/
└── src/
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── baseline_model.py
    ├── build_fingerprints.py
    ├── stratified_scaffold_split.py
    ├── inference.py
    ├── calibration.py
    ├── conformal.py
    └── mlflow_utils.py
```

### Main components

| Component | Purpose |
|---|---|
| `dataset.py` | Molecular graph dataset processing. |
| `model.py` | GCN, GAT, and hybrid graph Transformer architectures. |
| `train.py` | GNN training and evaluation. |
| `baseline_model.py` | Classical fingerprint-based classifiers. |
| `build_fingerprints.py` | Morgan fingerprint generation. |
| `stratified_scaffold_split.py` | Scaffold-aware dataset splitting. |
| `inference.py` | Model inference, confusion matrices, and optional calibrated probabilities. |
| `calibration.py` | Temperature fitting, Brier/NLL/ECE metrics, reliability diagrams, and the calibration JSON/report CLI. |
| `conformal.py` | Split conformal prediction sets and coverage evaluation on held-out calibration and test predictions. |
| `mlflow_utils.py` | Shared MLflow configuration. |
| `app.py` | Command-line pipeline runner. |
| `Snakefile` | Workflow automation. |

Generated datasets, processed graphs, model checkpoints, and local MLflow database files are runtime artifacts and should not be committed to version control.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/poloarol/bace-alzheimers-classification.git
cd bace-alzheimers-classification

pip install -r requirements.txt
```

The project uses Python, PyTorch, PyTorch Geometric, DeepChem, RDKit, scikit-learn, MLflow, and Snakemake. Compatible versions of the scientific and graph-learning dependencies are required.

## Usage

The following commands correspond to the current pipeline runner.

### 1. Generate the scaffold-aware split

Then open the displayed local MLflow URL in a browser. To use a shared
tracking server, set `MLFLOW_TRACKING_URI` or pass
`--mlflow-tracking-uri` to the relevant command.

The default experiments are:

| Experiment | Contents |
|---|---|
| `bace-preprocessing` | Scaffold splits and Morgan fingerprints |
| `bace-baseline` | Random Forest, Logistic Regression, and Gradient Boosting |
| `bace-gnn` | GCN, GAT, and GATGraphTransformer training, metrics, and logged models |
| `bace-calibration` | Temperature-scaling metrics and artifacts |
| `bace-conformal` | Prediction-set coverage, size metrics, and artifacts |

## Results

The table below is a historical result from an earlier training implementation
in which test accuracy influenced checkpoint selection; it is optimistic and
is not reproduced by the current code, which no longer scores the test split
during training. Regenerate results by running the pipeline.

| Metric | Score |
|---|---:|
| ROC-AUC | 0.885628851032983 |
| PR-AUC | 0.8915840796588492 |
| Balanced accuracy | 0.8178144255164914 |
| Inhibitor F1 | 0.8242074927953891 |

The current GNN code keeps the checkpoint with the highest **validation
accuracy** (early stopping patience of 20 epochs). Training logs train and
validation metrics by epoch and `best_validation_accuracy`; it does not score
the calibration or test splits. Each PyTorch model is saved under
its model-specific artifact path (`GCN-Model`, `GAT-Model`, or
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

The existing workflow writes the split artifact to:

```text
data/MoleculeNet/bace/clean/train_test_split.npz
```

The archive contains `train_idx`, `validation_idx`, `calibration_idx`, and
`test_idx`. The default proportions are 70% train, 10% validation, 10%
calibration, and 10% test. Override them with `--train-size`,
`--validation-size`, `--calibration-size`, and `--test-size`; proportions must
sum to 1. Use `--n-trials` to change the scaffold-assignment search budget.
The split command verifies class representation, complete index coverage,
scaffold separation, and the expected four-array archive contract before
logging the split artifact.

### 2. Generate Morgan fingerprints

```bash
python app.py fingerprints
```

This creates train, validation, calibration, and test fingerprint and label
CSV files under:

```text
data/MoleculeNet/bace/fingerprints/
```

The directory contains `<split>_fingerprints.csv` and `<split>_labels.csv` for
each of `train`, `validation`, `calibration`, and `test`.

Use `--radius` and `--n-bits` to change the fingerprint configuration.

```bash
python app.py baseline
```

This trains the Logistic Regression, Random Forest, and Gradient Boosting baselines and scores them on the validation split. The best model by validation PR-AUC is selected; only that model's held-out predictions are then temperature-calibrated and used for conformal evaluation.

### 4. Train a graph neural network

Train the GCN:

```bash
python app.py gnn --model GCN --epochs 100
```

Train the GAT:

```bash
python app.py gnn --model GAT --epochs 100
```

Train the hybrid architecture:

```bash
python app.py gnn --model GATGraphTransformer \
  --epochs 100 \
  --batch-size 32
```

The runner also supports model configuration options such as hidden channels and learning rate.

### 5. Run inference

Inference uses a model URI from an MLflow run.

For a classical model:

```bash
python app.py inference \
  --model-uri "runs:/<run-id>/model" \
  --model-flavor sklearn \
  --model-type RandomForestClassifier
```

For a graph model:

```bash
python app.py inference \
  --model-uri "runs:/<run-id>/GAT-Model" \
  --model-flavor pytorch \
  --model-type GAT
```

Inference outputs include confusion matrices in PNG and TXT formats under:

```text
data/MoleculeNet/bace/output/
```

Replace `<run-id>` with the identifier of the relevant MLflow run.

### 6. Run the complete workflow

To train all supported GNN architectures with one command:

```powershell
python app.py gnn --all-models --epochs 100
```

The existing `gnn --model <name>` form remains supported for legacy usage.
Each GNN run loads the named `train`, `validation`, `calibration`, and `test`
graph splits from the split archive. Missing processed graph files are created
automatically under `data/MoleculeNet/bace/processed/`. The validation split
controls checkpoint selection and early stopping. The GNN command logs
epoch-level train and validation metrics and does not score the calibration or
test splits; it only writes their raw logits, and final test results come from
the calibration and conformal stages. Processed graph
cache names include a hash of the split archive, so changing the split causes
the affected datasets to be rebuilt automatically.
Use `--hidden-channels` and `--learning-rate` to change model settings.

### Probability calibration

Calibration uses temperature scaling: a single positive temperature `T` is fitted on the **validation** logits and labels by minimising NLL of `softmax(logits / T)` with LBFGS on `log T` (so `T > 0`; the fit must land in [0.01, 100] and not worsen validation NLL). Probabilities become `softmax(logits / T)`. Argmax predictions, ROC-AUC, and PR-AUC are unchanged; only probability quality can improve. The test set is never used to fit `T`, and the calibration split is not used by this stage.

Data roles in the current code:

| Split | Used for |
|---|---|
| Train | Fitting model parameters. |
| Validation | GNN checkpoint selection (best validation accuracy) and early stopping, baseline selection (validation PR-AUC), and fitting `T`. |
| Calibration | Used to fit the split-conformal nonconformity threshold; not used for temperature fitting or model selection. Baseline final metrics may also be reported on it. |
| Test | Final evaluation only: calibration metrics, bootstrap intervals, and conformal coverage/set-size metrics. |

Calibration is a separate downstream stage: training writes only the labelled raw logits, and `src/calibration.py` is the single place that fits `T`, evaluates it on test, and writes the calibration artifact. This lets you recalibrate without retraining.

- `python app.py gnn ...` trains, writes validation/calibration/test prediction CSVs, then runs temperature calibration and conformal evaluation for that model.
- `python app.py baseline` selects the baseline by validation PR-AUC, writes its validation/calibration/test prediction CSVs (log of `predict_proba` as logits), then runs temperature calibration and conformal evaluation for the selected model.
- Snakemake runs `calibrate_gnn` after each `train_gnn`.

Each temperature calibration compares uncalibrated and scaled probabilities on the same test samples using Brier score, negative log-likelihood (NLL), and expected calibration error (ECE). Do not select models, temperatures, or thresholds from test metrics. Validation also drives checkpoint selection, so the temperature fit is not fully independent of it. The conformal stage uses the separate calibration split to fit a finite-sample threshold and evaluates prediction-set coverage and size on test. Temperature scaling changes the probability inputs to conformal prediction but does not use calibration or test labels to fit its temperature.

Outputs:

```text
data/MoleculeNet/bace/output/predictions/<ModelName>/validation_predictions.csv   # label, logit_0, logit_1
data/MoleculeNet/bace/output/predictions/<ModelName>/calibration_predictions.csv
data/MoleculeNet/bace/output/predictions/<ModelName>/test_predictions.csv
data/MoleculeNet/bace/output/predictions/selected_baseline.json                   # baseline only
model/<ModelName>.calibration.json                                                # canonical artifact: T, model, checkpoint, class convention, counts, before/after metrics
data/MoleculeNet/bace/output/calibration/<ModelName>/reliability_diagram.png
data/MoleculeNet/bace/output/calibration/<ModelName>/reliability_table.csv
data/MoleculeNet/bace/output/calibration/<ModelName>/calibration_report.json     # machine-readable before/after metrics, counts, binning, bootstrap intervals
data/MoleculeNet/bace/output/conformal/<ModelName>/conformal_predictions.csv     # test labels, probabilities, predicted labels, prediction sets, set sizes
data/MoleculeNet/bace/output/conformal/<ModelName>/conformal_report.json         # coverage, set size, class/group summaries
```

Calibration runs also log to the MLflow experiment `bace-calibration` (run tagged with `model`; metrics `temperature` and `test_{uncalibrated,calibrated}_{brier,nll,ece}`; artifacts under `calibration/`). Conformal runs log coverage, threshold, and prediction-set metrics to `bace-conformal`, with report and per-sample predictions under `conformal/`. Training runs no longer carry downstream calibration/conformal metrics. Snakemake declares the calibration JSON, diagram, table and report as outputs of `calibrate_gnn`; run conformal evaluation separately with the command below.

The JSON has a `schema_version`, records `positive_class` (class 1 = BACE-1 inhibitor), and loading it with `expected_model` refuses a temperature fitted for a different model. MLflow no longer receives calibration metrics from training; the calibration stage logs them itself (below).

To recalibrate (for example with a different bin count) without retraining:

```bash
python app.py calibrate --model-type GCN            # or GAT, GATGraphTransformer
python app.py calibrate --model-type best-baseline  # the baseline selected by validation PR-AUC
```

This prints sample and class counts, `T`, the before/after Brier, NLL, and ECE, and bootstrap intervals for the calibrated-minus-uncalibrated differences. `--bins` must be a positive integer (default 10); `--bootstrap N` sets the number of resamples (default 1000, `0` disables); the CLI also accepts `--seed`. Use `--output-dir` and `--temperature-output` to change locations.

To calibrate arbitrary predictions instead, pass CSVs with a `label` column (0/1) plus either `logit_0`/`logit_1` or `probability`, the probability of class 1 (not an arbitrary confidence score). Inputs are checked for finite values and binary labels. The fit input must come from the validation split and the evaluation input from the test split: a file named after another split (for example `test_predictions.csv` passed as `--fit-predictions`) is rejected, and the two files must differ. The calibration JSON and report record `fit_split`, `fit_predictions_file`, `fit_counts`, `eval_split`, `eval_predictions_file` and `eval_counts`:

```bash
python src/calibration.py \
  --fit-predictions validation_predictions.csv \
  --eval-predictions test_predictions.csv \
  --model-name GCN
```

`--validation` and `--test` remain accepted as aliases for these two options.

```bash
python app.py calibrate \
  --validation-predictions validation_predictions.csv \
  --test-predictions test_predictions.csv
```

**Calibrated inference.** `app.py inference` automatically passes `model/<model-type>.calibration.json` to `src/inference.py` when it exists (or pass `--calibration-file`). The model name must match the artifact. Scaling is applied to logits (log-probabilities for sklearn), never by dividing probabilities by `T`, and `calibrated_predictions.csv` is written next to the confusion matrix. A calibrated probability is not a validated estimate of prospective experimental success.

### Split conformal prediction

After training, the baseline and GNN commands automatically run conformal evaluation after temperature calibration. The runner expects each model's calibration and test CSVs under `data/MoleculeNet/bace/output/predictions/<ModelName>/`. If the calibration CSV is missing for a GNN (for example, it was trained before calibration-prediction export was added), `app.py conformal --model-type <GNN>` regenerates that file from the matching local checkpoint in `model/<GNN>.pth`, then continues with conformal evaluation. This is checkpoint inference only: it does not retrain or replace the model, and it uses the configured dataset's existing `clean/train_test_split.npz` calibration partition. The checkpoint, source dataset, and split file must exist; if the test prediction CSV is missing, rerun training to recreate the held-out prediction files. Baseline calibration predictions are produced during baseline training and are not reconstructed by this GNN checkpoint recovery path.

For example, to evaluate a previously trained GAT whose calibration CSV is absent:

```bash
python app.py conformal --model-type GAT
```

The same command works for GCN, GATGraphTransformer, and `best-baseline` when their corresponding saved prediction files exist. Explicit CSV inputs remain available for custom workflows. The separate calibration split is used to fit the finite-sample threshold; the test split is only used to report empirical coverage, prediction-set size, and related metrics.

Run the stage again for an existing model:

```bash
python app.py conformal --model-type GCN
python app.py conformal --model-type best-baseline
```

`--model-type` locates calibration/test prediction CSVs and uses
`model/<model-type>.calibration.json` when present. For explicit inputs, pass
both prediction files; a temperature artifact can optionally be supplied with
`--temperature-file`:

```bash
python app.py conformal \
  --calibration-predictions calibration_predictions.csv \
  --test-predictions test_predictions.csv \
  --temperature-file model/GCN.calibration.json \
  --confidence 0.90
```

The inputs follow the same CSV contract as temperature calibration: a `label`
column plus `logit_0`/`logit_1` or a class-1 `probability` column. `--groups`
accepts an optional headerless CSV with one group value per test sample and
adds coverage and average-set-size summaries by group. Group summaries are
descriptive; the marginal coverage guarantee relies on calibration and test
samples being exchangeable and does not correct scaffold distribution shift.
The report and prediction-set CSV are written to
`data/MoleculeNet/bace/output/conformal/<ModelName>/` by default.

**Interpreting the results.** The test split holds roughly 150 molecules, so a 10-bin reliability diagram has sparsely populated bins that can shift with one or two molecules. The diagram therefore includes a histogram of samples per bin, and `reliability_table.csv` lists per-bin counts, mean probabilities, and observed fractions for both probability sets. Ten bins is a fixed baseline; the bin count should not be tuned to improve the plot. Read Brier score and NLL alongside ECE, and do not treat a small ECE difference as a meaningful improvement. **Uncertainty.** The report and console include 95% paired percentile-bootstrap intervals (1000 resamples by default) over test molecules for each metric and for the calibrated-minus-uncalibrated difference (`delta`, negative means better for Brier, NLL and ECE). `T` is held fixed, and the same resampled molecules score both probability sets. These intervals capture test-sample noise only: they ignore variance in the fitted `T` and the scaffold-level dependence between molecules, so they are optimistic. If a `delta` interval contains 0, there is no evidence of improvement. Brier score is the binary positive-class version (`mean((p - y)^2)`), half the multiclass sum-over-classes convention.

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

Run split creation, fingerprint generation, baseline training, and the selected
GNN architecture (GCN by default) in order:

```
python app.py all
```

Pass `--all-models` to train every supported GNN architecture in the complete
pipeline. Without it, the legacy single-model behavior is retained:

```powershell
python app.py all --all-models
```

The complete workflow can be configured with the same options exposed by the
individual stages:

```powershell
python app.py all `
  --train-size 0.7 `
  --validation-size 0.1 `
  --calibration-size 0.1 `
  --test-size 0.1 `
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
```

## Test suite

Run the complete unit and workflow test suite with:

```powershell
python -m unittest discover -s tests -v
```

The suite covers split-file keys, index coverage, class representation,
scaffold isolation, graph-cache invalidation, fingerprint generation, baseline
models, GNN model construction and forward passes, inference outputs, MLflow
utilities, CLI parsing, train/validation model selection behavior, and
calibration (temperature fitting, input validation, edge cases such as T = 1,
overconfident logits, invalid probabilities, single-class labels, empty or
mismatched inputs, invalid JSON, model-identity mismatches, bootstrap intervals,
artifact round trips, CLI end to end), and conformal prediction thresholds,
coverage, groups, CLI outputs, missing-GNN-calibration-file recovery, and
runner dispatch. `pytest` is not required; the suite uses `unittest`.

Other supported architectures include `GCN` and `GATGraphTransformer`.

To inspect the planned workflow without executing it:

```bash
python app.py all --dry-run
```

### 7. Run the Snakemake workflow

Preview the execution plan:

```bash
snakemake --cores 1 -n
```

Execute the workflow:

```bash
snakemake --cores 1
```

The workflow orchestrates scaffold splitting, fingerprint generation, baseline training, graph-model training, and per-GNN calibration (`calibrate_gnn`, which reads the saved validation/test logits and can be rerun without retraining) according to `config.yaml`. Baselines are calibrated through `python app.py calibrate --model-type best-baseline` because the winning baseline is only known after training. Snakemake paths are built with `os.path.join`; set `python` in `config.yaml` (or `--config python=...`) to the interpreter that has the project dependencies.

## MLflow experiment tracking

The project uses a local SQLite tracking database by default.

Launch the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the local URL printed by MLflow to inspect experiments and logged artifacts.

The default experiments are:

| Experiment | Contents |
|---|---|
| `bace-preprocessing` | Scaffold splitting and fingerprint generation. |
| `bace-baseline` | Classical fingerprint-based classifiers. |
| `bace-gnn` | GNN training, metrics, and model artifacts. |
| `bace-calibration` | Temperature-scaling metrics and artifacts. |
| `bace-conformal` | Prediction-set coverage, size metrics, and artifacts. |

Tracking settings can be configured using `MLFLOW_TRACKING_URI` or the supported command-line options.

## Limitations

Several limitations should be addressed before treating the current results as a reliable estimate of prospective molecular activity prediction.

**Validation reuse.** Checkpoints are selected on the validation split, which is also used to fit the calibration temperature, so the temperature is not fitted on fully independent data. The test split is used only for final evaluation. Reported results in this README predate that fix and should be regenerated.

**Limited dataset size.** With approximately 1,530 molecules, model performance estimates and calibration metrics may be sensitive to the composition of the data partitions.

**Scaffold distribution shift.** Molecules with previously unseen scaffolds may differ substantially from the training compounds. Scaffold-aware evaluation helps investigate this challenge but does not eliminate distribution shift.

**Probability reliability.** Softmax probabilities are not necessarily calibrated. Temperature scaling is applied after training, but with a small validation split the fitted temperature is itself noisy, and a high predicted probability should not automatically be interpreted as a reliable estimate of activity. Calibration is assessed on a test split of roughly 150 molecules; bootstrap intervals ignore scaffold dependence and the uncertainty in `T`.

**Conformal coverage under distribution shift.** Split conformal prediction sets are implemented, but their marginal coverage relies on calibration and test samples being exchangeable. Scaffold-aware splitting deliberately separates scaffold groups, so that assumption may not hold across these partitions; empirical test coverage is descriptive and not a guarantee under arbitrary scaffold shift. Group summaries do not change the fitted threshold.

**Limited interpretability and uncertainty estimation.** The pipeline does not yet provide validated molecular explanations, ensembles, or scaffold-aware uncertainty estimates. Conformal sets provide set-valued predictions but do not establish prospective activity or correct distribution shift.

**Benchmark scope.** Results on one molecular target do not establish generalization to other targets or prospective drug-discovery settings.

## Next steps

The next development phase will focus on reliable molecular prediction rather than adding model architectures alone.

~~### 1. Correct the experimental protocol~~

~~Introduce a fixed, scaffold-aware four-way split:~~

~~| Partition | Target proportion | Purpose |~~
~~|---|---:|---|
~~| Training | 70% | Fit model parameters. |~~
~~| Validation | 10% | Checkpoint/model selection, early stopping, and fitting the temperature (currently). |~~
~~| Calibration | 10% | Reserved for conformal thresholds (not used by temperature scaling). |~~
~~| Test | 10% | Final, untouched evaluation. |~~

~~The target counts are approximate; scaffold grouping and class balance may affect the actual partition sizes.~~

~~The implementation should:~~

~~- Keep test-set metrics out of checkpoint selection (done).~~
~~- Use validation data for early stopping and model selection.~~
~~- Keep calibration data separate from model fitting and selection.~~
~~- Evaluate the frozen pipeline on the test set only after the modeling and calibration decisions are finalized.~~
~~- Save partition assignments and audit scaffold overlap, class distributions, and sample counts.~~

~~### 2. Probability calibration~~

~~**Status: implemented** (see [Probability calibration](#probability-calibration)). Temperature scaling is fitted on validation logits and evaluated on the test split with Brier score, NLL, ECE, reliability diagrams (with per-bin counts) and paired bootstrap intervals. Still planned: an independent temperature-fitting set (for example cross-fitting) and scaffold-aware uncertainty intervals; the current bootstrap intervals are optimistic (see above).~~

### 3. Conformal prediction

**Status: implemented.** The pipeline fits split-conformal thresholds on the held-out calibration predictions and evaluates test prediction sets (see [Split conformal prediction](#split-conformal-prediction)). Further work should investigate calibration strategies and coverage under scaffold distribution shift.

Current reports include:

- Empirical coverage
- Average prediction-set size
- Singleton prediction rate
- Coverage by activity class
- Coverage across scaffold novelty or molecular similarity groups

The nominal coverage target does not guarantee the same coverage under arbitrary distribution shift. This is particularly important when evaluating previously unseen molecular scaffolds.

### 4. Uncertainty quantification

Investigate deep ensembles and molecular novelty indicators.

Potential additions include:

- Ensemble predictive entropy
- Model disagreement
- Maximum Tanimoto similarity to training molecules
- Scaffold novelty indicators
- Error analysis as a function of uncertainty

The objective is to assess whether uncertainty measures help identify unreliable predictions, rather than assuming that every uncertainty score is informative.

### 5. Explainable AI for molecular graphs

Investigate molecular explanation methods such as GNNExplainer and atom-level attribution.

The planned outputs include molecular visualizations highlighting influential atoms, bonds, and subgraphs.

Explanations should be evaluated for fidelity and stability across model seeds. Highlighted substructures should not automatically be interpreted as causal determinants of biological activity.

### 6. Comparative evaluation

After correcting the evaluation protocol, compare the classical baselines and graph models under consistent conditions.

The goal is to establish whether graph-based learning improves prediction quality over molecular fingerprints and whether the more complex hybrid architecture provides a meaningful advantage.

### 7. Generalization to other molecular targets

A follow-up experiment on an additional molecular activity dataset could help determine whether the resulting evaluation, calibration, and uncertainty framework transfers beyond BACE-1.

## Project direction

The long-term goal is to develop a reproducible molecular machine learning pipeline that combines:

- Molecular graph learning
- Classical machine learning baselines
- Scaffold-aware evaluation
- Probability calibration
- Conformal prediction
- Predictive uncertainty estimation
- Molecular explainability
- Experiment tracking and workflow automation

The central research question is whether a molecular model can provide not only useful activity predictions, but also meaningful information about when those predictions are reliable and which structural features contribute to them.

## Disclaimer

This project is a research and educational implementation for molecular activity classification. Predictions are computational estimates and do not establish experimental inhibitory activity, clinical efficacy, or suitability as a therapeutic candidate.
