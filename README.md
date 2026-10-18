# BACE-1 Molecular Activity Classification

A reproducible molecular machine learning project comparing classical machine learning and graph neural networks (GNNs) for predicting whether small molecules inhibit beta-secretase 1 (BACE-1).

The project combines molecular representations, scaffold-aware evaluation, probability calibration, split conformal prediction, experiment tracking, and workflow automation. It investigates not only how accurately models classify molecular activity, but also how reliable their probabilities are and how prediction sets can communicate uncertainty.

**Current result:** On the current held-out scaffold-aware test split, a Morgan fingerprint + Gradient Boosting baseline outperforms the three evaluated graph neural networks. The graph models nevertheless provide a useful comparison for studying molecular representation learning, calibration, and uncertainty.

> **Scope:** This is a molecular activity classification task, not an Alzheimer's disease diagnostic model. Computational predictions do not establish experimental inhibitory activity, clinical efficacy, or therapeutic safety.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Presentation](#presentation)
- [Dataset](#dataset)
- [Methods](#methods)
- [Current Results](#current-results)
- [Data Audit](#data-audit)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Probability Calibration](#probability-calibration)
- [Split Conformal Prediction](#split-conformal-prediction)
- [Audit and Model Comparison](#audit-and-model-comparison)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Testing](#testing)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Project Direction](#project-direction)
- [Disclaimer](#disclaimer)

## Project Overview

BACE-1, also known as beta-secretase 1, is an enzyme involved in amyloid-beta production and has been investigated as a therapeutic target in Alzheimer's disease research.

This project uses molecular structures to predict a binary activity label:

- **Class 0:** Non-inhibitor.
- **Class 1:** BACE-1 inhibitor.

The workflow compares conventional molecular fingerprints with learned graph representations. It then evaluates classification performance, probability calibration, and conformal prediction sets on held-out molecules.

## Presentation

Explore the project's methodology, model comparisons, probability calibration, and conformal prediction results in the accompanying presentation.

[**View the project presentation (PowerPoint)**](https://docs.google.com/presentation/d/1drB5MubWnGnHKsfPZF--sIvAH_Bp1OaUSenGJC2klSM/edit?usp=sharing)

The presentation covers the BACE-1 classification task, molecular representations, GNN architectures, benchmark results, uncertainty quantification, and key findings.

NB: I am still updating it, as I explore the results and understand the results but also explore other areas of interests.

### Key features

- Molecule processing using RDKit, DeepChem, and PyTorch Geometric.
- Morgan fingerprint generation for classical machine learning.
- Scaffold-aware train, validation, calibration, and test partitions.
- Classical baseline classifiers: Logistic Regression, Random Forest, and Gradient Boosting.
- Graph Convolutional Network (GCN).
- Graph Attention Network (GAT).
- Hybrid GAT + Graph Transformer architecture.
- MLflow experiment tracking and artifact logging.
- Temperature scaling for probability calibration.
- Split conformal prediction with a nominal 90% coverage target.
- Classification, calibration, and prediction-set reports.
- Snakemake workflow orchestration.
- Dataset and partition-integrity auditing.

## Objectives

1. Compare classical machine learning and graph neural networks for molecular activity classification.
2. Evaluate Morgan fingerprints against atom-and-bond graph representations.
3. Investigate generalization using scaffold-aware data partitioning.
4. Assess probability quality using Brier score, negative log-likelihood, and expected calibration error.
5. Explore set-valued predictions through split conformal prediction.
6. Maintain reproducible experiments through MLflow and Snakemake.
7. Establish a foundation for molecular explainability and uncertainty-aware prediction.

## Dataset

The project uses the BACE dataset distributed through [MoleculeNet](https://moleculenet.org/), containing **1,513 molecular observations** in the audited dataset.

| Property | Value |
|---|---:|
| Total molecules | 1,513 |
| Non-inhibitors (class 0) | 822 |
| Inhibitors (class 1) | 691 |
| Class 0 proportion | 54.3% |
| Class 1 proportion | 45.7% |
| Exact duplicate SMILES in audit | 0 |

Molecules are represented by SMILES strings and processed into molecular fingerprints or graph structures.

### Label interpretation

The target is the source-provided binary MoleculeNet activity label. It should not be treated as equivalent to applying a strict pIC50 threshold of 7 to every molecule.

The audit identified 17 class-1 molecules with pIC50 below 7. No class-0 molecules had pIC50 at or above 7. The current experiments retain the source-provided labels rather than silently redefining the task.

### Data partitioning

The workflow uses four partitions with approximate target proportions of 70% training, 10% validation, 10% calibration, and 10% testing.

| Partition | Molecules | Primary purpose |
|---|---:|---|
| Training | 1,060 | Fit model parameters |
| Validation | 151 | Model selection, early stopping, temperature fitting |
| Calibration | 151 | Fit conformal prediction threshold |
| Test | 151 | Final evaluation |

The current split contains 1,513 molecules in total. Scaffold grouping can affect exact partition sizes and class balance.

The intended evaluation is scaffold-aware: the split-generation workflow is designed to keep molecular scaffold groups separated. The audit reported below verifies molecule-level partition integrity; pairwise scaffold-overlap and standardized-structure-overlap reports should also be retained when validating the exact split used for a benchmark.

## Methods

### 1. Molecular representations

**Morgan fingerprints**

Morgan fingerprints encode local molecular environments as fixed-length binary vectors. The current baseline uses radius 2 and 2,048 bits.

**Molecular graphs**

Graph representations encode atoms as nodes and chemical bonds as edges. Atom and bond features are used by graph neural networks to learn molecular representations through neighborhood aggregation and attention.

### 2. Classical machine learning

The project implements three classifiers using Morgan fingerprints:

- Logistic Regression
- Random Forest
- Gradient Boosting

The baseline models provide reference points for evaluating whether learned graph representations improve activity prediction.

The current headline baseline is `GradientBoostingClassifier`.

### 3. Graph neural networks

| Architecture | Description |
|---|---|
| GCN | Graph Convolutional Network that aggregates information from neighboring atoms. |
| GAT | Graph Attention Network that learns attention-weighted neighborhood representations. |
| GAT + Graph Transformer | Hybrid graph architecture combining graph attention with Transformer-style graph convolution. |

The graph models produce binary activity predictions from molecular graph features.

### 4. Evaluation metrics

**Classification**

- PR-AUC (area under the precision–recall curve)
- ROC-AUC (area under the receiver operating characteristic curve)
- Accuracy
- Balanced accuracy
- Inhibitor-class F1-score
- Confusion matrices

PR-AUC is particularly useful when assessing positive-class retrieval. ROC-AUC evaluates ranking across classification thresholds, while balanced accuracy and inhibitor F1 provide complementary views of classification behavior.

**Probability calibration**

- Brier score
- Negative log-likelihood (NLL)
- Expected calibration error (ECE)
- Reliability diagrams
- Paired bootstrap intervals for calibration changes

**Conformal prediction**

- Overall empirical coverage
- Class-specific coverage
- Average prediction-set size
- Singleton prediction rate
- Ambiguous prediction rate
- Empty-set rate

### 5. Model selection and data roles

Models are selected using validation PR-AUC. The calibration split is reserved for fitting the conformal threshold, and the test split is used for final evaluation.

The current temperature-scaling protocol fits the temperature on validation predictions. Because validation is also used for model selection and checkpoint selection, temperature fitting is not fully independent of those decisions.

## Current Results

The tables below report the latest comparison available for the current experimental configuration. All four models are evaluated on the same held-out test partition of 151 molecules: 82 non-inhibitors and 69 inhibitors.

These results represent **one split and one seed**, not a repeated-split benchmark.

### 1. Classification performance

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Accuracy | Balanced accuracy | Inhibitor F1 |
|---|---:|---:|---:|---:|---:|---:|
| Morgan + Gradient Boosting | 0.883 | **0.791** | **0.794** | **0.715** | **0.706** | **0.656** |
| GCN | 0.831 | 0.699 | 0.729 | 0.689 | 0.684 | 0.647 |
| GAT | 0.767 | 0.640 | 0.674 | 0.623 | 0.610 | 0.529 |
| GAT + Graph Transformer | 0.826 | 0.591 | 0.691 | 0.636 | 0.627 | 0.567 |

**Key finding:** The Morgan fingerprint baseline achieves the strongest test PR-AUC, ROC-AUC, accuracy, balanced accuracy, and inhibitor F1 in this comparison. GCN is the strongest graph model by test PR-AUC, balanced accuracy, and inhibitor F1.

The results show that graph learning did not outperform the classical baseline under the current experimental configuration. They do not establish that GNNs are inherently inferior for molecular activity prediction.

The validation-to-test performance gap is substantial for several models. Scaffold-aware evaluation is intended to probe generalization to unseen molecular scaffolds, but the current experiment does not isolate the cause of that gap.

The graph models have not undergone comprehensive hyperparameter optimization or repeated-seed evaluation. Training convergence and data-loader shuffling should be verified before interpreting the results as a stable architectural comparison.

### 2. Probability calibration

Temperature scaling applies a positive scalar temperature \(T\) to model logits:
```
\[
p_k = \frac{\exp(z_k/T)}
{\sum_j \exp(z_j/T)}
\]
```
The temperature is fitted by minimizing validation negative log-likelihood. A temperature above 1 softens the probability distribution; a temperature below 1 sharpens it.

The following metrics are measured on the test set before and after calibration. Lower values are better.

| Model | Temperature | Brier before | Brier after | NLL before | NLL after | ECE before | ECE after |
|---|---:|---:|---:|---:|---:|---:|---:|
| Morgan + Gradient Boosting | 0.841 | 0.187 | 0.190 | 0.557 | 0.570 | 0.081 | 0.120 |
| GCN | 6.105 | 0.252 | 0.215 | 1.766 | 0.667 | 0.237 | 0.105 |
| GAT | 2.314 | 0.249 | 0.232 | 0.873 | 0.680 | 0.153 | **0.074** |
| GAT + Graph Transformer | 1.654 | 0.250 | 0.237 | 0.874 | 0.720 | 0.168 | 0.125 |

Temperature scaling improves the reported Brier score, NLL, and ECE point estimates for the three graph models, although the magnitude differs by architecture. GAT achieves the lowest post-calibration ECE, while GCN shows the largest reduction in NLL.

The Morgan baseline's point estimates become slightly worse after temperature scaling. Paired bootstrap intervals for the baseline changes include zero, so this is not strong evidence of a meaningful deterioration.

ECE depends on binning and can be unstable with small test samples. Bootstrap intervals quantify test-sample variability conditional on the fitted temperature; they do not account for uncertainty in temperature estimation or dependence between chemically related molecules.

### 3. Split conformal prediction

Split conformal prediction returns a set of plausible labels rather than only one predicted class. The current nominal marginal coverage target is 90%.

| Model | Overall coverage | Class 0 coverage | Class 1 coverage | Average set size | Singleton rate | Ambiguous rate |
|---|---:|---:|---:|---:|---:|---:|
| Morgan + Gradient Boosting | **90.1%** | 97.6% | 81.2% | **1.48** | **51.7%** | 48.3% |
| GCN | 88.7% | 91.5% | 85.5% | 1.54 | 46.4% | 53.6% |
| GAT | 88.7% | 97.6% | 78.3% | 1.60 | 40.4% | 59.6% |
| GAT + Graph Transformer | 89.4% | 91.5% | **87.0%** | 1.67 | 33.1% | 66.9% |

Class 0 represents non-inhibitors; class 1 represents inhibitors.

A prediction set can take three forms:

- `{0}` — non-inhibitor only.
- `{1}` — inhibitor only.
- `{0, 1}` — both classes remain plausible.

The Morgan baseline produces the smallest average sets and the highest singleton rate, with 90.1% overall empirical coverage. However, its inhibitor-class coverage is 81.2%.

The GAT + Graph Transformer model has the highest observed inhibitor coverage at 87.0% and more balanced coverage across the two classes, but it also produces the largest average sets and the lowest singleton rate.

These results illustrate a trade-off between coverage and decisiveness. Smaller sets are not automatically preferable if they exclude the true label more frequently. Larger sets are not, by themselves, proof of intrinsically greater model uncertainty.

**Coverage caveat:** A nominal 90% marginal target does not guarantee 90% coverage within each class or under arbitrary distribution shift. Standard split-conformal coverage relies on exchangeability between calibration and test observations. Generalization to unseen chemical scaffolds can challenge that assumption, so the coverage reported here is empirical rather than a guarantee under scaffold shift.

### 4. Overall interpretation

The current experiment provides three complementary findings:

1. **Classification:** Morgan fingerprints + Gradient Boosting outperform the tested graph architectures on the current test split.
2. **Calibration:** Temperature scaling improves several probability-quality metrics for the graph models, but does not improve every metric for every model.
3. **Uncertainty:** Conformal prediction reveals class-wise coverage differences and varying levels of prediction-set ambiguity that are not captured by ROC-AUC or accuracy alone.

The next priority is to verify the training implementation and establish whether these findings persist across repeated scaffold splits and seeds.

## Data Audit

The audit command checks source labels, SMILES validity, graph counts, and partition assignments.

Run:

```bash
python app.py audit
```

Alternatively:

```bash
python src/audit_splits.py
```

The report is written to:

```text
data/MoleculeNet/bace/output/split_audit.json
```

### Dataset integrity

| Audit check | Result |
|---|---:|
| Total molecules | 1,513 |
| Class 0 molecules | 822 |
| Class 1 molecules | 691 |
| Exact duplicate SMILES | 0 |
| Invalid SMILES | 0 |
| Processed graph counts match molecule counts | Yes, in all partitions |
| Partitions disjoint | Yes |
| All molecules assigned | Yes |
| Processed labels match source labels | Yes |

### Partition counts

| Partition | Molecules | Class 0 | Class 1 | Processed graphs |
|---|---:|---:|---:|---:|
| Train | 1,060 | 576 | 484 | 1,060 |
| Validation | 151 | 82 | 69 | 151 |
| Calibration | 151 | 82 | 69 | 151 |
| Test | 151 | 82 | 69 | 151 |
| **Total** | **1,513** | **822** | **691** | **1,513** |

### Label discrepancy

The audit found 17 class-1 molecules with pIC50 below 7, representing approximately 1.1% of the dataset. No class-0 molecules had pIC50 at or above 7.

The source-provided activity labels are retained for the current task. The audit establishes the discrepancy but does not establish its cause.

### What the audit does not prove

Zero duplicate SMILES and disjoint partition indices do not automatically establish that no chemically equivalent structures occur across partitions, since the same structure can have different SMILES representations.

Likewise, molecule-level partition integrity does not by itself prove scaffold separation. The split-generation workflow is designed to keep scaffold groups separated, but a persisted pairwise scaffold-overlap report and a canonicalized-structure overlap check should be reviewed before describing the benchmark as fully validated for unseen-scaffold generalization.

## Repository Structure

The principal files and directories are organized as follows:

```text
bace-alzheimers-classification/
├── app.py
├── config.yaml
├── requirements.txt
├── Snakefile
├── README.md
├── notebooks/
├── model/
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
    ├── audit_splits.py
    ├── compare_models.py
    └── mlflow_utils.py
```

| Component | Purpose |
|---|---|
| `app.py` | Command-line pipeline runner |
| `config.yaml` | Workflow and model configuration |
| `Snakefile` | Snakemake workflow orchestration |
| `dataset.py` | Molecular graph dataset processing |
| `model.py` | GCN, GAT, and hybrid graph Transformer architectures |
| `train.py` | GNN training and evaluation |
| `baseline_model.py` | Classical fingerprint-based classifiers |
| `build_fingerprints.py` | Morgan fingerprint generation |
| `stratified_scaffold_split.py` | Scaffold-aware partition generation |
| `inference.py` | Model inference and confusion-matrix export |
| `calibration.py` | Temperature fitting and probability-quality metrics |
| `conformal.py` | Split conformal prediction and coverage evaluation |
| `audit_splits.py` | Dataset and partition audit |
| `compare_models.py` | Cross-model result comparison |
| `mlflow_utils.py` | Shared MLflow configuration |

Generated datasets, processed graphs, checkpoints, prediction files, and local MLflow database files are runtime artifacts. They generally should not be committed to version control.

## Installation

### Prerequisites

The project uses Python and scientific machine learning libraries, including:

- PyTorch
- PyTorch Geometric
- DeepChem
- RDKit
- scikit-learn
- MLflow
- Snakemake
- pandas and NumPy

Compatible versions of PyTorch, PyTorch Geometric, and the remaining dependencies are required. Refer to the installed `requirements.txt` and the official PyTorch Geometric installation instructions for platform-specific dependency requirements.

### Clone the repository

```bash
git clone https://github.com/poloarol/bace-alzheimers-classification.git
cd bace-alzheimers-classification
```

### Install dependencies

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails for PyTorch or PyTorch Geometric, install compatible versions for your operating system and hardware according to their official installation instructions.

## Usage

The root-level `app.py` runner provides commands for individual workflow stages. By default, the dataset root is `data/MoleculeNet/bace`; use `--dataset-root` to specify another location when supported by the command.

### 1. Generate the scaffold-aware split

```bash
python app.py split
```

The default split archive is:

```text
data/MoleculeNet/bace/clean/train_test_split.npz
```

It contains the arrays:

- `train_idx`
- `validation_idx`
- `calibration_idx`
- `test_idx`

The default target proportions are 70% training and 10% for each of validation, calibration, and test.

Example with explicit proportions:

```bash
python app.py split \
  --train-size 0.7 \
  --validation-size 0.1 \
  --calibration-size 0.1 \
  --test-size 0.1
```

The split-generation command supports additional options, including `--n-trials`, for the scaffold-assignment search budget. See the command help for available arguments.

### 2. Generate Morgan fingerprints

```bash
python app.py fingerprints
```

Fingerprint and label CSV files are written under:

```text
data/MoleculeNet/bace/fingerprints/
```

The directory contains split-specific files for training, validation, calibration, and test data.

The fingerprint configuration can be changed with `--radius` and `--n-bits` where supported. The current baseline uses radius 2 and 2,048 bits.

### 3. Train the classical baselines

```bash
python app.py baseline
```

This trains the supported fingerprint-based classifiers, including Logistic Regression, Random Forest, and Gradient Boosting.

The selected baseline is chosen using validation PR-AUC. Its prediction artifacts are then used by the downstream calibration and conformal stages.

### 4. Train graph neural networks

Train a GCN:

```bash
python app.py gnn --model GCN --epochs 100
```

Train a GAT:

```bash
python app.py gnn --model GAT --epochs 100
```

Train the hybrid GAT + Graph Transformer:

```bash
python app.py gnn \
  --model GATGraphTransformer \
  --epochs 100 \
  --batch-size 32
```

Train all supported graph architectures:

```bash
python app.py gnn --all-models --epochs 100
```

Available model options include hidden-channel and learning-rate settings. Consult `python app.py gnn --help` for the options supported by the current checkout.

Validation PR-AUC controls checkpoint selection and early stopping. Calibration and test data should not be used to select model parameters.

### 5. Run the complete pipeline

Run the default pipeline:

```bash
python app.py all
```

Run all supported GNN architectures and finish with the audit and cross-model comparison:

```bash
python app.py all --all-models
```

An example configuration is:

```bash
python app.py all \
  --train-size 0.7 \
  --validation-size 0.1 \
  --calibration-size 0.1 \
  --test-size 0.1 \
  --n-trials 100 \
  --radius 2 \
  --n-bits 2048 \
  --epochs 100 \
  --hidden-channels 64 \
  --learning-rate 0.01 \
  --batch-size 32
```

Verify command-line options with `python app.py all --help` if the current version exposes different options.

### 6. Run inference

Inference can load a model logged in MLflow using its model URI.

Example for a classical model:

```bash
python app.py inference \
  --model-uri "runs:/<run-id>/model" \
  --model-flavor sklearn \
  --model-type RandomForestClassifier
```

Example for a graph model:

```bash
python app.py inference \
  --model-uri "runs:/<run-id>/GAT-Model" \
  --model-flavor pytorch \
  --model-type GAT
```

Replace `<run-id>` with the identifier of the relevant MLflow run.

Confusion matrices are written under the model-specific output directory, typically within:

```text
data/MoleculeNet/bace/output/
```

### 7. Run Snakemake

Preview the workflow:

```bash
snakemake --cores 1 -n
```

Execute it:

```bash
snakemake --cores 1
```

The workflow is configured through `config.yaml` and orchestrates preprocessing, model training, calibration, conformal evaluation, auditing, and model comparison.

If Snakemake uses a different Python interpreter from the one containing the project dependencies, configure the interpreter as appropriate for the current environment.

## Probability Calibration

Temperature scaling adjusts model logits using a single positive scalar \(T\). The fit minimizes validation NLL; the resulting probabilities are evaluated on the held-out test data.

The current implementation writes calibration artifacts under:

```text
model/<ModelName>.calibration.json
data/MoleculeNet/bace/output/calibration/<ModelName>/
```

Typical artifacts include:

- Calibration JSON
- Reliability diagram
- Per-bin reliability table
- Machine-readable calibration report
- Brier score, NLL, and ECE before and after scaling
- Paired bootstrap intervals for metric changes

Recalibrate an existing GNN without retraining:

```bash
python app.py calibrate --model-type GCN
python app.py calibrate --model-type GAT
python app.py calibrate --model-type GATGraphTransformer
```

Recalibrate the selected baseline:

```bash
python app.py calibrate --model-type best-baseline
```

Use `python app.py calibrate --help` for options such as bin count, bootstrap resamples, seed, and output paths.

### Interpreting calibration metrics

- **Brier score:** Measures squared error between predicted probabilities and binary outcomes.
- **NLL:** Penalizes incorrect probabilities, especially confident incorrect predictions.
- **ECE:** Summarizes the difference between confidence and observed frequency across probability bins.

No single metric fully describes calibration. ECE is sensitive to binning, and a small numerical difference should not automatically be interpreted as a meaningful improvement.

Temperature scaling changes probabilities but ordinarily does not change the argmax class prediction or ranking metrics such as ROC-AUC and PR-AUC.

## Split Conformal Prediction

The conformal stage uses a separate calibration partition to estimate a nonconformity threshold and evaluates prediction sets on the test partition.

The nominal target is 90% marginal coverage.

Run conformal evaluation for a previously trained model:

```bash
python app.py conformal --model-type GCN
```

Other supported model types include:

```bash
python app.py conformal --model-type GAT
python app.py conformal --model-type GATGraphTransformer
python app.py conformal --model-type best-baseline
```

The pipeline uses `crepes` for conformal prediction. The current implementation uses nonconformity scores based on \(1 - P(\text{true class})\), with temperature-scaled probabilities where the calibration artifact is applied.

Typical outputs are written under:

```text
data/MoleculeNet/bace/output/conformal/<ModelName>/
```

They include:

- `conformal_predictions.csv`
- `conformal_report.json`

The prediction CSV contains labels, probabilities, predicted classes, prediction sets, and set sizes. The report summarizes coverage and set-size metrics, including class-wise summaries.

### Interpreting prediction sets

- `{0}` means only the non-inhibitor class is included.
- `{1}` means only the inhibitor class is included.
- `{0, 1}` means both classes are included.

A singleton set can be incorrect. An ambiguous set can still contain the true class even when the highest-probability class is incorrect.

The nominal 90% target is not a guarantee of 90% coverage for each class or under arbitrary distribution shift. Calibration and test examples must satisfy the relevant exchangeability assumptions for the standard split-conformal marginal coverage guarantee. Scaffold-based generalization may challenge those assumptions.

## Audit and Model Comparison

Run the data audit:

```bash
python app.py audit
```

Compare existing model artifacts:

```bash
python app.py compare
```

Retrain the baseline and graph models before comparing:

```bash
python app.py compare --retrain
```

The comparison writes:

```text
data/MoleculeNet/bace/output/comparison/comparison.csv
data/MoleculeNet/bace/output/comparison/comparison.json
```

It checks that model prediction labels match the labels associated with the expected partitions and logs the comparison to the `bace-comparison` MLflow experiment.

After retraining, rerun calibration, conformal evaluation, and comparison so the reports correspond to the latest prediction artifacts.

## MLflow Experiment Tracking

The project uses MLflow to track experiments, metrics, and artifacts.

Launch the local MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the local URL printed in the terminal.

The default experiments include:

| Experiment | Contents |
|---|---|
| `bace-preprocessing` | Scaffold splitting and fingerprint generation |
| `bace-baseline` | Classical fingerprint-based models |
| `bace-gnn` | GNN training, metrics, and model artifacts |
| `bace-calibration` | Temperature-scaling metrics and artifacts |
| `bace-conformal` | Prediction-set coverage, sizes, and artifacts |
| `bace-comparison` | Cross-model metrics, split metadata, and comparison tables |

The tracking URI can be configured using `MLFLOW_TRACKING_URI` or supported command-line options.

Generated model artifacts and the local tracking database are runtime files; do not commit them unless they are deliberately part of a versioned release.

## Testing

Run the unit and workflow test suite:

```bash
python -m unittest discover -s tests -v
```

The repository's tests cover areas including split validation, graph and fingerprint processing, model construction, inference, calibration, conformal prediction, CLI behavior, and workflow integration.

To inspect the planned workflow without executing it, where supported:

```bash
python app.py all --dry-run
```

A passing test suite verifies the tested software behavior; it does not, by itself, establish that model comparisons are statistically robust or that the data satisfy all assumptions required for conformal coverage.

## Limitations

Several limitations should be considered before treating the current results as estimates of prospective molecular activity prediction.

**Small dataset.** The benchmark contains 1,513 molecules. Test metrics, probability calibration, and conformal coverage may be sensitive to the composition of the partitions.

**Single seed and split.** The comparison has not yet been established across multiple independent scaffold splits and random seeds. Small metric differences should not be interpreted as definitive evidence of model superiority.

**Graph-model optimization.** The current graph architectures have not undergone comprehensive hyperparameter tuning. Training convergence and data-loader shuffling should be checked before the next benchmark.

**Validation reuse.** Validation data are used for model or checkpoint selection and for fitting the calibration temperature. Temperature estimation is therefore not fully independent of model selection.

**Chemical distribution shift.** Scaffold-aware evaluation probes generalization to different chemical structures but does not eliminate distribution shift. Calibration and test scaffolds may differ in ways that affect performance and uncertainty.

**Conformal coverage.** The usual split-conformal marginal coverage guarantee relies on exchangeability between calibration and test observations. Empirical coverage on a scaffold-aware split is descriptive and may not match nominal coverage under scaffold shift.

**Structure identity.** Zero exact duplicate SMILES does not rule out chemically equivalent structures with different representations. Standardized-structure overlap should be audited across partitions.

**Limited interpretability.** Validated atom-level and substructure explanations are not yet part of the reported benchmark.

**Benchmark scope.** Performance on one target does not establish transfer to other molecular targets or prospective drug discovery. Predicted BACE-1 inhibition does not establish clinical efficacy or safety in Alzheimer's disease.

## Next Steps

The next phase prioritizes experimental reliability, useful uncertainty estimates, and molecular explanations.
<strike>
```
### 1. Strengthen data and split validation

- Persist a pairwise scaffold-overlap report for all four partitions.
- Audit canonicalized and standardized molecular structures across partitions.
- Preserve stable molecule identifiers and split metadata.
- Investigate the 17 source-label/pIC50 discrepancies without silently changing the target labels.

### 2. Improve the benchmark

- Verify that training batches are reshuffled correctly between epochs.
- Check graph-model convergence and validation learning curves.
- Repeat experiments across multiple random seeds and scaffold splits.
- Perform targeted hyperparameter optimization.
- Report uncertainty intervals for comparative model performance.
- Keep model selection separate from final test evaluation.

### 3. Expand uncertainty quantification

Investigate whether uncertainty indicators help identify unreliable predictions. Candidate methods include:

- Deep ensembles and model disagreement.
- Predictive entropy.
- Maximum Tanimoto similarity to training molecules.
- Scaffold novelty indicators.
- Error analysis stratified by molecular similarity and novelty.
- Coverage and set-size analysis across chemical subgroups.

The objective is to assess whether uncertainty measures are empirically associated with prediction errors rather than assuming that every uncertainty score is informative.
```
</strike>

### 4. Add molecular explainability

Investigate graph explanation methods such as GNNExplainer and atom-level attribution to highlight influential atoms, bonds, and substructures.

Explanations should be assessed for fidelity and stability across model seeds. Highlighted substructures should not automatically be interpreted as causal determinants of biological activity.

### 5. Evaluate generalization

Assess whether the Morgan baseline's observed advantage persists with better-optimized graph models, multiple scaffold splits, and additional molecular activity datasets.

This will help determine whether the findings reflect the current configuration or a more general pattern across molecular prediction tasks.

## Project Direction

The long-term goal is to build a reproducible molecular machine learning workflow that combines:

- Classical molecular machine learning baselines.
- Graph neural networks and molecular representation learning.
- Scaffold-aware evaluation.
- Probability calibration.
- Conformal prediction.
- Predictive uncertainty estimation.
- Molecular explainability.
- Experiment tracking and workflow automation.

The central research question is whether a molecular model can provide not only useful activity predictions, but also meaningful information about when those predictions are reliable and which structural features contribute to them.

## Disclaimer

This project is for research and educational purposes. Model predictions are computational estimates and do not establish experimental inhibitory activity, clinical efficacy, or suitability as a therapeutic candidate.
