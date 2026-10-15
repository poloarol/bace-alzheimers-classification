# BACE-1 Molecular Activity Classification

A molecular machine learning project comparing classical machine learning and graph neural networks for predicting whether a small molecule inhibits beta-secretase 1 (BACE-1).

The project combines molecular graph learning, cheminformatics, scaffold-aware evaluation, experiment tracking, and reproducible workflow automation. The next development phase will focus on explainability, probability calibration, and uncertainty-aware predictions.

## Overview

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

**Important evaluation caveat:** In the current training implementation, test accuracy is evaluated during training and used for checkpoint selection. Consequently, the reported metrics should be considered preliminary rather than an unbiased final estimate of generalization performance.

A revised experimental protocol is planned before drawing stronger conclusions about model performance.

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
└── src/
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── baseline_model.py
    ├── build_fingerprints.py
    ├── stratified_scaffold_split.py
    ├── inference.py
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
| `inference.py` | Model inference and confusion-matrix generation. |
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

```bash
python app.py split
```

The existing workflow writes the split artifact to:

```text
data/MoleculeNet/bace/clean/train_test_split.npz
```

The current split is used by downstream fingerprint and graph-model workflows. The planned four-way train/validation/calibration/test protocol has not yet been implemented.

### 2. Generate Morgan fingerprints

```bash
python app.py fingerprints
```

Fingerprint features and labels are written under:

```text
data/MoleculeNet/bace/fingerprints/
```

### 3. Train classical baselines

```bash
python app.py baseline
```

This trains the Logistic Regression, Random Forest, and Gradient Boosting baselines and generates evaluation outputs.

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

```bash
python app.py all --model GAT
```

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

The workflow orchestrates scaffold splitting, fingerprint generation, baseline training, and graph-model training according to `config.yaml`.

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

Tracking settings can be configured using `MLFLOW_TRACKING_URI` or the supported command-line options.

## Limitations

Several limitations should be addressed before treating the current results as a reliable estimate of prospective molecular activity prediction.

**Evaluation leakage through checkpoint selection.** The current training workflow uses test accuracy during training to select a checkpoint. This allows the test set to influence model selection and can make the reported metrics optimistic.

**Limited dataset size.** With approximately 1,530 molecules, model performance estimates and calibration metrics may be sensitive to the composition of the data partitions.

**Scaffold distribution shift.** Molecules with previously unseen scaffolds may differ substantially from the training compounds. Scaffold-aware evaluation helps investigate this challenge but does not eliminate distribution shift.

**Probability reliability.** Softmax probabilities are not necessarily calibrated. A high predicted probability should not automatically be interpreted as a reliable estimate of activity.

**Limited interpretability and uncertainty estimation.** The current pipeline does not yet provide validated molecular explanations, calibrated predictive probabilities, or conformal prediction sets.

**Benchmark scope.** Results on one molecular target do not establish generalization to other targets or prospective drug-discovery settings.

## Next steps

The next development phase will focus on reliable molecular prediction rather than adding model architectures alone.

### 1. Correct the experimental protocol

Introduce a fixed, scaffold-aware four-way split:

| Partition | Target proportion | Purpose |
|---|---:|---|
| Training | 70% | Fit model parameters. |
| Validation | 10% | Model selection and checkpoint selection. |
| Calibration | 10% | Fit probability calibration and conformal thresholds. |
| Test | 10% | Final, untouched evaluation. |

The target counts are approximate; scaffold grouping and class balance may affect the actual partition sizes.

The implementation should:

- Remove test-set metrics from checkpoint selection.
- Use validation data for early stopping and model selection.
- Keep calibration data separate from model fitting and selection.
- Evaluate the frozen pipeline on the test set only after the modeling and calibration decisions are finalized.
- Save partition assignments and audit scaffold overlap, class distributions, and sample counts.

### 2. Probability calibration

Investigate temperature scaling and evaluate whether predicted probabilities agree with observed activity frequencies.

Planned metrics include:

- Brier score
- Negative log-likelihood
- Expected calibration error (ECE)
- Reliability diagrams

Calibration quality should be evaluated on held-out test data after the calibration method has been fitted.

### 3. Conformal prediction

Implement split conformal classification to return a prediction set rather than only a single predicted class.

Evaluate:

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
