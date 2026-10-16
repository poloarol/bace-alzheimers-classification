import os

configfile: "config.yaml"

DATASET_ROOT = config["dataset_root"]
RAW_CSV = os.path.join(DATASET_ROOT, "raw", "bace_processed.csv")
SPLIT_FILE = os.path.join(DATASET_ROOT, "clean", "train_test_split.npz")
FINGERPRINT_DIR = os.path.join(DATASET_ROOT, "fingerprints")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "output")
MODEL_DIR = "model"
PYTHON = config["python"]

PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
SELECTED_BASELINE = os.path.join(PREDICTIONS_DIR, "selected_baseline.json")
# Canonical calibration artifact, stored next to the model: model/<model>.calibration.json
CALIBRATION_JSON = os.path.join(MODEL_DIR, "{model}.calibration.json")
CALIBRATION_DIR = os.path.join(OUTPUT_DIR, "calibration", "{model}")

CONFORMAL_DIR = os.path.join(OUTPUT_DIR, "conformal", "{model}")
SPLIT_AUDIT = os.path.join(OUTPUT_DIR, "split_audit.json")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparison")
COMPARISON_FILES = [
    os.path.join(COMPARISON_DIR, "comparison.csv"),
    os.path.join(COMPARISON_DIR, "comparison.json"),
]
# The winning baseline is only known after training, so its calibration and
# conformal steps go through app.py and are tracked with marker files.
BASELINE_CALIBRATED = os.path.join(OUTPUT_DIR, "calibration", "best-baseline.done")
BASELINE_CONFORMAL = os.path.join(OUTPUT_DIR, "conformal", "best-baseline.done")
APP_ARGS = "--dataset-root " + DATASET_ROOT

SPLITS = ["train", "validation", "calibration", "test"]
FINGERPRINT_FILES = [
    os.path.join(FINGERPRINT_DIR, f"{split}_fingerprints.csv") for split in SPLITS
] + [
    os.path.join(FINGERPRINT_DIR, f"{split}_labels.csv") for split in SPLITS
]
SCRIPT = lambda name: os.path.join("src", name)


rule all:
    input:
        SPLIT_FILE,
        FINGERPRINT_FILES,
        SELECTED_BASELINE,
        expand(os.path.join(MODEL_DIR, "{model}.pth"), model=config["gnn_models"]),
        expand(CALIBRATION_JSON, model=config["gnn_models"]),
        expand(os.path.join(CALIBRATION_DIR, "calibration_report.json"), model=config["gnn_models"]),
        expand(os.path.join(CONFORMAL_DIR, "conformal_report.json"), model=config["gnn_models"]),
        BASELINE_CALIBRATED,
        BASELINE_CONFORMAL,
        SPLIT_AUDIT,
        COMPARISON_FILES,


rule split:
    input:
        RAW_CSV
    output:
        SPLIT_FILE
    shell:
        "{PYTHON} " + SCRIPT("stratified_scaffold_split.py") + " "
        "--input {input} --output {output} "
        "--train_size {config[train_size]} "
        "--validation_size {config[validation_size]} "
        "--calibration_size {config[calibration_size]} "
        "--test_size {config[test_size]} --n_trials {config[n_trials]} "
        "--mlflow-experiment bace-preprocessing"


rule fingerprints:
    input:
        csv=RAW_CSV,
        split=SPLIT_FILE
    output:
        FINGERPRINT_FILES
    shell:
        "{PYTHON} " + SCRIPT("build_fingerprints.py") + " "
        "--input_csv {input.csv} "
        "--train_test_split_npz {input.split} "
        "--output_dir " + FINGERPRINT_DIR + " "
        "--radius {config[radius]} --n_bits {config[n_bits]} "
        "--mlflow-experiment bace-preprocessing"


rule baselines:
    input:
        FINGERPRINT_FILES
    output:
        SELECTED_BASELINE
    shell:
        "{PYTHON} " + SCRIPT("baseline_model.py") + " "
        "--data " + FINGERPRINT_DIR + " "
        "--confusion-matrix-path " + OUTPUT_DIR + " "
        "--predictions-dir " + PREDICTIONS_DIR + " "
        "--mlflow-experiment bace-baseline"


rule train_gnn:
    input:
        split=SPLIT_FILE,
        csv=RAW_CSV
    output:
        os.path.join(MODEL_DIR, "{model}.pth"),
        os.path.join(PREDICTIONS_DIR, "{model}", "validation_predictions.csv"),
        os.path.join(PREDICTIONS_DIR, "{model}", "calibration_predictions.csv"),
        os.path.join(PREDICTIONS_DIR, "{model}", "test_predictions.csv")
    params:
        epochs=config["epochs"],
        hidden_channels=config["hidden_channels"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"]
    shell:
        "{PYTHON} " + SCRIPT("train.py") + " "
        "--dataset_path " + DATASET_ROOT + " "
        "--predictions-dir " + PREDICTIONS_DIR + " "
        "--model {wildcards.model} "
        "--epochs {params.epochs} "
        "--hidden-channels {params.hidden_channels} "
        "--learning-rate {params.learning_rate} "
        "--batch-size {params.batch_size} "
        "--model-output {output[0]} "
        "--mlflow-experiment bace-gnn"


# Single authoritative calibration step: fits T on validation logits, evaluates on test.
rule calibrate_gnn:
    input:
        checkpoint=os.path.join(MODEL_DIR, "{model}.pth"),
        validation=os.path.join(PREDICTIONS_DIR, "{model}", "validation_predictions.csv"),
        test=os.path.join(PREDICTIONS_DIR, "{model}", "test_predictions.csv")
    output:
        CALIBRATION_JSON,
        os.path.join(CALIBRATION_DIR, "reliability_diagram.png"),
        os.path.join(CALIBRATION_DIR, "reliability_table.csv"),
        os.path.join(CALIBRATION_DIR, "calibration_report.json")
    params:
        output_dir=CALIBRATION_DIR,
        bins=config["calibration_bins"],
        bootstrap=config["calibration_bootstrap"]
    shell:
        "{PYTHON} " + SCRIPT("calibration.py") + " "
        "--fit-predictions {input.validation} --eval-predictions {input.test} "
        "--model-name {wildcards.model} --checkpoint {input.checkpoint} "
        "--output-dir {params.output_dir} --temperature-output {output[0]} "
        "--bins {params.bins} --bootstrap {params.bootstrap} "
        "--mlflow-experiment bace-calibration"


# Conformal sets: fit on the calibration split, report on the untouched test split.
rule conformal_gnn:
    input:
        calibration=os.path.join(PREDICTIONS_DIR, "{model}", "calibration_predictions.csv"),
        test=os.path.join(PREDICTIONS_DIR, "{model}", "test_predictions.csv"),
        temperature=CALIBRATION_JSON
    output:
        os.path.join(CONFORMAL_DIR, "conformal_report.json"),
        os.path.join(CONFORMAL_DIR, "conformal_predictions.csv")
    params:
        output_dir=CONFORMAL_DIR,
        confidence=config["confidence"]
    shell:
        "{PYTHON} " + SCRIPT("conformal.py") + " "
        "--calibration-predictions {input.calibration} --test-predictions {input.test} "
        "--temperature-file {input.temperature} --model-name {wildcards.model} "
        "--output-dir {params.output_dir} --confidence {params.confidence} "
        "--mlflow-experiment bace-conformal"


rule calibrate_baseline:
    input:
        SELECTED_BASELINE
    output:
        touch(BASELINE_CALIBRATED)
    params:
        bins=config["calibration_bins"],
        bootstrap=config["calibration_bootstrap"]
    shell:
        "{PYTHON} app.py calibrate " + APP_ARGS + " --model-type best-baseline "
        "--bins {params.bins} --bootstrap {params.bootstrap}"


rule conformal_baseline:
    input:
        selection=SELECTED_BASELINE,
        calibrated=BASELINE_CALIBRATED
    output:
        touch(BASELINE_CONFORMAL)
    params:
        confidence=config["confidence"]
    shell:
        "{PYTHON} app.py conformal " + APP_ARGS + " --model-type best-baseline "
        "--confidence {params.confidence}"


rule audit:
    input:
        csv=RAW_CSV,
        split=SPLIT_FILE,
        # Trained GNNs build the processed graph datasets that the audit counts.
        models=expand(os.path.join(MODEL_DIR, "{model}.pth"), model=config["gnn_models"])
    output:
        SPLIT_AUDIT
    shell:
        "{PYTHON} " + SCRIPT("audit_splits.py") + " "
        "--dataset-path " + DATASET_ROOT + " --output {output}"


# One comparison run on the shared split; verifies that every model saw the same molecules.
rule compare:
    input:
        SPLIT_AUDIT,
        BASELINE_CALIBRATED,
        BASELINE_CONFORMAL,
        expand(CALIBRATION_JSON, model=config["gnn_models"]),
        expand(os.path.join(CONFORMAL_DIR, "conformal_report.json"), model=config["gnn_models"]),
        expand(os.path.join(CALIBRATION_DIR, "calibration_report.json"), model=config["gnn_models"])
    output:
        COMPARISON_FILES
    params:
        epochs=config["epochs"],
        hidden_channels=config["hidden_channels"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        confidence=config["confidence"]
    shell:
        "{PYTHON} " + SCRIPT("compare_models.py") + " "
        "--dataset-path " + DATASET_ROOT + " "
        "--epochs {params.epochs} --hidden-channels {params.hidden_channels} "
        "--learning-rate {params.learning_rate} --batch-size {params.batch_size} "
        "--confidence {params.confidence} --mlflow-experiment bace-comparison"
