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
        "--test_size {config[test_size]} --n_trials {config[n_trials]}"


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
        "--radius {config[radius]} --n_bits {config[n_bits]}"


rule baselines:
    input:
        FINGERPRINT_FILES
    output:
        SELECTED_BASELINE
    shell:
        "{PYTHON} " + SCRIPT("baseline_model.py") + " "
        "--data " + FINGERPRINT_DIR + " "
        "--confusion-matrix-path " + OUTPUT_DIR + " "
        "--predictions-dir " + PREDICTIONS_DIR


rule train_gnn:
    input:
        split=SPLIT_FILE,
        csv=RAW_CSV
    output:
        os.path.join(MODEL_DIR, "{model}.pth"),
        os.path.join(PREDICTIONS_DIR, "{model}", "validation_predictions.csv"),
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
        "--model-output {output[0]}"


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
        output_dir=CALIBRATION_DIR
    shell:
        "{PYTHON} " + SCRIPT("calibration.py") + " "
        "--fit-predictions {input.validation} --eval-predictions {input.test} "
        "--model-name {wildcards.model} --checkpoint {input.checkpoint} "
        "--output-dir {params.output_dir} --temperature-output {output[0]} "
        "--mlflow-experiment bace-calibration"
