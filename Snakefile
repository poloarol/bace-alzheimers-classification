configfile: "config.yaml"

DATASET_ROOT = config["dataset_root"]
RAW_CSV = DATASET_ROOT + r"\raw\bace_processed.csv"
SPLIT_FILE = DATASET_ROOT + r"\clean\train_test_split.npz"
FINGERPRINT_DIR = DATASET_ROOT + r"\fingerprints"
OUTPUT_DIR = DATASET_ROOT + r"\output"
MODEL_DIR = "model"
PYTHON = config["python"]

BASELINE_MODELS = [
    "RandomForestClassifier",
    "LogisticRegression",
    "GradientBoostingClassifier",
]
FINGERPRINT_FILES = [
    FINGERPRINT_DIR + r"\train_fingerprints.csv",
    FINGERPRINT_DIR + r"\test_fingerprints.csv",
    FINGERPRINT_DIR + r"\train_labels.csv",
    FINGERPRINT_DIR + r"\test_labels.csv",
]


rule all:
    input:
        SPLIT_FILE,
        FINGERPRINT_FILES,
        expand(OUTPUT_DIR + r"\{model}\confusion_matrix.png", model=BASELINE_MODELS),
        expand(MODEL_DIR + r"\{model}.pth", model=config["gnn_models"]),


rule split:
    input:
        RAW_CSV
    output:
        SPLIT_FILE
    shell:
        "{PYTHON} src\\stratified_scaffold_split.py "
        "--input {input} --output {output} "
        "--test_size {config[test_size]} --n_trials {config[n_trials]}"


rule fingerprints:
    input:
        csv=RAW_CSV,
        split=SPLIT_FILE
    output:
        FINGERPRINT_FILES
    shell:
        "{PYTHON} src\\build_fingerprints.py "
        "--input_csv {input.csv} "
        "--train_test_split_npz {input.split} "
        "--output_dir " + FINGERPRINT_DIR + " "
        "--radius {config[radius]} --n_bits {config[n_bits]}"


rule baselines:
    input:
        train=FINGERPRINT_DIR + r"\train_fingerprints.csv",
        test=FINGERPRINT_DIR + r"\test_fingerprints.csv"
    output:
        expand(OUTPUT_DIR + r"\{model}\confusion_matrix.png", model=BASELINE_MODELS)
    shell:
        "{PYTHON} src\\baseline_model.py "
        "--data " + FINGERPRINT_DIR + " "
        "--confusion-matrix-path " + OUTPUT_DIR


rule train_gnn:
    input:
        split=SPLIT_FILE,
        csv=RAW_CSV
    output:
        MODEL_DIR + r"\{model}.pth"
    params:
        epochs=config["epochs"],
        hidden_channels=config["hidden_channels"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"]
    shell:
        "{PYTHON} src\\train.py "
        "--dataset_path " + DATASET_ROOT + " "
        "--model {wildcards.model} "
        "--epochs {params.epochs} "
        "--hidden-channels {params.hidden_channels} "
        "--learning-rate {params.learning_rate} "
        "--batch-size {params.batch_size} "
        "--model-output {output}"
