import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"


@dataclass(frozen=True)
class PipelinePaths:
    dataset_root: Path
    raw_csv: Path
    split_file: Path
    fingerprint_directory: Path
    output_directory: Path

    @classmethod
    def from_dataset_root(cls, dataset_root: str) -> "PipelinePaths":
        root = Path(dataset_root).resolve()
        return cls(
            dataset_root=root,
            raw_csv=root / "raw" / "bace_processed.csv",
            split_file=root / "clean" / "train_test_split.npz",
            fingerprint_directory=root / "fingerprints",
            output_directory=root / "output",
        )


class PipelineRunner:
    def __init__(
        self,
        paths: PipelinePaths,
        tracking_uri: str | None,
        dry_run: bool = False,
    ) -> None:
        self.paths = paths
        self.tracking_uri = tracking_uri
        self.dry_run = dry_run

    def _run(self, script_name: str, arguments: Sequence[str]) -> None:
        command = [
            sys.executable,
            str(SOURCE_DIRECTORY / script_name),
            *arguments,
        ]
        print(">", subprocess.list2cmdline(command))
        if not self.dry_run:
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    def _tracking_arguments(self, experiment: str) -> list[str]:
        arguments = ["--mlflow-experiment", experiment]
        if self.tracking_uri:
            arguments.extend(["--mlflow-tracking-uri", self.tracking_uri])
        return arguments

    def run_split(self, test_size: float, n_trials: int) -> None:
        self._require_file(self.paths.raw_csv)
        self._run(
            "stratified_scaffold_split.py",
            [
                "--input",
                str(self.paths.raw_csv),
                "--output",
                str(self.paths.split_file),
                "--test_size",
                str(test_size),
                "--n_trials",
                str(n_trials),
                *self._tracking_arguments("bace-preprocessing"),
            ],
        )

    def run_fingerprints(self, radius: int, n_bits: int) -> None:
        self._require_file(self.paths.raw_csv)
        self._require_file(self.paths.split_file)
        self._run(
            "build_fingerprints.py",
            [
                "--input_csv",
                str(self.paths.raw_csv),
                "--train_test_split_npz",
                str(self.paths.split_file),
                "--output_dir",
                str(self.paths.fingerprint_directory),
                "--radius",
                str(radius),
                "--n_bits",
                str(n_bits),
                *self._tracking_arguments("bace-preprocessing"),
            ],
        )

    def run_baselines(self) -> None:
        self._require_file(self.paths.fingerprint_directory / "train_fingerprints.csv")
        self._require_file(self.paths.fingerprint_directory / "test_fingerprints.csv")
        self._run(
            "baseline_model.py",
            [
                "--data",
                str(self.paths.fingerprint_directory),
                "--confusion-matrix-path",
                str(self.paths.output_directory),
                *self._tracking_arguments("bace-baseline"),
            ],
        )

    def run_gnn_training(
        self,
        model: str,
        epochs: int,
        hidden_channels: int,
        learning_rate: float,
        batch_size: int,
    ) -> None:
        self._require_file(self.paths.raw_csv)
        self._require_file(self.paths.split_file)
        self._run(
            "train.py",
            [
                "--dataset_path",
                str(self.paths.dataset_root),
                "--model",
                model,
                "--epochs",
                str(epochs),
                "--hidden-channels",
                str(hidden_channels),
                "--learning-rate",
                str(learning_rate),
                "--batch-size",
                str(batch_size),
                *self._tracking_arguments("bace-gnn"),
            ],
        )

    def run_inference(
        self,
        model_uri: str,
        model_flavor: str,
        model_type: str,
        batch_size: int,
    ) -> None:
        arguments = [
            "--model-uri",
            model_uri,
            "--model-flavor",
            model_flavor,
            "--model-type",
            model_type,
            "--output-dir",
            str(self.paths.output_directory),
            "--batch-size",
            str(batch_size),
        ]
        if model_flavor == "sklearn":
            arguments.extend([
                "--fingerprint-dir",
                str(self.paths.fingerprint_directory),
            ])
        else:
            arguments.extend([
                "--dataset-root",
                str(self.paths.dataset_root),
            ])
        if self.tracking_uri:
            arguments.extend(["--mlflow-tracking-uri", self.tracking_uri])
        self._run("inference.py", arguments)

    def run_complete_pipeline(
        self,
        model: str,
        test_size: float,
        n_trials: int,
        radius: int,
        n_bits: int,
        epochs: int,
        hidden_channels: int,
        learning_rate: float,
        batch_size: int,
        include_gnn: bool,
    ) -> None:
        self.run_split(test_size, n_trials)
        self.run_fingerprints(radius, n_bits)
        self.run_baselines()
        if include_gnn:
            self.run_gnn_training(
                model,
                epochs,
                hidden_channels,
                learning_rate,
                batch_size,
            )

    def _require_file(self, path: Path) -> None:
        if self.dry_run:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Required pipeline input does not exist: {path}")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        default="data/MoleculeNet/bace",
        help="Root BACE directory containing raw, clean, and processed data.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow tracking URI passed to pipeline scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )


def _add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=100)


def _add_fingerprint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=("GCN", "GAT", "GATGraphTransformer"),
        default="GCN",
        help="Graph model architecture.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)


def _add_inference_arguments(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument("--model-uri", required=required)
    parser.add_argument(
        "--model-flavor",
        choices=("sklearn", "pytorch"),
        required=required,
    )
    parser.add_argument("--model-type", required=required)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BACE classification pipeline or an individual step."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("split", help="Create the scaffold split.")
    _add_common_arguments(split_parser)
    _add_split_arguments(split_parser)

    fingerprints_parser = commands.add_parser(
        "fingerprints",
        help="Build train and test Morgan fingerprints.",
    )
    _add_common_arguments(fingerprints_parser)
    _add_fingerprint_arguments(fingerprints_parser)

    baseline_parser = commands.add_parser(
        "baseline",
        help="Train and evaluate scikit-learn baselines.",
    )
    _add_common_arguments(baseline_parser)

    gnn_parser = commands.add_parser("gnn", help="Train and evaluate the GNN.")
    _add_common_arguments(gnn_parser)
    _add_training_arguments(gnn_parser)

    inference_parser = commands.add_parser(
        "inference",
        help="Run a tracked model on the test set.",
    )
    _add_common_arguments(inference_parser)
    _add_inference_arguments(inference_parser, required=True)
    inference_parser.add_argument("--batch-size", type=int, default=32)

    all_parser = commands.add_parser("all", help="Run split, fingerprints, baselines, and GNN training.")
    _add_common_arguments(all_parser)
    _add_split_arguments(all_parser)
    _add_fingerprint_arguments(all_parser)
    _add_training_arguments(all_parser)
    all_parser.add_argument(
        "--skip-gnn",
        action="store_true",
        help="Run the preprocessing and baseline stages only.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = PipelinePaths.from_dataset_root(args.dataset_root)
    runner = PipelineRunner(
        paths,
        tracking_uri=args.mlflow_tracking_uri,
        dry_run=args.dry_run,
    )

    if args.command == "split":
        runner.run_split(args.test_size, args.n_trials)
    elif args.command == "fingerprints":
        runner.run_fingerprints(args.radius, args.n_bits)
    elif args.command == "baseline":
        runner.run_baselines()
    elif args.command == "gnn":
        runner.run_gnn_training(
            args.model,
            args.epochs,
            args.hidden_channels,
            args.learning_rate,
            args.batch_size,
        )
    elif args.command == "inference":
        runner.run_inference(
            args.model_uri,
            args.model_flavor,
            args.model_type,
            args.batch_size,
        )
    else:
        runner.run_complete_pipeline(
            args.model,
            args.test_size,
            args.n_trials,
            args.radius,
            args.n_bits,
            args.epochs,
            args.hidden_channels,
            args.learning_rate,
            args.batch_size,
            include_gnn=not args.skip_gnn,
        )


if __name__ == "__main__":
    main()
    
# Inspect the complete pipeline without running it
# python app.py all --dry-run

# Run the full training pipeline
# python app.py all

# Run each stage independently
# python app.py split
# python app.py fingerprints
# python app.py baseline
# python app.py gnn --model GCN --epochs 100
# python app.py gnn --model GAT --epochs 100
# python app.py gnn --model GATGraphTransformer --epochs 100

# Run test inference from a tracked MLflow model
# python app.py inference `
#   --model-uri "runs:/<run-id>/model" `
#   --model-flavor sklearn `
#   --model-type RandomForestClassifier