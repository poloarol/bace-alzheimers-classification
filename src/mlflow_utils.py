from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> None:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def log_parameters(parameters: dict[str, Any]) -> None:
    mlflow.log_params({
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in parameters.items()
    })


def log_files(paths: list[str | Path]) -> None:
    for path in paths:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"MLflow artifact does not exist: {artifact_path}")
        mlflow.log_artifact(str(artifact_path))
