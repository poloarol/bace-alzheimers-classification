import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CALIBRATION_FILE_SUFFIX = ".calibration.json"


def probabilities_to_logits(probabilities: np.ndarray) -> torch.Tensor:
    """Two-class log-probabilities, equivalent to logits up to a constant shift."""
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return torch.as_tensor(np.log(probabilities), dtype=torch.float64)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Fit a positive temperature T by minimising NLL of softmax(logits / T)."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[0] == 0:
        raise ValueError("logits must be a non-empty (N, C) tensor matching labels.")

    # Optimise log(T) so that T = exp(log_t) is always positive.
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> np.ndarray:
    """Positive-class probabilities from temperature-scaled logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logits = torch.as_tensor(logits, dtype=torch.float64)
    return torch.softmax(logits / temperature, dim=1)[:, 1].numpy()


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels, dtype=float)) ** 2))


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10):
    """Per-bin mean confidence, observed positive rate and count (positive class)."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    confidence, accuracy, counts = [], [], []
    for b in range(n_bins):
        mask = indices == b
        counts.append(int(mask.sum()))
        confidence.append(probabilities[mask].mean() if mask.any() else np.nan)
        accuracy.append(labels[mask].mean() if mask.any() else np.nan)
    return np.asarray(confidence), np.asarray(accuracy), np.asarray(counts)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    confidence, accuracy, counts = reliability_bins(labels, probabilities, n_bins)
    populated = counts > 0
    weights = counts[populated] / counts.sum()
    return float(np.sum(weights * np.abs(accuracy[populated] - confidence[populated])))


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "Brier": brier_score(labels, probabilities),
        "NLL": negative_log_likelihood(labels, probabilities),
        "ECE": expected_calibration_error(labels, probabilities, n_bins),
    }


def plot_reliability_diagram(
    labels: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    filepath: str,
    n_bins: int = 10,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, probabilities in (("Uncalibrated", uncalibrated), ("Temperature scaled", calibrated)):
        confidence, accuracy, _ = reliability_bins(labels, probabilities, n_bins)
        populated = ~np.isnan(confidence)
        axis.plot(confidence[populated], accuracy[populated], marker="o", label=name)
    axis.set_xlabel("Mean predicted probability (inhibitor)")
    axis.set_ylabel("Observed inhibitor fraction")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(filepath, dpi=150)
    plt.close(figure)


def calibrate_and_compare(
    validation_logits: torch.Tensor,
    validation_labels,
    test_logits: torch.Tensor,
    test_labels,
    output_directory: str,
    n_bins: int = 10,
) -> dict:
    """Fit T on validation, then compare uncalibrated vs scaled probabilities on the same test samples."""
    temperature = fit_temperature(validation_logits, torch.as_tensor(np.array(validation_labels)))
    test_labels = np.asarray(test_labels, dtype=int)
    uncalibrated = apply_temperature(test_logits, 1.0)
    calibrated = apply_temperature(test_logits, temperature)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        test_labels, uncalibrated, calibrated, str(output / "reliability_diagram.png"), n_bins
    )
    return {
        "temperature": temperature,
        "uncalibrated": calibration_metrics(test_labels, uncalibrated, n_bins),
        "calibrated": calibration_metrics(test_labels, calibrated, n_bins),
        "uncalibrated_probabilities": uncalibrated,
        "calibrated_probabilities": calibrated,
    }


def save_predictions(labels, logits: torch.Tensor, filepath) -> None:
    """Write labels and raw two-class logits in the CSV format read by the calibration CLI."""
    import pandas as pd

    logits = torch.as_tensor(logits, dtype=torch.float64).numpy()
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "label": np.asarray(labels, dtype=int),
        "logit_0": logits[:, 0],
        "logit_1": logits[:, 1],
    }).to_csv(filepath, index=False)


def prediction_paths(predictions_directory, model_name: str) -> tuple[Path, Path]:
    """Validation and test prediction CSV paths for a model."""
    directory = Path(predictions_directory) / model_name
    return directory / "validation_predictions.csv", directory / "test_predictions.csv"


def calibration_path_for(model_path: str) -> Path:
    return Path(model_path).with_suffix(CALIBRATION_FILE_SUFFIX)


def save_temperature(temperature: float, filepath, **metadata) -> None:
    Path(filepath).write_text(
        json.dumps({"temperature": temperature, **metadata}, indent=2), encoding="utf-8"
    )


def load_temperature(filepath) -> float:
    temperature = float(json.loads(Path(filepath).read_text(encoding="utf-8"))["temperature"])
    if temperature <= 0:
        raise ValueError("Stored temperature must be positive.")
    return temperature


def _read_predictions(csv_path: str):
    """Read a CSV with `label` plus either `logit_0`,`logit_1` or `probability` (positive class)."""
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if "label" not in frame:
        raise ValueError(f"{csv_path} needs a 'label' column.")
    if {"logit_0", "logit_1"} <= set(frame.columns):
        logits = torch.as_tensor(frame[["logit_0", "logit_1"]].to_numpy(), dtype=torch.float64)
    elif "probability" in frame:
        p = frame["probability"].to_numpy(dtype=np.float64)
        logits = probabilities_to_logits(np.stack([1 - p, p], axis=1))
    else:
        raise ValueError(f"{csv_path} needs 'logit_0','logit_1' or 'probability' columns.")
    return logits, frame["label"].to_numpy(dtype=int)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a temperature on validation predictions and compare calibration on test."
    )
    parser.add_argument("--validation", required=True, help="Validation predictions CSV.")
    parser.add_argument("--test", required=True, help="Test predictions CSV.")
    parser.add_argument("--output-dir", default="model/calibration", help="Directory for the diagram and metrics.")
    parser.add_argument(
        "--temperature-output",
        help="Path for the temperature JSON. Defaults to <output-dir>/temperature.calibration.json.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of ECE/reliability bins.")
    args = parser.parse_args()

    validation_logits, validation_labels = _read_predictions(args.validation)
    test_logits, test_labels = _read_predictions(args.test)
    result = calibrate_and_compare(
        validation_logits, validation_labels, test_logits, test_labels, args.output_dir, args.bins
    )

    temperature_path = args.temperature_output or str(
        Path(args.output_dir) / f"temperature{CALIBRATION_FILE_SUFFIX}"
    )
    save_temperature(result["temperature"], temperature_path, fitted_on="validation")

    print(f"Temperature: {result['temperature']:.4f}")
    for kind in ("uncalibrated", "calibrated"):
        print(f"{kind}: {result[kind]}")
    print(f"Saved temperature to: {temperature_path}")
    print(f"Saved reliability diagram to: {Path(args.output_dir) / 'reliability_diagram.png'}")


if __name__ == "__main__":
    main()


# Usage notes:
# - Fit T on validation data only; evaluate the before/after comparison on test data.
#   Never fit T on the test set.
# - Python API:
#       from calibration import fit_temperature, apply_temperature, save_temperature, load_temperature
#       T = fit_temperature(validation_logits, validation_labels)   # logits: (N, 2) raw, pre-softmax
#       save_temperature(T, "model/GCN.calibration.json")
#       probabilities = apply_temperature(test_logits, load_temperature("model/GCN.calibration.json"))
# - Scaling divides logits by T, so predicted classes (argmax) and ROC-AUC/PR-AUC are unchanged.
#   T > 1 softens overconfident predictions; T < 1 sharpens underconfident ones.
# - Sklearn models have no logits: use probabilities_to_logits(predict_proba(X)).
# - CLI input CSVs need a `label` column plus either `logit_0`,`logit_1` or `probability`
#   (positive-class probability):
#       python src/calibration.py --validation val_preds.csv --test test_preds.csv --output-dir model/calibration
