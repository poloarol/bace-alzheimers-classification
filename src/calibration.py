import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

CALIBRATION_FILE_SUFFIX = ".calibration.json"
# Fixed documented baseline; do not tune the bin count to make a plot look better.
DEFAULT_BINS = 10


def probabilities_to_logits(probabilities: np.ndarray) -> torch.Tensor:
    """Two-class log-probabilities, equivalent to logits up to a constant shift.

    Expects a non-empty finite (N, 2) array of probabilities in [0, 1] whose rows sum to 1.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities must have shape (N, 2).")
    if probabilities.shape[0] == 0:
        raise ValueError("probabilities must not be empty.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1.")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return torch.as_tensor(np.log(probabilities), dtype=torch.float64)


TEMPERATURE_BOUNDS = (1e-2, 1e2)


def _validate_fit_inputs(logits: torch.Tensor, labels: torch.Tensor, max_iter: int):
    logits = torch.as_tensor(logits)
    if not logits.is_floating_point():
        raise ValueError("logits must be a floating-point tensor.")
    logits = logits.to(torch.float64)

    labels = torch.as_tensor(labels)
    if labels.is_floating_point():
        if not torch.isfinite(labels).all() or (labels != labels.round()).any():
            raise ValueError("labels must be integer class indices.")
    labels = labels.to(torch.long).view(-1)

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (N, C) with C >= 2.")
    if logits.shape[0] == 0 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be non-empty and match the number of labels.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values.")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError(f"labels must be in [0, {logits.shape[1] - 1}].")
    if labels.unique().numel() < 2:
        raise ValueError("labels contain a single class; the temperature is not identifiable.")
    if torch.allclose(logits, logits[:, :1].expand_as(logits)):
        raise ValueError("logits are identical across classes; the temperature is not identifiable.")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    return logits, labels


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Fit a positive temperature T by minimising NLL of softmax(logits / T) with LBFGS.

    Raises ValueError for invalid inputs and RuntimeError if the optimiser does not
    return a finite, in-bounds temperature that improves on T = 1.
    """
    logits, labels = _validate_fit_inputs(logits, labels, max_iter)

    # Optimise log(T) so that T = exp(log_t) is always positive.
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(log_t.exp().item())
    low, high = TEMPERATURE_BOUNDS
    if not np.isfinite(temperature) or not low <= temperature <= high:
        raise RuntimeError(
            f"Temperature fit did not converge to a value in [{low}, {high}]: {temperature}."
        )
    with torch.no_grad():
        fitted_nll = F.cross_entropy(logits / temperature, labels).item()
        initial_nll = F.cross_entropy(logits, labels).item()
    if fitted_nll > initial_nll + 1e-9:
        raise RuntimeError(
            f"Temperature fit increased validation NLL ({initial_nll:.6f} -> {fitted_nll:.6f})."
        )
    return temperature


def apply_temperature(logits: torch.Tensor, temperature: float) -> np.ndarray:
    """Positive-class probabilities from temperature-scaled two-class logits.

    Accepts tensors that require grad; the result is detached from the graph.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    logits = torch.as_tensor(logits).detach().to(torch.float64).cpu()
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("logits must have shape (N, 2).")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values.")
    return torch.softmax(logits / temperature, dim=1)[:, 1].numpy()


def _validate_binary_predictions(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-D int labels and float positive-class probabilities after validation."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be 1-D.")
    if labels.shape[0] == 0:
        raise ValueError("labels and probabilities must not be empty.")
    if labels.shape != probabilities.shape:
        raise ValueError(
            f"labels and probabilities must have the same length ({labels.shape[0]} != {probabilities.shape[0]})."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinite values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must lie in [0, 1].")
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all():
            raise ValueError("labels contain NaN or infinite values.")
    elif labels.dtype.kind not in "iub":
        raise ValueError("labels must be numeric 0/1 values.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary (0 or 1).")
    return labels.astype(int), probabilities


def negative_log_likelihood(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Binary NLL (log loss) of the positive-class probability.

    Probabilities are clipped to [1e-12, 1 - 1e-12] to avoid infinite loss.
    """
    labels, probabilities = _validate_binary_predictions(labels, probabilities)
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Binary Brier score: mean((p_positive - y)^2).

    This is the positive-class convention, in [0, 1]. It is half of the multiclass
    convention (sum of squared errors over both classes), so it is not directly
    comparable with implementations that sum over classes.
    """
    labels, probabilities = _validate_binary_predictions(labels, probabilities)
    return float(np.mean((probabilities - labels) ** 2))


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10):
    """Per-bin mean confidence, observed positive rate and count (positive class)."""
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    labels, probabilities = _validate_binary_predictions(labels, probabilities)
    labels = labels.astype(float)
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


def bootstrap_metric_intervals(
    labels,
    uncalibrated,
    calibrated,
    n_bins: int = DEFAULT_BINS,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired percentile-bootstrap intervals over test molecules.

    The temperature is held fixed (it was fitted on validation), and the same resampled
    molecules score both probability sets, so `delta` (calibrated - uncalibrated; negative
    means improvement for all three metrics) is a paired difference. Caveats: this captures
    test-sample noise only, not temperature-fit variance, and ordinary resampling ignores
    scaffold-level dependence, so the intervals are optimistic. Treat an interval for
    `delta` that contains 0 as no evidence of improvement.
    """
    labels, uncalibrated = _validate_binary_predictions(labels, uncalibrated)
    _, calibrated = _validate_binary_predictions(labels, calibrated)
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    generator = np.random.default_rng(seed)
    names = ("Brier", "NLL", "ECE")
    samples = {kind: {m: np.empty(n_resamples) for m in names} for kind in ("uncalibrated", "calibrated")}
    for i in range(n_resamples):
        index = generator.integers(0, labels.shape[0], labels.shape[0])
        for kind, probabilities in (("uncalibrated", uncalibrated), ("calibrated", calibrated)):
            metrics = calibration_metrics(labels[index], probabilities[index], n_bins)
            for name in names:
                samples[kind][name][i] = metrics[name]

    lower, upper = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100

    def interval(values: np.ndarray) -> dict:
        low, high = np.percentile(values, [lower, upper])
        return {"lower": float(low), "upper": float(high)}

    return {
        "method": "paired percentile bootstrap over test samples (temperature held fixed)",
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "caveat": "Ignores scaffold-level dependence and temperature-fit variance; intervals are optimistic.",
        "uncalibrated": {m: interval(samples["uncalibrated"][m]) for m in names},
        "calibrated": {m: interval(samples["calibrated"][m]) for m in names},
        "delta": {m: interval(samples["calibrated"][m] - samples["uncalibrated"][m]) for m in names},
    }


def reliability_table(labels, uncalibrated, calibrated, n_bins: int = DEFAULT_BINS) -> list[dict]:
    """Per-bin counts, mean probability and observed fraction for both probability sets."""
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = {
        name: reliability_bins(labels, probabilities, n_bins)
        for name, probabilities in (("uncalibrated", uncalibrated), ("calibrated", calibrated))
    }
    for b in range(n_bins):
        row = {"bin_lower": edges[b], "bin_upper": edges[b + 1]}
        for name, (confidence, accuracy, counts) in bins.items():
            row[f"{name}_count"] = int(counts[b])
            row[f"{name}_mean_probability"] = confidence[b]
            row[f"{name}_observed_fraction"] = accuracy[b]
        rows.append(row)
    return rows


def save_reliability_table(rows: list[dict], filepath) -> None:
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_reliability_diagram(
    labels: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    filepath: str,
    n_bins: int = DEFAULT_BINS,
) -> None:
    """Reliability curves with a companion histogram of samples per bin.

    Sparse bins move a lot with one or two molecules; read the counts before the curve.
    """
    figure, (axis, histogram) = plt.subplots(
        2, 1, figsize=(6, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    width = 0.4 / n_bins
    centers = (np.arange(n_bins) + 0.5) / n_bins
    for offset, (name, probabilities) in zip(
        (-width / 2, width / 2),
        (("Uncalibrated", uncalibrated), ("Temperature scaled", calibrated)),
    ):
        confidence, accuracy, counts = reliability_bins(labels, probabilities, n_bins)
        populated = ~np.isnan(confidence)
        line = axis.plot(confidence[populated], accuracy[populated], marker="o", label=name)[0]
        histogram.bar(centers + offset, counts, width=width, color=line.get_color(), label=name)
    axis.set_ylabel("Observed inhibitor fraction")
    axis.set_ylim(0, 1)
    axis.legend()
    histogram.set_xlabel("Mean predicted probability (inhibitor)")
    histogram.set_ylabel("Samples per bin")
    histogram.set_xlim(0, 1)
    figure.tight_layout()
    figure.savefig(filepath, dpi=150)
    plt.close(figure)


def calibrate_and_compare(
    validation_logits: torch.Tensor,
    validation_labels,
    test_logits: torch.Tensor,
    test_labels,
    output_directory: str,
    n_bins: int = DEFAULT_BINS,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 0,
) -> dict:
    """Fit T on validation, then compare uncalibrated vs scaled probabilities on the same test samples.

    Writes reliability_diagram.png and reliability_table.csv (per-bin counts). Brier, NLL and
    ECE should be read together; small differences, especially in ECE on a small test set,
    should not be presented as a meaningful improvement.
    """
    temperature = fit_temperature(validation_logits, torch.as_tensor(np.array(validation_labels)))
    test_labels = np.asarray(test_labels, dtype=int)
    uncalibrated = apply_temperature(test_logits, 1.0)
    calibrated = apply_temperature(test_logits, temperature)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        test_labels, uncalibrated, calibrated, str(output / "reliability_diagram.png"), n_bins
    )
    save_reliability_table(
        reliability_table(test_labels, uncalibrated, calibrated, n_bins),
        output / "reliability_table.csv",
    )
    uncertainty = (
        bootstrap_metric_intervals(
            test_labels, uncalibrated, calibrated, n_bins, n_bootstrap, seed=bootstrap_seed
        )
        if n_bootstrap > 0
        else None
    )
    return {
        "temperature": temperature,
        "uncertainty": uncertainty,
        "n_test_samples": int(test_labels.shape[0]),
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


def calibration_prediction_path(predictions_directory, model_name: str) -> Path:
    """Calibration-split prediction CSV path used by split conformal prediction."""
    return Path(predictions_directory) / model_name / "calibration_predictions.csv"


def calibration_path_for(model_path: str) -> Path:
    return Path(model_path).with_suffix(CALIBRATION_FILE_SUFFIX)


SCHEMA_VERSION = 1
POSITIVE_CLASS_NOTE = "class 1 = BACE-1 inhibitor"


def save_temperature(temperature: float, filepath, **metadata) -> None:
    """Write the canonical calibration artifact (temperature plus metadata) as JSON.

    Creates parent directories. Extra keyword arguments (model name, checkpoint, metrics,
    sample counts, ...) are stored alongside the temperature.
    """
    if not isinstance(temperature, (int, float, np.floating)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive, finite number.")
    reserved = {"schema_version", "temperature", "positive_class"} & set(metadata)
    if reserved:
        raise ValueError(f"metadata may not override reserved keys: {sorted(reserved)}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "temperature": float(temperature),
        "positive_class": POSITIVE_CLASS_NOTE,
        "method": "temperature scaling: softmax(logits / T)",
        **metadata,
    }
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__} to JSON.")


def load_calibration(filepath, expected_model: str | None = None) -> dict:
    """Load and validate a calibration artifact.

    If expected_model is given and the artifact names a different model, raise so a
    temperature fitted for one model is never applied to another.
    """
    try:
        payload = json.loads(Path(filepath).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Calibration file {filepath} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration file {filepath} must contain a JSON object.")
    version = payload.get("schema_version", 1)
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration schema_version: {version!r}.")
    temperature = payload.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Stored temperature must be a positive, finite number.")
    stored_model = payload.get("model")
    if expected_model is not None and stored_model is not None and stored_model != expected_model:
        raise ValueError(
            f"Calibration file is for model {stored_model!r}, not {expected_model!r}."
        )
    return payload


def load_temperature(filepath, expected_model: str | None = None) -> float:
    return float(load_calibration(filepath, expected_model)["temperature"])


def _read_predictions(csv_path: str):
    """Read and validate a prediction CSV.

    Needs a `label` column (0/1) plus either `logit_0`,`logit_1` or `probability`. The
    `probability` column is the probability of class 1 (BACE-1 inhibitor), not an
    arbitrary confidence score, and must lie in [0, 1].
    """
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"{csv_path} contains no rows.")
    if "label" not in frame:
        raise ValueError(f"{csv_path} needs a 'label' column.")
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{csv_path}: 'label' must contain only 0 or 1.")

    has_logits = {"logit_0", "logit_1"} <= set(frame.columns)
    if has_logits and "probability" in frame:
        raise ValueError(f"{csv_path}: provide logit_0/logit_1 or probability, not both.")
    if has_logits:
        values = frame[["logit_0", "logit_1"]].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64,
            copy=True,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{csv_path}: logit columns must be numeric and finite.")
        logits = torch.as_tensor(values, dtype=torch.float64)
    elif "probability" in frame:
        p = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(dtype=np.float64)
        logits = probabilities_to_logits(np.stack([1 - p, p], axis=1))
    else:
        raise ValueError(f"{csv_path} needs 'logit_0','logit_1' or 'probability' columns.")
    return logits, labels.astype(int)


def class_counts(labels) -> dict:
    labels = np.asarray(labels, dtype=int)
    return {"n": int(labels.shape[0]), "class_0": int((labels == 0).sum()), "class_1": int((labels == 1).sum())}


def _positive_int(value: str) -> int:
    import argparse

    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.")
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer.")
    return number


FIT_SPLIT = "validation"
EVAL_SPLIT = "test"
KNOWN_SPLITS = ("train", "validation", "calibration", "test")


def check_split_filename(path: str, expected_split: str, role: str) -> None:
    """Reject a file whose name states a different split than the role it is given.

    Files named after a known split (for example `test_predictions.csv`) must match the
    role; other names are allowed and are recorded as-is in the metadata.
    """
    name = Path(path).name.lower()
    for split in KNOWN_SPLITS:
        if name.startswith(split) and split != expected_split:
            raise ValueError(
                f"{path} looks like {split!r} predictions but was given as the {role} input, "
                f"which must come from the {expected_split!r} split."
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Fit a temperature on {FIT_SPLIT}-split predictions and compare calibration on "
        f"{EVAL_SPLIT}-split predictions. The {EVAL_SPLIT} split is never used to fit; the "
        "calibration split is reserved for conformal prediction and is not used here."
    )
    parser.add_argument(
        "--fit-predictions", "--validation", dest="fit_predictions", required=True,
        help=f"Predictions CSV from the {FIT_SPLIT} split; the temperature is fitted on it.",
    )
    parser.add_argument(
        "--eval-predictions", "--test", dest="eval_predictions", required=True,
        help=f"Predictions CSV from the {EVAL_SPLIT} split; used only for the before/after comparison.",
    )
    parser.add_argument("--output-dir", default="model/calibration", help="Directory for the diagram and table.")
    parser.add_argument(
        "--temperature-output",
        help="Path for the calibration JSON (temperature, metadata, metrics). "
        "Defaults to <output-dir>/calibration.json.",
    )
    parser.add_argument("--model-name", help="Model the predictions came from; stored in the JSON.")
    parser.add_argument("--checkpoint", help="Checkpoint or run identifier; stored in the JSON.")
    parser.add_argument("--bins", type=_positive_int, default=DEFAULT_BINS, help="Number of ECE/reliability bins.")
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Bootstrap resamples for 95%% metric intervals on the test set (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap random seed.")
    parser.add_argument("--mlflow-experiment", help="If set, log metrics (test_<kind>_<metric>) and artifacts to this MLflow experiment.")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking URI (defaults to the repository setting).")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0.")
    if Path(args.fit_predictions).resolve() == Path(args.eval_predictions).resolve():
        parser.error("--fit-predictions and --eval-predictions must be different files.")
    try:
        check_split_filename(args.fit_predictions, FIT_SPLIT, "--fit-predictions")
        check_split_filename(args.eval_predictions, EVAL_SPLIT, "--eval-predictions")
    except ValueError as error:
        parser.error(str(error))

    fit_logits, fit_labels = _read_predictions(args.fit_predictions)
    eval_logits, eval_labels = _read_predictions(args.eval_predictions)
    fit_counts = class_counts(fit_labels)
    eval_counts = class_counts(eval_labels)
    print(f"Fit split ({FIT_SPLIT}) samples: {fit_counts}")
    print(f"Evaluation split ({EVAL_SPLIT}) samples: {eval_counts}")

    result = calibrate_and_compare(
        fit_logits, fit_labels, eval_logits, eval_labels, args.output_dir, args.bins,
        args.bootstrap, args.seed,
    )

    temperature_path = args.temperature_output or str(Path(args.output_dir) / "calibration.json")
    save_temperature(
        result["temperature"],
        temperature_path,
        model=args.model_name,
        checkpoint=args.checkpoint,
        fit_split=FIT_SPLIT,
        fit_predictions_file=str(args.fit_predictions),
        fit_counts=fit_counts,
        fit_split_note="The fit split also drove checkpoint selection, so the fit is not fully independent.",
        eval_split=EVAL_SPLIT,
        eval_predictions_file=str(args.eval_predictions),
        eval_counts=eval_counts,
        n_bins=args.bins,
        metrics={"uncalibrated": result["uncalibrated"], "calibrated": result["calibrated"]},
    )

    print(f"Temperature: {result['temperature']:.4f}")
    for kind in ("uncalibrated", "calibrated"):
        print(f"{kind}: {result[kind]}")
    if result["uncertainty"]:
        level = result["uncertainty"]["confidence"]
        for metric, bounds in result["uncertainty"]["delta"].items():
            print(f"delta {metric} ({level:.0%} CI, calibrated - uncalibrated): [{bounds['lower']:.4f}, {bounds['upper']:.4f}]")

    report_path = Path(args.output_dir) / "calibration_report.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "positive_class": POSITIVE_CLASS_NOTE,
        "temperature": result["temperature"],
        "fit_split": FIT_SPLIT,
        "fit_predictions_file": str(args.fit_predictions),
        "fit_counts": fit_counts,
        "eval_split": EVAL_SPLIT,
        "eval_predictions_file": str(args.eval_predictions),
        "eval_counts": eval_counts,
        "binning": {"n_bins": args.bins, "scheme": "fixed-width on [0, 1] for P(class 1)"},
        "brier_convention": "binary positive-class: mean((p - y)^2)",
        "metrics": {"uncalibrated": result["uncalibrated"], "calibrated": result["calibrated"]},
        "uncertainty": result["uncertainty"],
    }
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    if args.mlflow_experiment:
        import mlflow

        from mlflow_utils import configure_mlflow

        configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
        with mlflow.start_run(run_name=f"calibration-{args.model_name or 'custom'}"):
            mlflow.set_tags({"stage": "calibration", "model": args.model_name or "custom"})
            mlflow.log_params({"checkpoint": args.checkpoint or "", "n_bins": args.bins})
            mlflow.log_metric("temperature", result["temperature"])
            mlflow.log_metrics({
                f"test_{kind}_{metric.lower()}": value
                for kind in ("uncalibrated", "calibrated")
                for metric, value in result[kind].items()
            })
            mlflow.log_artifacts(args.output_dir, artifact_path="calibration")
            mlflow.log_artifact(temperature_path, artifact_path="calibration")

    print(f"Saved calibration artifact to: {temperature_path}")
    print(f"Saved calibration report to: {report_path}")
    print(f"Saved reliability diagram to: {Path(args.output_dir) / 'reliability_diagram.png'}")
    print(f"Saved per-bin table to: {Path(args.output_dir) / 'reliability_table.csv'}")


if __name__ == "__main__":
    main()


# Usage notes:
# - Fit T on validation data only; evaluate the before/after comparison on test data.
#   Never fit T on the test set, and do not pick a model, temperature or threshold from
#   test calibration metrics. Validation also drives checkpoint selection, so the fit is not
#   fully independent; the calibration split is reserved for conformal prediction.
# - Calibration is its own stage: training writes labelled raw logits, this script fits and
#   evaluates T and writes one canonical JSON (temperature, model, checkpoint, metrics, counts),
#   and inference.py loads that JSON.
# - Python API:
#       from calibration import fit_temperature, apply_temperature, save_temperature, load_temperature
#       T = fit_temperature(validation_logits, validation_labels)   # logits: (N, 2) raw, pre-softmax
#       save_temperature(T, "model/GCN.calibration.json", model="GCN")
#       probabilities = apply_temperature(test_logits, load_temperature("model/GCN.calibration.json", "GCN"))
# - Scaling divides logits by T, so predicted classes (argmax) and ROC-AUC/PR-AUC are unchanged.
#   T > 1 softens overconfident predictions; T < 1 sharpens underconfident ones.
# - Interpreting ECE and reliability diagrams: the test split is small (~150 molecules), so a
#   10-bin diagram has sparse bins that shift with one or two molecules. Use the 10-bin default
#   as a fixed baseline, check the per-bin counts (histogram panel / reliability_table.csv),
#   read Brier and NLL alongside ECE, and do not claim a small ECE difference is a meaningful
#   improvement. Bootstrap intervals (paired, over test molecules) are reported but ignore
#   scaffold-level dependence between molecules, so they are optimistic.
# - Sklearn models have no logits: use probabilities_to_logits(predict_proba(X)).
# - CLI input CSVs need a `label` column (0/1) plus either `logit_0`,`logit_1` or `probability`
#   (the probability of class 1, i.e. a BACE-1 inhibitor; not an arbitrary confidence score):
#       python src/calibration.py --fit-predictions validation_predictions.csv --eval-predictions test_predictions.csv --model-name GCN --output-dir model/calibration
#   The fit input must be the validation split and the evaluation input the test split; files named
#   after a different split (e.g. test_predictions.csv as the fit input) are rejected. --validation
#   and --test remain as aliases.
# - Do not divide a probability by T; convert to log-probabilities with probabilities_to_logits first.
# - Calibrated probabilities are not a validated estimate of prospective experimental success.
