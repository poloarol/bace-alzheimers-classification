# BACE-1 Project — Improvement Roadmap

Improvements are ordered from easiest to most difficult. The objective is to make the current benchmark reliable and reproducible before investing in more complex modeling.

## Phase 1 — Quick Fixes

- [ ] Fix the dataset path in `config.yaml` using portable path separators.
- [ ] Clarify and validate class-label semantics: `0 = non-inhibitor`, `1 = inhibitor`.
- [ ] Update the README to reflect the current implementation, experiment protocol, and reported results.
- [ ] Record experiment metadata: random seed, split signature, software versions, model configuration, and best epoch.
- [ ] Clean up dependencies by removing unused packages and documenting supported versions.

## Phase 2 — Data Integrity

- [ ] Add stable molecule IDs to prediction files so every prediction can be traced to its source molecule.
- [ ] Validate molecule alignment across Morgan fingerprints, graph features, calibration predictions, and test predictions.
- [ ] Fix graph-cache invalidation to include raw dataset hashes, split hashes, and featurization settings.
- [ ] Audit duplicate molecules using canonical SMILES and check for conflicting labels.
- [ ] Audit scaffold splits, including partition sizes, class balance, scaffold counts, and empty scaffolds.
- [ ] Strengthen split validation so invalid candidate partitions are rejected during the search.

## Phase 3 — Training and Evaluation

- [ ] Fix batch iteration in `train.py` so the training `DataLoader` reshuffles batches between epochs.
- [ ] Improve early stopping so the patience and evaluation frequency allow training to stop before the maximum epoch.
- [ ] Standardize model selection by using validation PR-AUC consistently across classical and graph models.
- [ ] Record checkpoint provenance, including best epoch, checkpoint hash, and selection metric.
- [ ] Add integration tests for data processing to verify that molecule identities and labels remain consistent.

## Phase 4 — Calibration and Conformal Prediction

- [ ] Make `crepes` the authoritative conformal prediction implementation.
- [ ] Remove duplicate threshold logic or test that it agrees with the library's prediction-set behavior.
- [ ] Verify calibration/test separation using molecule IDs, not just matching labels.
- [ ] Test edge cases, including ties, small calibration sets, invalid probabilities, and empty prediction sets.
- [ ] Align reports with the executed method, confidence level, and configuration.
- [ ] Document class-wise coverage and scaffold-shift limitations in the final results.

## Phase 5 — Workflow and Reproducibility

- [ ] Replace Snakemake marker-only outputs with actual prediction files and reports.
- [ ] Choose one canonical experiment workflow and keep CLI and Snakemake behavior consistent.
- [ ] Test incremental execution to ensure missing or stale artifacts trigger reruns.
- [ ] Add an end-to-end smoke test covering preprocessing, training, calibration, conformal evaluation, and model comparison.
- [ ] Verify installation from a clean environment and document the complete run procedure.

## Phase 6 — Experimental Validation

- [ ] Run a clean benchmark after completing the preceding fixes.
- [ ] Repeat training across a few random seeds while holding the scaffold split fixed.
- [ ] Quantify variability in model performance and report uncertainty honestly.
- [ ] Analyze performance by scaffold novelty to better understand generalization.
- [ ] Publish a reproducible benchmark with fixed configurations, split metadata, results, and limitations.

## Recommended First Five Tasks

1. Add molecule IDs to prediction artifacts.
2. Fix graph-cache invalidation.
3. Correct training batch iteration and early stopping.
4. Finish the `crepes` integration and tests.
5. Run a clean end-to-end benchmark.

## Explicitly Deprioritized

Until the current pipeline is reliable, avoid spending time on:

- Adding more GNN architectures.
- Extensive hyperparameter searches.
- Class-conditional or scaffold-conditional conformal methods.
- Complex explainability methods.
- Expanding the dataset before the existing workflow is validated.

## Completion Criteria

The project is ready for its next research iteration when:

- [ ] The benchmark runs from a clean environment.
- [ ] Every prediction can be traced to a molecule and split.
- [ ] Cached features cannot silently become stale.
- [ ] Model selection, calibration, and evaluation use clearly documented data partitions.
- [ ] Reported metrics can be regenerated from saved artifacts.
- [ ] Results and limitations are documented well enough for another researcher to understand and reproduce the experiment.

**Guiding principle:** prioritize data integrity, reproducibility, and interpretation before adding model complexity.