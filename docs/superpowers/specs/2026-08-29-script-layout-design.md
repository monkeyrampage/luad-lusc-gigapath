# Script Layout Design

## Goal
Reorganize standalone research scripts into functional subdirectories while preserving scientific behavior, reported results, checkpoints, data paths, and reproducibility.

## Target layout

- `scripts/training/`: `train.py`, `train_classical.py`, `train_intermediate.py`
- `scripts/evaluation/`: `evaluate.py`, `eval_checkpoint.py`, `cptac_inference.py`, `bootstrap_ci.py`, `learning_curve.py`
- `scripts/analysis/`: `attention_stats.py`, `attention_heatmap.py`, `counterfactual_tile_removal.py`
- `scripts/figures/`: `make_attention_grid.py`, `make_cptac_figures.py`, `make_disagreement_attention_grid_v2.py`, `make_disagreement_grid_v3.py`, `make_ieee_figures.py`
- `scripts/data_prep/`: `cptac_download_and_extract.py`

Existing `data/`, `models/`, `splits/`, `results/`, dependency files, and scientific outputs remain in place.

## Execution model
Document executable scripts as Python modules from the repository root, for example `python -m scripts.training.train` and `python -m scripts.evaluation.cptac_inference`. Add `__init__.py` files to `scripts/` and each category directory.

## Documentation and quality gates
Update `README.md`, `REPRODUCE.md`, script usage headers where needed, and `.github/workflows/quality.yml`. CI must compile Python sources, ensure no legacy root-level research scripts remain, ensure the expected categorized scripts exist, and reject stale documented commands that reference the old root paths.

## Constraints
- No changes to model implementations, training/evaluation algorithms, metrics, checkpoints, split CSVs, or reported numerical results.
- No dependency-version changes.
- Preserve individual script filenames to retain provenance.
- Perform the migration on an isolated GitHub branch and merge only after GitHub Actions passes.
- Planning artifacts under `docs/superpowers/` are temporary and should not remain in the final public repository.
