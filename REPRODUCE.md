# REPRODUCE.md — Step-by-Step Reproduction Guide

This document covers how to reproduce every reported result, what each script does, and the boundary of what can and cannot be re-run from scratch.

Read the three tiers first, then jump to the section you need.

- **Tier 1 — from cached embeddings:** runnable now, CPU is enough. Reproduces the reported numbers, tables, and figures.
- **Tier 2 — from raw slides to embeddings:** code is included, but needs the WSIs (re-download from GDC/TCIA), the GigaPath weights, and a GPU. The intermediate tiles were not retained, so this is documented rather than one-command.
- **Tier 3 — GigaPath pretraining:** out of scope. GigaPath is used as released.

Throughout, `--tag full` means the real 1,025-slide run. `--tag mini` is a fast subset for smoke tests. All committed results are `full`.

All repository scripts are organized under `scripts/` and should be invoked as Python modules from the repository root, for example:

```bash
python -m scripts.training.train --model abmil --tag full
```

---

## 0. Setup

### 0.1 Data layout

Scripts read the data root from the `LUNG_WSI_DATA` environment variable. The root must contain `embeddings/` and `project/`:

```text
<base>/
  embeddings/
    gigapath_embeddings.h5      (25 GB, TCGA GigaPath, 1536-dim)
    classical_embeddings.h5     (6.7 GB, TCGA handcrafted, 404-dim)
    cptac_gigapath_v2.h5        (1.8 GB, CPTAC GigaPath, 5x)
    labels.csv
    cptac_labels_v2.csv
  project/                      (this repo)
```

Embeddings are on Google Drive (link in README). Set the root once per shell:

```bash
export LUNG_WSI_DATA=/path/to/base
```

If unset, scripts fall back to `~/research_data`.

### 0.2 Environment — option A: minimal CPU (enough for Tier 1 inference)

Verified working. Reproduces evaluation and CPTAC numbers; it is not intended for fast GPU MIL retraining.

```bash
conda create -n tcga_eval python=3.10 -y
conda activate tcga_eval
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py scikit-learn matplotlib xgboost
```

### 0.3 Environment — option B: full GPU (for retraining)

```bash
conda env create -f environment.yml
conda activate tcga_eval
```

This installs the GPU torch build and extraction/training dependencies. Use this option if you intend to retrain the MIL models or re-extract embeddings.

---

## 1. Tier 1 — Reproduce from embeddings

### 1.1 Reproduce without overwriting committed results (recommended)

To prove a re-run matches the committed numbers without touching `results/`, run in a sandbox directory. The embeddings are symlinked, checkpoints are copied, and generated outputs land fresh:

```bash
# build the sandbox once
mkdir -p ~/repro_base/project
ln -s "$LUNG_WSI_DATA/embeddings" ~/repro_base/embeddings
cp -r "$LUNG_WSI_DATA/project/"* ~/repro_base/project/
rm -rf ~/repro_base/project/results
mkdir -p ~/repro_base/project/results
cp -r "$LUNG_WSI_DATA/project/results/checkpoints" ~/repro_base/project/results/checkpoints

# run everything pointed at the sandbox
export LUNG_WSI_DATA=~/repro_base
cd ~/repro_base/project
```

Now any command below writes into `~/repro_base/project/results/`, leaving the real repo untouched. Diff the new `results/logs/.../results.json` against the committed ones to confirm they match.

To run against the real repo instead, keep `LUNG_WSI_DATA` pointed at the real base and `cd` into the real `project/` directory.

### 1.2 CPTAC external validation (fastest check)

Loads the trained checkpoints and the CPTAC embeddings. CPU, seconds.

```bash
python -m scripts.evaluation.cptac_inference --all-models
```

Expected:

```text
ABMIL (GigaPath)         CPTAC AUC 0.9593   Acc 0.8943   F1 0.8822   CM [[203,24],[19,161]]
Gated ABMIL (GigaPath)   CPTAC AUC 0.9046
MeanPool MLP (GigaPath)  CPTAC AUC 0.9391
```

Single model:

```bash
python -m scripts.evaluation.cptac_inference --model abmil
```

### 1.3 Retrain the GigaPath MIL models (GPU recommended)

```bash
python -m scripts.training.train --model abmil       --tag full
python -m scripts.training.train --model gated_abmil --tag full
python -m scripts.training.train --model meanpool_mlp --tag full
```

Defaults match the reported experiments: 50 epochs, lr 1e-4, wd 1e-4, dropout 0.25, hidden 512, 500 tiles/slide subsampled per epoch, early stop on validation AUC (patience 10). Outputs are written to `results/checkpoints/<model>_full/best.pt` and `results/logs/<model>_full/`.

On CPU this is slow but possible because the embeddings are precomputed. GPU execution is recommended for retraining.

### 1.4 Train the classical and intermediate models

```bash
# SVM and classical MLP (mean-pooled handcrafted features)
python -m scripts.training.train_classical --model both --tag full

# PCA+SVM, XGBoost, BoVW, ResNet18-cap (all or one at a time)
python -m scripts.training.train_intermediate --model all --tag full
# or: --model pca_svm | xgboost | bovw | resnet
```

BoVW vocabulary size is `--bovw-k 256` (default), matching the reported experiment.

### 1.5 Test-set table and figures

```bash
python -m scripts.evaluation.evaluate --tag full
```

The per-model test metrics are written at training time. The training entry points are:

- `scripts/training/train.py`
- `scripts/training/train_classical.py`
- `scripts/training/train_intermediate.py`

Each writes `results/logs/<model>_full/results.json`; those JSON files are committed in this repository.

`scripts/evaluation/evaluate.py` reads the committed `results.json` files and regenerates the TCGA test table and figures (ROC curves, training curves, confusion matrices, embedding PCA). It also writes `results/figures/model_comparison_full.csv`. It does not retrain and does not read checkpoints, so the `results.json` files must exist.

Expected test AUCs in the table: XGBoost 0.606, RBF-SVM 0.574, MLP 0.608, PCA+SVM 0.662, BoVW 0.717, ResNet 0.976, MeanPool 0.981, Gated 0.985, ABMIL 0.988, Ensemble 0.990.

### 1.5b Recompute TCGA test metrics from checkpoints (no retraining)

To independently recompute the GigaPath models' TCGA test metrics from trained weights, use the checkpoint evaluator. It loads `results/checkpoints/<model>_full/best.pt`, runs the repository's test forward pass, and prints AUC, accuracy, F1, and the confusion matrix. It trains nothing and writes nothing.

```bash
python -m scripts.evaluation.eval_checkpoint --model abmil       --tag full
python -m scripts.evaluation.eval_checkpoint --model gated_abmil --tag full
python -m scripts.evaluation.eval_checkpoint --model meanpool_mlp --tag full
```

Expected:

```text
abmil         TCGA test AUC 0.9878   Acc 0.9379   F1 0.9359   CM [[78,7],[3,73]]
gated_abmil   TCGA test AUC 0.9848   Acc 0.9441   F1 0.9427   CM [[78,7],[2,74]]
meanpool_mlp  TCGA test AUC 0.9807   Acc 0.9379   F1 0.9333   CM [[81,4],[6,70]]
```

The classical baselines ship no checkpoints, so their test metrics are reproduced by reading the committed `results.json` files or by retraining via `scripts.training.train_classical`.

### 1.6 Bootstrap confidence intervals

```bash
python -m scripts.evaluation.bootstrap_ci --tag full --n-bootstrap 1000 --seed 42
```

Writes `results/figures/bootstrap_ci_full.csv`. ABMIL CI [0.9721, 0.9989].

### 1.7 Learning curve (data efficiency)

```bash
python -m scripts.evaluation.learning_curve --tag full --models abmil meanpool bovw --repeats 3
```

Trains each model at 25/50/75/100/150/200/250/300 patients per class, 3 seeds. Writes `results/logs/learning_curve/learning_curve_full.json`. ABMIL is approximately 0.980 from n=25; BoVW remains below 0.73. GPU is strongly recommended because this runs many retrains.

### 1.8 Counterfactual tile removal

```bash
python -m scripts.analysis.counterfactual_tile_removal --tag full
```

Removes tiles by attention versus at random and re-scores. Writes `results/logs/counterfactual/counterfactual_full.json`. Removing the top 50% by attention changes AUC from 0.988 to 0.875; random removal remains at 0.988.

### 1.9 Attention analyses

```bash
python -m scripts.analysis.attention_stats --tag full
python -m scripts.analysis.attention_heatmap --tag full --n-slides 6
```

These produce attention-concentration statistics and spatial heatmap outputs under `results/figures/attention_heatmaps/`.

### 1.10 Regenerate figures

```bash
python -m scripts.figures.make_ieee_figures
python -m scripts.figures.make_cptac_figures
python -m scripts.figures.make_attention_grid
python -m scripts.figures.make_disagreement_attention_grid_v2
python -m scripts.figures.make_disagreement_grid_v3
```

These read committed result files and write publication-style images under `results/figures/` and `results/figures/ieee/`.

---

## 2. Tier 2 — Raw slides to embeddings (documented, needs GPU + data)

This is the stage that produced the HDF5 embeddings. It is not one-command because the raw tiles were not retained. To re-run it you need the WSIs, the GigaPath weights, and a GPU.

### 2.1 What you need

- **Raw slides.** TCGA-LUAD and TCGA-LUSC diagnostic slides from the GDC Data Portal (download with `gdc-client`). CPTAC-LUAD and CPTAC-LSCC from TCIA.
- **GigaPath weights.** Model `prov-gigapath/prov-gigapath` on Hugging Face. Requires an HF account, accepting the license, and authentication.
- **A GPU.** Tile extraction + GigaPath encoding is GPU work. CPU is impractical here.
- **The full environment** from section 0.3, including GigaPath and OpenSlide dependencies.

### 2.2 File → stage map

| Stage | Script | Input | Output |
|---|---|---|---|
| TCGA tiling + GigaPath encoding | `process_batch.py` (external data-root utility) | raw TCGA SVS | `gigapath_embeddings.h5`, `classical_embeddings.h5` |
| CPTAC download + tiling + encoding | `scripts/data_prep/cptac_download_and_extract.py` | TCIA CPTAC slides | `cptac_gigapath_v2.h5` |
| Patient-level splits | `data/splits.py` | `labels.csv` | `splits/{train,val,test}_full.csv` |

`process_batch.py` lives in the data root (`~/research_data/`), not in this repository's `project/` folder, because it operates on the raw-slide tree. The CPTAC extraction script is included in this repository.

Run the CPTAC extraction entry point from the repository root:

```bash
python -m scripts.data_prep.cptac_download_and_extract --query-only
# or run the full extraction pipeline:
python -m scripts.data_prep.cptac_download_and_extract
```

### 2.3 Tiling parameters (to match the embeddings exactly)

- Read each slide at 10x (about 1.0 micron/pixel).
- 224x224 tiles, no overlap.
- Keep a tile if mean pixel value < 220 (drops background).
- GigaPath tile encoder outputs 1536-dim; handcrafted LBP+GLCM+HOG is 404-dim.
- CPTAC was processed at its native 5x level rather than 10x; see the magnification note in section 3.

---

## 3. Data integrity notes

- **1029 vs 1025.** `gigapath_embeddings.h5` contains 1029 slides, but `labels.csv` has 1025. The 4 extra slides (61ce296a, 065e8ed3, 2873a1d0, d3519b7a) are orphans never used in any split. The experiments use the 1025 labeled set.
- **Validation vs test AUC.** `scripts/evaluation/cptac_inference.py` prints the checkpoint's TCGA validation AUC (0.9878 / 0.9848 / 0.9807). The held-out TCGA test AUCs are 0.988 / 0.985 / 0.981. Same models, different split.
- **CPTAC magnification.** CPTAC embeddings were extracted at 5x while TCGA training used 10x. The 0.959 CPTAC result therefore mixes institution and magnification shift. A 10x CPTAC re-extraction was attempted but abandoned when the LSCC source slides returned 404 errors, so the repository reports the 5x results and discloses this limitation.
- **Splits are patient-level.** No patient appears in more than one split. This is enforced in `data/splits.py`.

---

## 4. One-line reviewer check

```bash
git clone https://github.com/monkeyrampage/luad-lusc-gigapath.git project
# download embeddings from the Drive link in README into ../embeddings/
conda create -n tcga_eval python=3.10 -y && conda activate tcga_eval
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py scikit-learn matplotlib xgboost
export LUNG_WSI_DATA=$(pwd)/..      # parent holding project/ and embeddings/
cd project && python -m scripts.evaluation.cptac_inference --all-models
# -> ABMIL CPTAC AUC 0.9593
```
