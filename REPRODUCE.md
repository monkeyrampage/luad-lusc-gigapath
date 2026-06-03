# REPRODUCE.md — Step-by-Step Reproduction Guide

This document covers how to reproduce every result, what each script does, and the
honest boundary of what can and cannot be re-run from scratch.

Read the three tiers first, then jump to the section you need.

- **Tier 1 — from cached embeddings:** runnable now, CPU is enough. Reproduces every
  number, table, and figure in the paper. Most of this document.
- **Tier 2 — from raw slides to embeddings:** code is included, but needs the WSIs
  (re-download from GDC/TCIA), the GigaPath weights, and a GPU. The intermediate tiles
  were not retained, so this is documented, not one-command.
- **Tier 3 — GigaPath pretraining:** out of scope. GigaPath is used as released.

Throughout, `--tag full` means the real 1,025-slide run (what the paper reports).
`--tag mini` is a fast subset for smoke tests. All committed results are `full`.

---

## 0. Setup

### 0.1 Data layout

Scripts read the data root from the `DSAI543_DATA` environment variable. The root
must contain `embeddings/` and `project/`:

```
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
export DSAI543_DATA=/path/to/base
```

If unset, scripts fall back to `~/research_data`.

### 0.2 Environment — option A: minimal CPU (enough for Tier 1 inference)

Verified working. Reproduces all evaluation and CPTAC numbers; cannot retrain the
GPU MIL models quickly but can run everything that only reads embeddings.

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

This installs the GPU torch build (CUDA 12.8) and all extraction/training deps.
Use this only if you intend to retrain the MIL models or re-extract embeddings.

---

## 1. Tier 1 — Reproduce from embeddings

### 1.1 Reproduce without overwriting committed results (recommended)

To prove a re-run matches the committed numbers without touching `results/`, run in a
sandbox. The embeddings are symlinked (no re-copy), checkpoints are copied (so
inference can read them), and the generated outputs land fresh:

```bash
# build the sandbox once
mkdir -p ~/repro_base/project
ln -s "$DSAI543_DATA/embeddings" ~/repro_base/embeddings
cp -r "$DSAI543_DATA/project/"* ~/repro_base/project/
rm -rf ~/repro_base/project/results
mkdir -p ~/repro_base/project/results
cp -r "$DSAI543_DATA/project/results/checkpoints" ~/repro_base/project/results/checkpoints

# run everything pointed at the sandbox
export DSAI543_DATA=~/repro_base
cd ~/repro_base/project
```

Now any command below writes into `~/repro_base/project/results/`, leaving the real
repo untouched. Diff the new `results/logs/.../results.json` against the committed
ones to confirm they match.

To run against the real repo instead, just keep `DSAI543_DATA` pointed at the real
base and `cd` into the real `project/`.

### 1.2 CPTAC external validation (fastest check)

Loads the trained checkpoints and the CPTAC embeddings. CPU, seconds.

```bash
python3 cptac_inference.py --all-models
```

Expected:

```
ABMIL (GigaPath)         CPTAC AUC 0.9593   Acc 0.8943   F1 0.8822   CM [[203,24],[19,161]]
Gated ABMIL (GigaPath)   CPTAC AUC 0.9046
MeanPool MLP (GigaPath)  CPTAC AUC 0.9391
```

Single model: `python3 cptac_inference.py --model abmil`.

### 1.3 Retrain the GigaPath MIL models (GPU recommended)

```bash
python3 train.py --model abmil       --tag full
python3 train.py --model gated_abmil --tag full
python3 train.py --model meanpool_mlp --tag full
```

Defaults match the paper: 50 epochs, lr 1e-4, wd 1e-4, dropout 0.25, hidden 512,
500 tiles/slide subsampled per epoch, early stop on val AUC (patience 10). Writes
checkpoints to `results/checkpoints/<model>_full/best.pt` and logs to
`results/logs/<model>_full/`.

On CPU this is slow but possible (the embeddings are precomputed, so it is matrix
work, not image encoding). On the project GPU each model trains in minutes.

### 1.4 Train the classical and intermediate models

```bash
# SVM and classical MLP (mean-pooled handcrafted features)
python3 train_classical.py --model both --tag full

# PCA+SVM, XGBoost, BoVW, ResNet18-cap (run all, or one at a time)
python3 train_intermediate.py --model all --tag full
# or: --model pca_svm | xgboost | bovw | resnet
```

BoVW vocabulary size is `--bovw-k 256` (default), matching the paper.

### 1.5 Test-set evaluation

```bash
python3 evaluate.py --tag full
```

Reproduces the TCGA test table (AUC/accuracy/F1 for every model). Reads the trained
checkpoints and writes `results/logs/<model>_full/results.json`. Expected AUCs:
XGBoost 0.606, RBF-SVM 0.574, MLP 0.608, PCA+SVM 0.662, BoVW 0.717, ResNet 0.976,
MeanPool 0.981, Gated 0.985, ABMIL 0.988, Ensemble 0.990.

### 1.6 Bootstrap confidence intervals

```bash
python3 bootstrap_ci.py --tag full --n-bootstrap 1000 --seed 42
```

Writes `results/figures/bootstrap_ci_full.csv`. ABMIL CI [0.9721, 0.9989].

### 1.7 Learning curve (data efficiency)

```bash
python3 learning_curve.py --tag full --models abmil meanpool bovw --repeats 3
```

Trains each model at 25/50/75/100/150/200/250/300 patients per class, 3 seeds.
Writes `results/logs/learning_curve/learning_curve_full.json`. ABMIL ~0.980 from
n=25; BoVW never exceeds 0.73. (GPU strongly recommended — this is many retrains.)

### 1.8 Counterfactual tile removal

```bash
python3 counterfactual_tile_removal.py --tag full
```

Removes tiles by attention vs at random and re-scores. Writes
`results/logs/counterfactual/counterfactual_full.json`. Top-50%-by-attention
removed: AUC 0.988 -> 0.875; random removal stays 0.988.

### 1.9 Regenerate all figures

```bash
python3 make_ieee_figures.py      # ROC, bootstrap, confusion, learning curve,
                                   # counterfactual, attention heatmaps
python3 make_cptac_figures.py     # CPTAC ROC + TCGA-vs-CPTAC comparison
```

These read the result JSONs and write PNG/PDF into `results/figures/ieee/`.
(`make_ieee_figures.py` and `make_cptac_figures.py` take no required flags.)

---

## 2. Tier 2 — Raw slides to embeddings (documented, needs GPU + data)

This is the step that produced the HDF5 embeddings. It is **not** one-command now,
because the raw tiles were not retained. To re-run it you need: the WSIs, the
GigaPath weights, and a GPU. The code is included.

### 2.1 What you need

- **Raw slides.** TCGA-LUAD and TCGA-LUSC diagnostic slides from the GDC Data Portal
  (download with `gdc-client`). CPTAC-LUAD and CPTAC-LSCC from TCIA.
- **GigaPath weights.** Gated model `prov-gigapath/prov-gigapath` on HuggingFace.
  Requires a HF account, accepting the license, and `huggingface-cli login`.
- **A GPU.** Tile extraction + GigaPath encoding is GPU work. CPU is impractical here.
- **The full environment** (section 0.3), which includes the GigaPath and OpenSlide deps.

### 2.2 File -> stage map

| Stage | Script | Input | Output |
|---|---|---|---|
| TCGA tiling + GigaPath encoding | `process_batch.py` (on the VM, in `~/research_data/`) | raw TCGA SVS | `gigapath_embeddings.h5`, `classical_embeddings.h5` |
| CPTAC download + tiling + encoding | `cptac_download_and_extract.py` | TCIA CPTAC slides | `cptac_gigapath_v2.h5` |
| Patient-level splits | `data/splits.py` | `labels.csv` | `splits/{train,val,test}_full.csv` |

Note: `process_batch.py` lives in the data root (`~/research_data/`), not in this
repo's `project/` folder, because it operates on the raw-slide tree. The CPTAC
extraction script is included in the repo.

### 2.3 Tiling parameters (to match the embeddings exactly)

- Read each slide at 10x (about 1.0 micron/pixel).
- 224x224 tiles, no overlap.
- Keep a tile if mean pixel value < 220 (drops background).
- GigaPath tile encoder outputs 1536-dim; handcrafted LBP+GLCM+HOG is 404-dim.
- CPTAC was processed at its native 5x level (not 10x) — see the magnification note
  in section 3.

---

## 3. Data integrity notes (so nothing surprises you)

- **1029 vs 1025.** `gigapath_embeddings.h5` contains 1029 slides, but `labels.csv`
  has 1025. The 4 extra (61ce296a, 065e8ed3, 2873a1d0, d3519b7a) are orphans never
  used in any split. The experiments use the 1025 labeled set. The paper reports 1025.
- **Val vs test AUC.** `cptac_inference.py` prints the checkpoint's TCGA *validation*
  AUC (0.9878 / 0.9848 / 0.9807). The paper's Table II reports TCGA *test* AUC
  (0.988 / 0.985 / 0.981). Same models, different split. Not a discrepancy.
- **CPTAC magnification.** CPTAC embeddings were extracted at 5x while TCGA training
  used 10x. The 0.959 CPTAC result therefore mixes institution and magnification
  shift. A 10x CPTAC re-extraction was attempted but abandoned (the LSCC slides
  404'd on the source server), so the paper reports the 5x numbers and discloses
  this as a limitation.
- **Splits are patient-level.** No patient appears in more than one split. This is
  enforced in `data/splits.py` and is why test AUC is trustworthy (no leakage).

---

## 4. One-line summary for a reviewer

```bash
git clone https://github.com/monkeyrampage/luad-lusc-gigapath.git project
# download embeddings from the Drive link in README into ../embeddings/
conda create -n tcga_eval python=3.10 -y && conda activate tcga_eval
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py scikit-learn matplotlib xgboost
export DSAI543_DATA=$(pwd)/..      # parent holding project/ and embeddings/
cd project && python3 cptac_inference.py --all-models
# -> ABMIL CPTAC AUC 0.9593, matching the paper
```
