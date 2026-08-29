# Lung Cancer Subtype Classification: LUAD vs LUSC

Benchmark of handcrafted features against the GigaPath pathology foundation model
for classifying lung adenocarcinoma (LUAD) vs lung squamous cell carcinoma (LUSC)
from whole-slide images (WSIs).

The headline result: attention-based MIL (ABMIL) over GigaPath embeddings reaches
**AUC 0.988** on the TCGA test set and **0.959** on an external CPTAC cohort with no
retraining. The best classical model (bag-of-visual-words + SVM) reaches only 0.717.
The tile representation, not the aggregation method, drives performance.

---

## Results

### TCGA test set (161 slides)

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| XGBoost (classical) | 0.606 | 0.553 | 0.514 |
| RBF-SVM (classical) | 0.574 | 0.553 | 0.533 |
| MLP (classical) | 0.608 | 0.596 | 0.552 |
| PCA + SVM (classical) | 0.662 | 0.677 | 0.644 |
| BoVW + SVM (classical) | 0.717 | 0.677 | 0.649 |
| ResNet18-cap MIL | 0.976 | 0.913 | 0.913 |
| MeanPool MLP (GigaPath) | 0.981 | 0.938 | 0.933 |
| Gated ABMIL (GigaPath) | 0.985 | 0.944 | 0.943 |
| **ABMIL (GigaPath)** | **0.988** | **0.938** | **0.936** |
| Ensemble (3 GigaPath) | 0.990 | 0.944 | 0.942 |

### External validation: CPTAC (407 slides, zero-shot)

| Model | TCGA AUC | CPTAC AUC | Drop | CPTAC Acc |
|---|---|---|---|---|
| MeanPool MLP | 0.981 | 0.939 | 0.042 | 0.855 |
| Gated ABMIL | 0.985 | 0.905 | 0.080 | 0.799 |
| **ABMIL** | **0.988** | **0.959** | **0.029** | **0.894** |

CPTAC was processed at 5x and TCGA at 10x, so this tests transfer across both
institution and magnification.

---

## What reproduces, and what does not

This repo ships **trained model checkpoints** and the **code**, but not the large
data. There are three levels of reproduction:

**1. From cached embeddings (fast, fully reproducible here).**
Everything downstream of the GigaPath embeddings: training every model, evaluation,
bootstrap CIs, learning curves, counterfactual analysis, CPTAC inference, and all
figures. The embeddings (HDF5) are downloaded separately (see below). This is what
the Quick Start covers and runs in seconds to minutes on a CPU.

**2. From raw slides to embeddings (code included, not run instantly).**
Going from raw WSIs to the HDF5 embeddings requires: the TCGA/CPTAC slides
(re-downloadable from GDC and TCIA), the GigaPath model weights (gated, from
HuggingFace), and a GPU. The extraction code is included but the intermediate tiles
were not retained, so this step is documented rather than one-command. See
`REPRODUCE.md`, Tier 2.

**3. GigaPath pretraining (out of scope).**
The GigaPath foundation model itself is used as released. We do not retrain it.

Full step-by-step for all three is in **`REPRODUCE.md`**.

---

## Quick Start: verify the reported numbers

Reproduces the CPTAC table in seconds. CPU is fine; no GPU needed because the
embeddings are precomputed.

```bash
# 1. clone the code
git clone https://github.com/monkeyrampage/luad-lusc-gigapath.git project

# 2. download the embeddings (see "Data" below) into a sibling folder:
#    <base>/
#      embeddings/   <- the 3 .h5 + 2 .csv files
#      project/      <- this repo

# 3. minimal CPU environment
conda create -n tcga_eval python=3.10 -y
conda activate tcga_eval
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy h5py scikit-learn matplotlib

# 4. point at the data root and run
export LUNG_WSI_DATA=/path/to/base      # the folder containing embeddings/ and project/
cd project
python3 cptac_inference.py --all-models
```

Expected output:

```
ABMIL (GigaPath)         TCGA 0.9878   CPTAC 0.9593
Gated ABMIL (GigaPath)   TCGA 0.9848   CPTAC 0.9046
MeanPool MLP (GigaPath)  TCGA 0.9807   CPTAC 0.9391
```

(The "TCGA" column here is the checkpoint's validation AUC; the test-set AUCs are in
the table above. Same models, different split.)

---

## Data

The embeddings are too large for git and are hosted separately.

**Google Drive:** https://drive.google.com/drive/folders/1Bt8cffqrvFw-DYbN3ijev-PlaqjtucIp?usp=sharing (anyone with the link)

Files needed (place all in `<base>/embeddings/`):

| File | Size | What it is |
|---|---|---|
| `gigapath_embeddings.h5` | 25 GB | TCGA GigaPath tile embeddings (1536-dim) |
| `classical_embeddings.h5` | 6.7 GB | TCGA handcrafted features (404-dim) |
| `cptac_gigapath_v2.h5` | 1.8 GB | CPTAC GigaPath embeddings (5x) |
| `labels.csv` | - | TCGA slide labels |
| `cptac_labels_v2.csv` | - | CPTAC slide labels |

To verify only the CPTAC numbers you need just `cptac_gigapath_v2.h5` and
`cptac_labels_v2.csv` plus the checkpoints (already in the repo).

The raw WSIs are not redistributed. TCGA slides are at the GDC Data Portal
(projects TCGA-LUAD, TCGA-LUSC); CPTAC slides are at TCIA (CPTAC-LUAD, CPTAC-LSCC).

---

## Repository layout

```
project/
  data/
    dataset.py            MIL dataset, 500-tile subsample (redrawn each epoch)
    splits.py             patient-level stratified splits
  models/
    model.py              ABMIL, GatedABMIL, MeanPoolMLP (input 1536, hidden 512)
  train.py                train GigaPath MIL models
  train_classical.py      XGBoost, RBF-SVM, MLP, PCA+SVM, BoVW
  train_intermediate.py   ResNet18-cap MIL, MeanPool MLP
  evaluate.py             test-set metrics for any trained model
  cptac_inference.py      zero-shot CPTAC evaluation (loads a checkpoint)
  bootstrap_ci.py         1000-sample bootstrap confidence intervals
  learning_curve.py       AUC vs training set size (3 seeds)
  counterfactual_tile_removal.py   attention-vs-random tile removal
  attention_stats.py      attention entropy / concentration
  attention_heatmap.py    per-slide attention overlays
  make_ieee_figures.py    regenerate all paper figures
  make_cptac_figures.py   CPTAC ROC + comparison figures
  cptac_download_and_extract.py    CPTAC raw -> embeddings (needs GPU + GigaPath)
  configs/                model/training configs
  splits/                 the exact train/val/test CSVs used
  results/
    checkpoints/          trained model weights (.pt) -- shipped in the repo
    logs/                 per-model results.json + training history
    figures/              all generated figures (png + pdf)
  environment.yml         full conda environment (GPU)
  requirements.txt        pip requirements
```

---

## Paths

All scripts read the data root from the `LUNG_WSI_DATA` environment variable,
defaulting to `~/research_data` if unset. The root must contain `embeddings/` and
`project/`. Set it once per shell:

```bash
export LUNG_WSI_DATA=/path/to/base
```

---

## Method summary

- **Tiles:** 224x224 at 10x, background/pen-mark filtered. ~4.1M tiles over 1,025 slides.
- **Features:** GigaPath tile encoder (1536-dim, frozen) vs handcrafted LBP+GLCM+HOG (404-dim).
- **Aggregation:** mean-pool, attention MIL (ABMIL), gated attention MIL.
- **Splits:** patient-level, no patient in more than one split (no leakage).
- **Training:** AdamW (lr 1e-4, wd 1e-4), cosine schedule, early stop on val AUC.
  500 tiles/slide subsampled per epoch (redrawn each epoch); all tiles used at test time.

See the paper and `REPRODUCE.md` for full detail.
