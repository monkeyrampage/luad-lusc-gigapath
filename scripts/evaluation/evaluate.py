"""
evaluate.py
-----------
Generates all evaluation outputs needed for benchmark evaluation.
Run after training all models.

Outputs (in results/figures/):
  roc_curves.png          — ROC curves for all 5 models
  training_curves.png     — Loss + AUC per epoch for GigaPath models
  confusion_matrices.png  — Confusion matrix grid
  embedding_pca.png       — PCA of GigaPath embeddings colored by label
  attention_heatmap.png   — Top/bottom attention tiles for best ABMIL slide
  model_comparison.csv    — Full metrics table for all models

Usage:
  python3 evaluate.py --tag mini
  python3 evaluate.py --tag full
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR    = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
RESULTS_DIR = os.path.join(BASE_DIR, "project", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colors per model
MODEL_COLORS = {
    "abmil":         "#2196F3",
    "gated_abmil":   "#9C27B0",
    "meanpool_mlp":  "#4CAF50",
    "classical_mlp": "#FF9800",
    "svm":           "#F44336",
    "pca_svm":       "#FF5722",
    "xgboost":       "#795548",
    "bovw":          "#FFC107",
    "resnet_mil":    "#00BCD4",
}

MODEL_LABELS = {
    "abmil":         "ABMIL (GigaPath)",
    "gated_abmil":   "Gated ABMIL (GigaPath)",
    "meanpool_mlp":  "MeanPool MLP (GigaPath)",
    "resnet_mil":    "ResNet18-cap MIL (GigaPath)",
    "classical_mlp": "MLP (Classical)",
    "bovw":          "BoVW + SVM (Classical)",
    "pca_svm":       "PCA + SVM (Classical)",
    "svm":           "RBF-SVM (Classical)",
    "xgboost":       "XGBoost (Classical)",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_results(model_name, tag):
    path = os.path.join(RESULTS_DIR, "logs", f"{model_name}_{tag}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_history(model_name, tag):
    path = os.path.join(RESULTS_DIR, "logs", f"{model_name}_{tag}", "history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_probs_labels(results, split="test"):
    """Extract probs and labels from results dict."""
    if split in results:
        return results[split]["probs"], results[split]["labels"]
    # GigaPath model results are flat (not nested by split)
    return results.get("probs"), results.get("labels")


# ─── Plot 1: ROC Curves ───────────────────────────────────────────────────────

def plot_roc_curves(tag, models):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    for model_name in models:
        res = load_results(model_name, tag)
        if res is None:
            print(f"  [SKIP] {model_name} — no results found")
            continue

        # Handle both flat and nested result formats
        if "test" in res:
            probs  = res["test"]["probs"]
            labels = res["test"]["labels"]
            roc_auc_val = res["test"]["auc"]
        else:
            probs  = res.get("probs", [])
            labels = res.get("labels", [])
            roc_auc_val = res.get("test_auc", 0)

        if not probs:
            continue

        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val     = auc(fpr, tpr)
        label       = f"{MODEL_LABELS[model_name]} (AUC={auc_val:.4f})"
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[model_name], label=label)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {tag} split", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"roc_curves_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─── Plot 2: Training Curves ──────────────────────────────────────────────────

def plot_training_curves(tag, models):
    gp_models = [m for m in models if m in ("abmil", "gated_abmil", "meanpool_mlp")]
    n = len(gp_models)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, model_name in enumerate(gp_models):
        hist = load_history(model_name, tag)
        if hist is None:
            continue

        epochs    = [h["epoch"]     for h in hist]
        tr_loss   = [h["train_loss"] for h in hist]
        val_loss  = [h["val_loss"]   for h in hist]
        tr_auc    = [h["train_auc"]  for h in hist]
        val_auc   = [h["val_auc"]    for h in hist]

        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]

        # Loss
        ax = axes[i][0]
        ax.plot(epochs, tr_loss,  label="Train", color=color,    lw=2)
        ax.plot(epochs, val_loss, label="Val",   color=color,    lw=2, linestyle="--")
        ax.set_title(f"{label} — Loss", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # AUC
        ax = axes[i][1]
        ax.plot(epochs, tr_auc,  label="Train", color=color, lw=2)
        ax.plot(epochs, val_auc, label="Val",   color=color, lw=2, linestyle="--")
        ax.set_title(f"{label} — AUC", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim([0.5, 1.02])
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Training Curves — {tag} split", fontsize=14, y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"training_curves_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── Plot 3: Confusion Matrices ───────────────────────────────────────────────

def plot_confusion_matrices(tag, models):
    valid = []
    for m in models:
        res = load_results(m, tag)
        if res is None:
            continue
        cm = res["test"]["cm"] if "test" in res else res.get("cm")
        if cm:
            valid.append((m, np.array(cm)))

    if not valid:
        return

    n   = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, valid):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["LUAD", "LUSC"])
        ax.set_yticklabels(["LUAD", "LUSC"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(MODEL_LABELS[model_name], fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    fig.suptitle(f"Confusion Matrices (Test) — {tag}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"confusion_matrices_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─── Plot 4: Embedding PCA ────────────────────────────────────────────────────

def plot_embedding_pca(tag):
    import h5py
    from data.dataset import load_split_csv

    SPLITS_DIR = os.path.join(BASE_DIR, "project", "splits")
    GP_H5      = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
    suffix     = f"_{tag}" if tag else ""
    test_csv   = os.path.join(SPLITS_DIR, f"test{suffix}.csv")

    if not os.path.exists(test_csv):
        print(f"  [SKIP] PCA — {test_csv} not found")
        return

    rows = load_split_csv(test_csv)
    X, labels = [], []

    with h5py.File(GP_H5, "r") as h:
        for r in rows:
            fid = r["file_id"]
            if fid not in h:
                continue
            feats = h[fid]["features"][:].mean(axis=0)  # mean pool slide
            X.append(feats)
            labels.append(r["label"])

    X = np.stack(X)
    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl, color in [("LUAD", "#2196F3"), ("LUSC", "#F44336")]:
        mask = [l == lbl for l in labels]
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   label=lbl, c=color, alpha=0.6, s=20, edgecolors="none")

    ax.set_title(f"PCA of GigaPath Embeddings (Test Set) — {tag}", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"embedding_pca_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─── Table: Model Comparison CSV ─────────────────────────────────────────────

def save_comparison_table(tag, models):
    rows = []
    for model_name in models:
        res = load_results(model_name, tag)
        if res is None:
            continue
        if "test" in res:
            t = res["test"]
        else:
            t = res

        rows.append({
            "model":     MODEL_LABELS[model_name],
            "val_auc":   round(res["val"]["auc"]  if "val" in res else 0, 4),
            "test_auc":  round(t.get("auc",  0), 4),
            "test_acc":  round(t.get("acc",  0), 4),
            "precision": round(t.get("precision", 0), 4),
            "recall":    round(t.get("recall", 0), 4),
            "f1":        round(t.get("f1", 0), 4),
        })

    if not rows:
        return

    out = os.path.join(FIGURES_DIR, f"model_comparison_{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Print table
    print(f"\n{'Model':<30} {'Val AUC':>8} {'Test AUC':>9} "
          f"{'Acc':>7} {'F1':>7}")
    print("-" * 65)
    for r in rows:
        print(f"{r['model']:<30} {r['val_auc']:>8.4f} {r['test_auc']:>9.4f} "
              f"{r['test_acc']:>7.4f} {r['f1']:>7.4f}")

    print(f"\n  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="mini")
    args = parser.parse_args()

    tag    = args.tag
    models = ["abmil", "gated_abmil", "meanpool_mlp", "resnet_mil", "bovw", "classical_mlp", "pca_svm", "svm", "xgboost"]

    print(f"Generating evaluation outputs for tag='{tag}'...")
    print(f"Output dir: {FIGURES_DIR}\n")

    print("1. ROC curves...")
    plot_roc_curves(tag, models)

    print("2. Training curves...")
    plot_training_curves(tag, models)

    print("3. Confusion matrices...")
    plot_confusion_matrices(tag, models)

    print("4. Embedding PCA...")
    plot_embedding_pca(tag)

    print("5. Model comparison table...")
    save_comparison_table(tag, models)

    print(f"\nDone. All figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
