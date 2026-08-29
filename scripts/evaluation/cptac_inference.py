"""
cptac_inference.py
------------------
Zero-shot generalization test: runs trained ABMIL checkpoint on CPTAC slides.
No retraining — pure inference on unseen institution data.

Outputs:
  results/logs/cptac_inference/results.json
  results/figures/ieee/fig_cptac_roc.png
  results/figures/ieee/fig_cptac_roc.pdf

Usage:
  python3 cptac_inference.py
  python3 cptac_inference.py --model gated_abmil
  python3 cptac_inference.py --all-models
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, roc_curve, auc)

rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi":     300,
    "figure.dpi":      300,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

sys.path.insert(0, os.path.dirname(__file__))
from models.model import ABMIL, GatedABMIL, MeanPoolMLP

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
EMBED_DIR  = os.path.join(BASE_DIR, "embeddings")
CPTAC_H5   = os.path.join(EMBED_DIR, "cptac_gigapath_v2.h5")
CPTAC_CSV  = os.path.join(EMBED_DIR, "cptac_labels_v2.csv")
CKPT_DIR   = os.path.join(BASE_DIR, "project", "results", "checkpoints")
LOG_DIR    = os.path.join(BASE_DIR, "project", "results", "logs", "cptac_inference")
IEEE_DIR   = os.path.join(BASE_DIR, "project", "results", "figures", "ieee")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(IEEE_DIR, exist_ok=True)

# Model registry
MODELS = {
    "abmil":        (ABMIL,       "ABMIL (GigaPath)",        "#0077BB"),
    "gated_abmil":  (GatedABMIL,  "Gated ABMIL (GigaPath)",  "#CC3311"),
    "meanpool_mlp": (MeanPoolMLP, "MeanPool MLP (GigaPath)",  "#009988"),
}


def load_model(model_name, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_full", "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cls, _, _ = MODELS[model_name]
    ckpt      = torch.load(ckpt_path, map_location=device)
    model     = cls(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded {model_name} (val AUC={ckpt['val_auc']:.4f} on TCGA)")
    return model


def load_cptac_data():
    """Load CPTAC labels and check H5 availability."""
    import h5py

    with open(CPTAC_CSV) as f:
        rows = list(csv.DictReader(f))

    with h5py.File(CPTAC_H5, "r") as h:
        available = set(h.keys())

    valid_rows = [r for r in rows if r["slide_id"] in available]
    print(f"  CPTAC slides in CSV:  {len(rows)}")
    print(f"  Available in H5:      {len(available)}")
    print(f"  Valid for inference:  {len(valid_rows)}")

    luad = sum(1 for r in valid_rows if r["label"] == "LUAD")
    lusc = sum(1 for r in valid_rows if r["label"] == "LUSC")
    print(f"  LUAD: {luad}  LUSC: {lusc}")

    return valid_rows


def run_inference(model, rows, device):
    """Run ABMIL inference on all CPTAC slides."""
    import h5py

    probs, labels = [], []

    with h5py.File(CPTAC_H5, "r") as h:
        for r in rows:
            fid   = r["slide_id"]
            label = int(r["label_int"])

            features = h[fid]["features"][:]
            bag      = torch.tensor(features, dtype=torch.float32).to(device)

            with torch.inference_mode():
                logits, _ = model(bag)

            prob = torch.softmax(logits, dim=0)[1].item()
            probs.append(prob)
            labels.append(label)

    return np.array(probs), np.array(labels)


def compute_metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    return {
        "auc":       float(roc_auc_score(labels, probs)),
        "acc":       float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "cm":        confusion_matrix(labels, preds).tolist(),
        "probs":     probs.tolist(),
        "labels":    labels.tolist(),
    }


def plot_roc_comparison(all_results, tcga_results=None):
    """Plot CPTAC ROC curves, optionally with TCGA comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Left: CPTAC ROC curves
    ax = axes[0]
    ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.4)

    for model_name, res in all_results.items():
        _, label, color = MODELS[model_name]
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        auc_val     = res["auc"]
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{label} ({auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — CPTAC (External Validation)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    # Right: TCGA vs CPTAC AUC comparison bar chart
    ax2 = axes[1]
    model_names = list(all_results.keys())
    cptac_aucs  = [all_results[m]["auc"] for m in model_names]
    labels_text = [MODELS[m][1].replace(" (GigaPath)", "") for m in model_names]
    colors      = [MODELS[m][2] for m in model_names]
    x           = np.arange(len(model_names))
    width       = 0.35

    bars1 = ax2.bar(x - width/2, cptac_aucs, width,
                    label="CPTAC (External)", color=colors, alpha=0.9)

    if tcga_results:
        tcga_aucs = [tcga_results.get(m, {}).get("test_auc", 0)
                     for m in model_names]
        bars2 = ax2.bar(x + width/2, tcga_aucs, width,
                        label="TCGA (Internal)", color=colors,
                        alpha=0.4, edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_text, fontsize=7, rotation=10)
    ax2.set_ylabel("Test AUC")
    ax2.set_title("TCGA vs. CPTAC AUC")
    ax2.legend(fontsize=7)
    ax2.set_ylim([0.5, 1.05])

    # Value labels on bars
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", fontsize=7)

    fig.suptitle("External Validation: TCGA-Trained Models on CPTAC",
                 fontsize=10)
    fig.tight_layout()

    out = os.path.join(IEEE_DIR, "fig_cptac_roc")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"\nPlot saved: {out}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="abmil",
                        choices=list(MODELS.keys()))
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading CPTAC data...")
    rows = load_cptac_data()

    if not rows:
        print("No CPTAC slides available yet — run cptac_download_and_extract.py first")
        return

    models_to_run = list(MODELS.keys()) if args.all_models else [args.model]
    all_results   = {}

    for model_name in models_to_run:
        print(f"\nRunning inference: {model_name}...")
        try:
            model  = load_model(model_name, device)
            probs, labels = run_inference(model, rows, device)
            metrics       = compute_metrics(probs, labels)
            all_results[model_name] = metrics

            print(f"  AUC:  {metrics['auc']:.4f}")
            print(f"  Acc:  {metrics['acc']:.4f}")
            print(f"  F1:   {metrics['f1']:.4f}")
            print(f"  CM:   {metrics['cm']}")

            # Save per-model results
            out_path = os.path.join(LOG_DIR, f"{model_name}_results.json")
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Load TCGA results for comparison
    tcga_results = {}
    for model_name in models_to_run:
        tcga_path = os.path.join(BASE_DIR, "project", "results",
                                 "logs", f"{model_name}_full", "results.json")
        if os.path.exists(tcga_path):
            with open(tcga_path) as f:
                r = json.load(f)
            tcga_results[model_name] = {
                "test_auc": r.get("test", r).get("auc", 0)
            }

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Model':<25} {'TCGA AUC':>10} {'CPTAC AUC':>10} {'Drop':>8}")
    print("-" * 65)
    for model_name in models_to_run:
        if model_name not in all_results:
            continue
        cptac_auc = all_results[model_name]["auc"]
        tcga_auc  = tcga_results.get(model_name, {}).get("test_auc", 0)
        drop      = tcga_auc - cptac_auc
        print(f"{MODELS[model_name][1]:<25} {tcga_auc:>10.4f} "
              f"{cptac_auc:>10.4f} {drop:>+8.4f}")

    # Plot
    if all_results:
        plot_roc_comparison(all_results, tcga_results)

    # Save combined results
    out_path = os.path.join(LOG_DIR, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved: {out_path}")


if __name__ == "__main__":
    main()
