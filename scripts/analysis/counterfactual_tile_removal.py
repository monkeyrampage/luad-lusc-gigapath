"""
counterfactual_tile_removal.py
-------------------------------
Validates ABMIL attention maps by progressively removing top-attended tiles
and measuring AUC drop on the test set.

Experiment:
  For each removal fraction k in [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
    1. Run ABMIL forward pass to get attention weights
    2. Remove top-k% highest attention tiles
    3. Re-run inference on remaining tiles
    4. Record AUC

Also runs random tile removal as control (same fractions, random tiles).

If AUC drops faster with attention removal than random removal,
the attention maps are causally meaningful.

Output:
  results/figures/ieee/fig_counterfactual_full.png
  results/logs/counterfactual/counterfactual_full.json

Usage:
  python -m scripts.analysis.counterfactual_tile_removal --tag full
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
from sklearn.metrics import roc_auc_score, accuracy_score

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
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

sys.path.insert(0, os.path.dirname(__file__))
from models.model import ABMIL

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
GP_H5      = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
CKPT_PATH  = os.path.join(BASE_DIR, "project", "results",
                           "checkpoints", "abmil_full", "best.pt")
SPLITS_DIR = os.path.join(BASE_DIR, "project", "splits")
OUT_DIR    = os.path.join(BASE_DIR, "project", "results", "figures", "ieee")
LOG_DIR    = os.path.join(BASE_DIR, "project", "results", "logs", "counterfactual")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# Removal fractions to test
REMOVAL_FRACTIONS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
N_RANDOM_SEEDS    = 5   # repeat random removal N times for stable estimate


def load_model(device):
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model = ABMIL(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded checkpoint (val AUC={ckpt['val_auc']:.4f})")
    return model


def run_inference(model, features, device):
    """Full forward pass — returns (prob, attn_weights)."""
    bag = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.inference_mode():
        logits, attn = model(bag)
    prob = torch.softmax(logits, dim=0)[1].item()
    return prob, attn.cpu().numpy()


def run_inference_subset(model, features, keep_mask, device):
    """Inference on subset of tiles defined by keep_mask."""
    subset = features[keep_mask]
    if len(subset) == 0:
        return 0.5   # random guess if all tiles removed
    bag = torch.tensor(subset, dtype=torch.float32).to(device)
    with torch.inference_mode():
        logits, _ = model(bag)
    return torch.softmax(logits, dim=0)[1].item()


def evaluate_removal(model, all_features, all_labels, all_attns,
                     fraction, mode, device, seed=None):
    """
    Remove top-k% tiles by attention (mode='attention') or randomly (mode='random').
    Returns AUC on remaining tiles.
    """
    if seed is not None:
        np.random.seed(seed)

    probs = []
    for features, attn in zip(all_features, all_attns):
        n = len(features)
        n_remove = max(0, int(n * fraction))

        if n_remove >= n:
            probs.append(0.5)
            continue

        if mode == "attention":
            # Remove highest attention tiles
            remove_idx = np.argsort(attn)[-n_remove:] if n_remove > 0 else []
        else:
            # Remove random tiles
            remove_idx = np.random.choice(n, n_remove, replace=False) if n_remove > 0 else []

        keep_mask = np.ones(n, dtype=bool)
        if len(remove_idx) > 0:
            keep_mask[remove_idx] = False

        prob = run_inference_subset(model, features, keep_mask, device)
        probs.append(prob)

    auc = roc_auc_score(all_labels, probs)
    acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in probs])
    return auc, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="full")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading model...")
    model = load_model(device)

    # Load test split
    suffix   = f"_{args.tag}" if args.tag else ""
    test_csv = os.path.join(SPLITS_DIR, f"test{suffix}.csv")
    with open(test_csv) as f:
        rows = list(csv.DictReader(f))

    # Pre-load all features and compute attention weights
    import h5py
    print(f"\nPre-computing attention weights for {len(rows)} test slides...")
    all_features, all_labels, all_attns = [], [], []

    with h5py.File(GP_H5, "r") as h:
        for r in rows:
            fid = r["file_id"]
            if fid not in h:
                continue
            features = h[fid]["features"][:]
            _, attn  = run_inference(model, features, device)
            all_features.append(features)
            all_labels.append(int(r["label_int"]))
            all_attns.append(attn)

    all_labels = np.array(all_labels)
    print(f"  {len(all_features)} slides loaded")

    # Baseline AUC (no removal)
    baseline_probs = []
    for features, attn in zip(all_features, all_attns):
        bag = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.inference_mode():
            logits, _ = model(bag)
        baseline_probs.append(torch.softmax(logits, dim=0)[1].item())
    baseline_auc = roc_auc_score(all_labels, baseline_probs)
    print(f"  Baseline AUC: {baseline_auc:.4f}")

    # Run experiment
    results = {
        "baseline_auc": baseline_auc,
        "attention":    {},
        "random":       {},
    }

    print(f"\n{'Fraction':>10} {'Attn AUC':>10} {'Rand AUC':>10} {'Attn Drop':>11} {'Rand Drop':>11}")
    print("-" * 56)

    for frac in REMOVAL_FRACTIONS:
        # Attention removal
        attn_auc, attn_acc = evaluate_removal(
            model, all_features, all_labels, all_attns,
            frac, "attention", device)

        # Random removal — average over N seeds
        rand_aucs = []
        for seed in range(N_RANDOM_SEEDS):
            rauc, _ = evaluate_removal(
                model, all_features, all_labels, all_attns,
                frac, "random", device, seed=seed)
            rand_aucs.append(rauc)
        rand_auc = float(np.mean(rand_aucs))
        rand_std = float(np.std(rand_aucs))

        results["attention"][str(frac)] = {
            "auc": attn_auc, "acc": attn_acc,
            "drop": baseline_auc - attn_auc
        }
        results["random"][str(frac)] = {
            "auc": rand_auc, "std": rand_std,
            "drop": baseline_auc - rand_auc
        }

        print(f"{frac:>10.0%} {attn_auc:>10.4f} {rand_auc:>10.4f} "
              f"{baseline_auc-attn_auc:>+11.4f} {baseline_auc-rand_auc:>+11.4f}")

    # Save results
    out_json = os.path.join(LOG_DIR, f"counterfactual_{args.tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot
    fracs       = [float(k) for k in results["attention"].keys()]
    attn_aucs   = [results["attention"][str(f)]["auc"]  for f in fracs]
    rand_aucs   = [results["random"][str(f)]["auc"]     for f in fracs]
    rand_stds   = [results["random"][str(f)]["std"]     for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Left: AUC vs removal fraction
    ax = axes[0]
    ax.plot([f*100 for f in fracs], attn_aucs,
            color="#0077BB", marker="o", lw=1.5, label="Attention-guided removal")
    ax.plot([f*100 for f in fracs], rand_aucs,
            color="#CC3311", marker="s", lw=1.5, linestyle="--",
            label="Random removal (mean)")
    ax.fill_between([f*100 for f in fracs],
                    [r-s for r,s in zip(rand_aucs, rand_stds)],
                    [r+s for r,s in zip(rand_aucs, rand_stds)],
                    alpha=0.15, color="#CC3311")
    ax.axhline(y=baseline_auc, color="gray", linestyle=":", lw=0.8)
    ax.text(2, baseline_auc+0.003, f"Baseline ({baseline_auc:.3f})",
            fontsize=7, color="gray")
    ax.set_xlabel("Tiles Removed (%)")
    ax.set_ylabel("Test AUC")
    ax.set_title("AUC vs. Tile Removal")
    ax.legend(fontsize=7)
    ax.set_xlim([-1, 52])
    ax.set_ylim([0.5, 1.02])

    # Right: AUC drop
    ax2 = axes[1]
    attn_drops = [baseline_auc - a for a in attn_aucs]
    rand_drops = [baseline_auc - r for r in rand_aucs]

    ax2.plot([f*100 for f in fracs], attn_drops,
             color="#0077BB", marker="o", lw=1.5, label="Attention-guided")
    ax2.plot([f*100 for f in fracs], rand_drops,
             color="#CC3311", marker="s", lw=1.5, linestyle="--",
             label="Random")
    ax2.axhline(y=0, color="gray", linestyle=":", lw=0.8)
    ax2.set_xlabel("Tiles Removed (%)")
    ax2.set_ylabel("AUC Drop (from baseline)")
    ax2.set_title("AUC Drop vs. Tile Removal")
    ax2.legend(fontsize=7)
    ax2.set_xlim([-1, 52])

    fig.suptitle("Counterfactual Tile Removal — Attention Map Validation",
                 fontsize=10)
    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, f"fig_counterfactual_{args.tag}.png")
    out_pdf = os.path.join(OUT_DIR, f"fig_counterfactual_{args.tag}.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Plot saved: {out_png}")


if __name__ == "__main__":
    main()
