"""
make_cptac_figures.py
---------------------
1. Fixed CPTAC ROC + TCGA vs CPTAC comparison figure
2. Updated bootstrap CI forest plot including CPTAC results
3. Per-class metrics table for CPTAC

Usage:
  python3 make_cptac_figures.py
"""

import json, os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample

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

BASE_DIR  = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
LOGS      = os.path.join(BASE_DIR, "project", "results", "logs")
IEEE_DIR  = os.path.join(BASE_DIR, "project", "results", "figures", "ieee")
os.makedirs(IEEE_DIR, exist_ok=True)

COLORS = {
    "abmil":        "#0077BB",
    "gated_abmil":  "#CC3311",
    "meanpool_mlp": "#009988",
}
LABELS = {
    "abmil":        "ABMIL (GigaPath)",
    "gated_abmil":  "Gated ABMIL (GigaPath)",
    "meanpool_mlp": "MeanPool MLP (GigaPath)",
}
MODELS = ["abmil", "gated_abmil", "meanpool_mlp"]


def load_cptac(model):
    path = os.path.join(LOGS, "cptac_inference", f"{model}_results.json")
    with open(path) as f:
        return json.load(f)


def load_tcga(model):
    path = os.path.join(LOGS, f"{model}_full", "results.json")
    with open(path) as f:
        r = json.load(f)
    return r.get("test", r)


def bootstrap_auc(probs, labels, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ─── Figure 1: Fixed CPTAC ROC + comparison ───────────────────────────────────
def fig_cptac_roc():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Left: ROC curves
    ax = axes[0]
    ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.4)

    for model in MODELS:
        r   = load_cptac(model)
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        ax.plot(fpr, tpr, color=COLORS[model], lw=1.5,
                label=f"{LABELS[model]} ({r['auc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — CPTAC (External)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    # Right: grouped bar TCGA vs CPTAC
    ax2   = axes[1]
    x     = np.arange(len(MODELS))
    width = 0.35

    cptac_aucs = [load_cptac(m)["auc"] for m in MODELS]
    tcga_aucs  = [load_tcga(m)["auc"]  for m in MODELS]
    xlabels    = ["ABMIL", "Gated\nABMIL", "MeanPool\nMLP"]

    b1 = ax2.bar(x - width/2, cptac_aucs, width,
                 label="CPTAC (External)",
                 color=[COLORS[m] for m in MODELS], alpha=0.9)
    b2 = ax2.bar(x + width/2, tcga_aucs, width,
                 label="TCGA (Internal)",
                 color=[COLORS[m] for m in MODELS], alpha=0.35,
                 edgecolor="black", linewidth=0.5)

    for bar, v in zip(b1, cptac_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.004,
                 f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
    for bar, v in zip(b2, tcga_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.004,
                 f"{v:.3f}", ha="center", fontsize=7, color="gray")

    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=8)
    ax2.set_ylabel("Test AUC")
    ax2.set_title("TCGA vs. CPTAC AUC")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.set_ylim([0.5, 1.06])
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("External Validation: TCGA-Trained Models on CPTAC", fontsize=10)
    fig.tight_layout()
    out = os.path.join(IEEE_DIR, "fig_cptac_roc")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved: {out}.png")


# ─── Figure 2: Bootstrap CI including CPTAC ───────────────────────────────────
def fig_bootstrap_with_cptac():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.0))

    for ax_idx, (tag, title) in enumerate([("full", "TCGA (Internal Test)"),
                                            ("cptac", "CPTAC (External)")]):
        ax    = axes[ax_idx]
        names, means, lowers, uppers, colors = [], [], [], [], []

        for model in MODELS:
            try:
                if tag == "full":
                    r      = load_tcga(model)
                    probs  = np.array(r["probs"])
                    labels = np.array(r["labels"])
                else:
                    r      = load_cptac(model)
                    probs  = np.array(r["probs"])
                    labels = np.array(r["labels"])

                mean, lo, hi = bootstrap_auc(probs, labels)
                names.append(LABELS[model].replace(" (GigaPath)", ""))
                means.append(mean)
                lowers.append(mean - lo)
                uppers.append(hi - mean)
                colors.append(COLORS[model])
            except Exception as e:
                print(f"  [SKIP] {model} {tag}: {e}")

        y_pos = np.arange(len(names))
        ax.barh(y_pos, means,
                xerr=[lowers, uppers],
                color=colors, alpha=0.75,
                error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("AUC (95% Bootstrap CI)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim([0.6, 1.05])
        ax.axvline(x=0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)

        for i, (m, u) in enumerate(zip(means, uppers)):
            ax.text(min(m + u + 0.003, 1.03), i,
                    f"{m:.3f}", va="center", fontsize=8)

    fig.suptitle("Bootstrap 95% Confidence Intervals — TCGA vs. CPTAC",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(IEEE_DIR, "fig_bootstrap_tcga_vs_cptac")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Saved: {out}.png")


# ─── Table: Per-class metrics ─────────────────────────────────────────────────
def print_perclass_metrics():
    print(f"\n{'='*75}")
    print(f"{'Model':<25} {'AUC':>6} {'Acc':>6} {'F1':>6} "
          f"{'LUAD Sens':>10} {'LUSC Sens':>10}")
    print("-" * 75)

    for model in MODELS:
        r  = load_cptac(model)
        cm = np.array(r["cm"])
        luad_sens = cm[0,0] / (cm[0,0] + cm[0,1])
        lusc_sens = cm[1,1] / (cm[1,0] + cm[1,1])
        print(f"{LABELS[model]:<25} {r['auc']:>6.3f} {r['acc']:>6.3f} "
              f"{r['f1']:>6.3f} {luad_sens:>10.3f} {lusc_sens:>10.3f}")

    print(f"\nCM format: [[TP_LUAD, FP_LUSC], [FP_LUAD, TP_LUSC]]")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("1. Fixed CPTAC ROC figure...")
    fig_cptac_roc()

    print("2. Bootstrap CI TCGA vs CPTAC...")
    fig_bootstrap_with_cptac()

    print("3. Per-class metrics on CPTAC:")
    print_perclass_metrics()

    print(f"\nAll outputs saved to {IEEE_DIR}/")


if __name__ == "__main__":
    main()
