"""
bootstrap_ci.py
---------------
Computes bootstrap 95% confidence intervals for AUC, accuracy, and F1
for all trained models on the test set.

Outputs:
  results/figures/bootstrap_ci.csv     — full CI table
  results/figures/bootstrap_ci.png     — forest plot of AUC CIs

Usage:
  python -m scripts.evaluation.bootstrap_ci --tag full --n-bootstrap 1000
  python -m scripts.evaluation.bootstrap_ci --tag mini --n-bootstrap 1000
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

BASE_DIR    = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
RESULTS_DIR = os.path.join(BASE_DIR, "project", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Model display names and order (worst to best for forest plot)
MODELS = [
    ("xgboost",        "XGBoost (Classical)"),
    ("svm",            "RBF-SVM (Classical)"),
    ("classical_mlp",  "MLP (Classical)"),
    ("pca_svm",        "PCA + SVM (Classical)"),
    ("bovw",           "BoVW + SVM (Classical)"),
    ("resnet_mil",     "ResNet18-cap MIL"),
    ("meanpool_mlp",   "MeanPool MLP (GigaPath)"),
    ("gated_abmil",    "Gated ABMIL (GigaPath)"),
    ("abmil",          "ABMIL (GigaPath)"),
]


def load_test_data(model_name, tag):
    """Load probs and labels from saved results.json."""
    path = os.path.join(RESULTS_DIR, "logs", f"{model_name}_{tag}", "results.json")
    if not os.path.exists(path):
        return None, None

    with open(path) as f:
        res = json.load(f)

    # Handle nested format (GigaPath models + classical)
    if "test" in res:
        probs  = res["test"]["probs"]
        labels = res["test"]["labels"]
    else:
        probs  = res.get("probs")
        labels = res.get("labels")

    if not probs or not labels:
        return None, None

    return np.array(probs), np.array(labels)


def bootstrap_metrics(probs, labels, n_bootstrap=1000, seed=42):
    """
    Bootstrap confidence intervals for AUC, accuracy, F1.
    Returns dict with mean, ci_lower, ci_upper for each metric.
    """
    rng = np.random.RandomState(seed)
    n   = len(labels)

    auc_scores, acc_scores, f1_scores = [], [], []

    for _ in range(n_bootstrap):
        idx    = rng.choice(n, n, replace=True)
        y_true = labels[idx]
        y_prob = probs[idx]
        y_pred = (y_prob >= 0.5).astype(int)

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true)) < 2:
            continue

        auc_scores.append(roc_auc_score(y_true, y_prob))
        acc_scores.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    def ci(scores):
        arr = np.array(scores)
        return {
            "mean":     float(np.mean(arr)),
            "std":      float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "auc": ci(auc_scores),
        "acc": ci(acc_scores),
        "f1":  ci(f1_scores),
    }


def format_ci(d):
    return f"{d['mean']:.4f} [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]"


def plot_forest(results, tag):
    """Forest plot of AUC with 95% CI for all models."""
    names  = []
    means  = []
    lowers = []
    uppers = []

    for model_name, display_name in MODELS:
        if model_name not in results:
            continue
        auc = results[model_name]["auc"]
        names.append(display_name)
        means.append(auc["mean"])
        lowers.append(auc["mean"] - auc["ci_lower"])
        uppers.append(auc["ci_upper"] - auc["mean"])

    n   = len(names)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55 + 1.5)))

    colors = []
    for name in names:
        if "GigaPath" in name:
            colors.append("#2196F3")
        elif "ResNet" in name:
            colors.append("#00BCD4")
        elif "BoVW" in name:
            colors.append("#FFC107")
        else:
            colors.append("#F44336")

    y_pos = np.arange(n)
    ax.barh(y_pos, means, xerr=[lowers, uppers],
            align="center", color=colors, alpha=0.8,
            error_kw={"elinewidth": 2, "capsize": 4, "ecolor": "black"})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Test AUC (95% Bootstrap CI)", fontsize=11)
    ax.set_title(f"Model Comparison — Bootstrap 95% CI ({tag} split)", fontsize=13)
    ax.axvline(x=0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlim([0.4, 1.02])
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (m, lo, hi) in enumerate(zip(means, lowers, uppers)):
        ax.text(min(m + hi + 0.005, 1.01), i,
                f"{m:.4f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.8, label="GigaPath MIL"),
        Patch(facecolor="#00BCD4", alpha=0.8, label="ResNet18-cap MIL"),
        Patch(facecolor="#FFC107", alpha=0.8, label="BoVW"),
        Patch(facecolor="#F44336", alpha=0.8, label="Classical"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"bootstrap_ci_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Forest plot saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",         type=str, default="full")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    print(f"Bootstrap CI (n={args.n_bootstrap}, tag={args.tag})\n")

    all_results = {}
    rows        = []

    for model_name, display_name in MODELS:
        probs, labels = load_test_data(model_name, args.tag)
        if probs is None:
            print(f"  [SKIP] {display_name} — no results found")
            continue

        print(f"  {display_name}...")
        ci = bootstrap_metrics(probs, labels,
                               n_bootstrap=args.n_bootstrap,
                               seed=args.seed)
        all_results[model_name] = ci

        rows.append({
            "model":          display_name,
            "auc_mean":       round(ci["auc"]["mean"],     4),
            "auc_ci_lower":   round(ci["auc"]["ci_lower"], 4),
            "auc_ci_upper":   round(ci["auc"]["ci_upper"], 4),
            "auc_std":        round(ci["auc"]["std"],      4),
            "acc_mean":       round(ci["acc"]["mean"],     4),
            "acc_ci_lower":   round(ci["acc"]["ci_lower"], 4),
            "acc_ci_upper":   round(ci["acc"]["ci_upper"], 4),
            "f1_mean":        round(ci["f1"]["mean"],      4),
            "f1_ci_lower":    round(ci["f1"]["ci_lower"],  4),
            "f1_ci_upper":    round(ci["f1"]["ci_upper"],  4),
        })

    # Print table
    print(f"\n{'Model':<30} {'AUC (95% CI)':>28} {'Acc (95% CI)':>28} {'F1 (95% CI)':>28}")
    print("-" * 118)
    for model_name, display_name in MODELS:
        if model_name not in all_results:
            continue
        ci = all_results[model_name]
        print(f"{display_name:<30} {format_ci(ci['auc']):>28} "
              f"{format_ci(ci['acc']):>28} {format_ci(ci['f1']):>28}")

    # Save CSV
    csv_path = os.path.join(FIGURES_DIR, f"bootstrap_ci_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV saved → {csv_path}")

    # Forest plot
    plot_forest(all_results, args.tag)

    # McNemar test between ABMIL and Gated ABMIL
    print("\n── McNemar Test: ABMIL vs Gated ABMIL ──")
    p1, l1 = load_test_data("abmil",       args.tag)
    p2, l2 = load_test_data("gated_abmil", args.tag)

    if p1 is not None and p2 is not None:
        from statsmodels.stats.contingency_tables import mcnemar
        pred1 = (p1 >= 0.5).astype(int)
        pred2 = (p2 >= 0.5).astype(int)

        # Contingency table: both correct, only m1 correct, only m2 correct, both wrong
        both_correct   = ((pred1 == l1) & (pred2 == l2)).sum()
        only1_correct  = ((pred1 == l1) & (pred2 != l2)).sum()
        only2_correct  = ((pred1 != l1) & (pred2 == l2)).sum()
        both_wrong     = ((pred1 != l1) & (pred2 != l2)).sum()

        table = [[both_correct, only1_correct],
                 [only2_correct, both_wrong]]

        result = mcnemar(table, exact=False, correction=True)
        print(f"  Contingency table:")
        print(f"    Both correct:   {both_correct}")
        print(f"    Only ABMIL:     {only1_correct}")
        print(f"    Only Gated:     {only2_correct}")
        print(f"    Both wrong:     {both_wrong}")
        print(f"  McNemar statistic: {result.statistic:.4f}")
        print(f"  p-value:           {result.pvalue:.4f}")
        if result.pvalue > 0.05:
            print("  → No significant difference (p > 0.05)")
        else:
            print("  → Significant difference (p ≤ 0.05)")

    print("\nDone.")


if __name__ == "__main__":
    main()
