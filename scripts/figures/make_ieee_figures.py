"""
make_ieee_figures.py
--------------------
Regenerates all project figures with IEEE-compliant settings:
  - 300 DPI
  - Font sizes >= 8pt (labels 10-12pt)
  - Color-blind safe palette
  - Viridis colormap for heatmaps (grayscale-safe)
  - Single column: 3.5" wide
  - Double column: 7.16" wide

Usage:
  python -m scripts.figures.make_ieee_figures
"""

import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc
from PIL import Image
import h5py
import torch

sys.path.insert(0, os.path.join(os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data")), "project"))
from models.model import ABMIL, GatedABMIL, MeanPoolMLP

# ─── IEEE style defaults ───────────────────────────────────────────────────────
rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.5,
    "lines.markersize": 5,
})

DPI        = 300
COL1       = 3.5     # single column width inches
COL2       = 7.16    # double column width inches

# ─── Color-blind safe palette ─────────────────────────────────────────────────
COLORS = {
    "abmil":         "#0077BB",   # blue
    "gated_abmil":   "#CC3311",   # red
    "meanpool_mlp":  "#009988",   # teal
    "resnet_mil":    "#EE7733",   # orange
    "bovw":          "#AA4499",   # purple
    "classical_mlp": "#BBBBBB",   # gray
    "pca_svm":       "#44BB99",   # mint
    "svm":           "#DDCC77",   # sand
    "xgboost":       "#882255",   # wine
}

MODEL_LABELS = {
    "abmil":         "ABMIL (GigaPath)",
    "gated_abmil":   "Gated ABMIL (GigaPath)",
    "meanpool_mlp":  "MeanPool MLP (GigaPath)",
    "resnet_mil":    "ResNet18-cap MIL",
    "bovw":          "BoVW + SVM",
    "classical_mlp": "MLP (Classical)",
    "pca_svm":       "PCA + SVM",
    "svm":           "RBF-SVM",
    "xgboost":       "XGBoost",
}

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
RESULTS    = os.path.join(BASE_DIR, "project", "results")
LOGS       = os.path.join(RESULTS, "logs")
FIGS       = os.path.join(RESULTS, "figures")
IEEE_DIR   = os.path.join(FIGS, "ieee")
os.makedirs(IEEE_DIR, exist_ok=True)


def load_results(model, tag="full"):
    path = os.path.join(LOGS, f"{model}_{tag}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_history(model, tag="full"):
    path = os.path.join(LOGS, f"{model}_{tag}", "history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─── Figure 1: ROC Curves ─────────────────────────────────────────────────────
def fig_roc_curves(tag="full"):
    models = ["abmil", "gated_abmil", "meanpool_mlp", "resnet_mil",
              "bovw", "classical_mlp", "svm"]

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.65))
    ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.4, label="Random")

    for model in models:
        res = load_results(model, tag)
        if res is None:
            continue
        t = res.get("test", res)
        if not t.get("probs"):
            continue
        fpr, tpr, _ = roc_curve(t["labels"], t["probs"])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[model], lw=1.5,
                label=f"{MODEL_LABELS[model]} ({auc_val:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    out = os.path.join(IEEE_DIR, f"fig_roc_{tag}.pdf")
    fig.savefig(out, dpi=DPI, format="pdf")
    fig.savefig(out.replace(".pdf",".png"), dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


# ─── Figure 2: Learning Curve ─────────────────────────────────────────────────
def fig_learning_curve(tag="full"):
    lc_path = os.path.join(LOGS, "learning_curve", f"learning_curve_{tag}.json")
    if not os.path.exists(lc_path):
        print("  [SKIP] learning curve JSON not found")
        return

    with open(lc_path) as f:
        d = json.load(f)

    STYLES = {
        "abmil":    {"color": COLORS["abmil"],        "marker": "o", "ls": "-",
                     "label": "ABMIL (GigaPath)"},
        "meanpool": {"color": COLORS["meanpool_mlp"],  "marker": "s", "ls": "-",
                     "label": "MeanPool MLP (GigaPath)"},
        "bovw":     {"color": COLORS["bovw"],          "marker": "^", "ls": "--",
                     "label": "BoVW + SVM (Classical)"},
    }

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.6))

    for model, sizes in d.items():
        style = STYLES.get(model, {})
        ns    = sorted([int(k) for k in sizes.keys() if int(k) <= 300])
        means = [sizes[str(n)]["auc_mean"] for n in ns]
        stds  = [sizes[str(n)]["auc_std"]  for n in ns]

        ax.plot(ns, means, color=style["color"], marker=style["marker"],
                linestyle=style["ls"], lw=1.5, markersize=5,
                label=style["label"])
        ax.fill_between(ns,
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        alpha=0.15, color=style["color"])

    # Annotate ABMIL saturation
    ax.annotate("ABMIL: AUC=0.980 at\n25 patients/class",
                xy=(25, 0.9802), xytext=(120, 0.86),
                arrowprops=dict(arrowstyle="->", color=COLORS["abmil"], lw=1),
                fontsize=8, color=COLORS["abmil"])

    ax.set_xlabel("Training Patients per Class")
    ax.set_ylabel("Test AUC")
    ax.set_title("Learning Curve: Foundation Model vs. Classical Features\n"
                 "(mean ± std, 3 seeds)")
    ax.legend(loc="lower right")
    ax.set_ylim([0.55, 1.02])
    ax.set_xticks([25, 50, 75, 100, 150, 200, 250, 300])
    ax.axhline(y=0.98, color="gray", linestyle=":", lw=0.8, alpha=0.6)

    out = os.path.join(IEEE_DIR, f"fig_learning_curve_{tag}.pdf")
    fig.savefig(out, dpi=DPI, format="pdf")
    fig.savefig(out.replace(".pdf",".png"), dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


# ─── Figure 3: Bootstrap CI Forest Plot ───────────────────────────────────────
def fig_bootstrap_ci(tag="full"):
    csv_path = os.path.join(FIGS, f"bootstrap_ci_{tag}.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] bootstrap CI CSV not found")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    names  = [r["model"] for r in rows]
    means  = [float(r["auc_mean"])     for r in rows]
    lowers = [float(r["auc_mean"]) - float(r["auc_ci_lower"]) for r in rows]
    uppers = [float(r["auc_ci_upper"]) - float(r["auc_mean"]) for r in rows]

    fig, ax = plt.subplots(figsize=(COL2, max(3.5, len(names) * 0.45)))
    y_pos   = np.arange(len(names))

    bar_colors = []
    for name in names:
        if "GigaPath" in name or "ResNet" in name:
            bar_colors.append(COLORS["abmil"])
        elif "BoVW" in name:
            bar_colors.append(COLORS["bovw"])
        else:
            bar_colors.append(COLORS["svm"])

    ax.barh(y_pos, means, xerr=[lowers, uppers],
            color=bar_colors, alpha=0.75,
            error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Test AUC (95% Bootstrap CI)")
    ax.set_title("Model Comparison with 95% Bootstrap Confidence Intervals")
    ax.axvline(x=0.5, color="gray", linestyle="--", lw=0.8)
    ax.set_xlim([0.4, 1.05])

    # Value labels
    for i, (m, u) in enumerate(zip(means, uppers)):
        ax.text(min(m + u + 0.005, 1.03), i, f"{m:.3f}",
                va="center", fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["abmil"], alpha=0.75, label="GigaPath-based"),
        Patch(facecolor=COLORS["bovw"],  alpha=0.75, label="BoVW"),
        Patch(facecolor=COLORS["svm"],   alpha=0.75, label="Classical"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    out = os.path.join(IEEE_DIR, f"fig_bootstrap_ci_{tag}.pdf")
    fig.savefig(out, dpi=DPI, format="pdf")
    fig.savefig(out.replace(".pdf",".png"), dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


# ─── Figure 4: Training Curves ────────────────────────────────────────────────
def fig_training_curves(tag="full"):
    gp_models = ["abmil", "gated_abmil", "meanpool_mlp"]
    valid     = [(m, load_history(m, tag)) for m in gp_models
                 if load_history(m, tag)]
    if not valid:
        print("  [SKIP] no history files found")
        return

    n   = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(COL2, 2.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (model, hist) in zip(axes, valid):
        epochs   = [h["epoch"]      for h in hist]
        tr_auc   = [h["train_auc"]  for h in hist]
        val_auc  = [h["val_auc"]    for h in hist]

        ax.plot(epochs, tr_auc,  color=COLORS[model], lw=1.5, label="Train")
        ax.plot(epochs, val_auc, color=COLORS[model], lw=1.5,
                linestyle="--", label="Val")
        ax.set_title(MODEL_LABELS[model], fontsize=8)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("AUC", fontsize=8)
        ax.set_ylim([0.5, 1.02])
        ax.legend(fontsize=7)

    fig.suptitle("Training Curves — GigaPath MIL Models", fontsize=9)
    fig.tight_layout()

    out = os.path.join(IEEE_DIR, f"fig_training_curves_{tag}.pdf")
    fig.savefig(out, dpi=DPI, format="pdf")
    fig.savefig(out.replace(".pdf",".png"), dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


# ─── Figure 5: Confusion Matrices ─────────────────────────────────────────────
def fig_confusion_matrices(tag="full"):
    models = ["abmil", "gated_abmil", "meanpool_mlp", "bovw", "svm"]
    valid  = []
    for m in models:
        res = load_results(m, tag)
        if res is None:
            continue
        t  = res.get("test", res)
        cm = t.get("cm")
        if cm:
            valid.append((m, np.array(cm)))

    if not valid:
        print("  [SKIP] no confusion matrices found")
        return

    n   = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(COL2, 2.2))
    if n == 1:
        axes = [axes]

    for ax, (model, cm) in zip(axes, valid):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["LUAD","LUSC"], fontsize=7)
        ax.set_yticklabels(["LUAD","LUSC"], fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.set_title(MODEL_LABELS[model], fontsize=7)
        thresh = cm.max() * 0.6
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if cm[i,j] > thresh else "#1a1a1a")

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=9)
    fig.tight_layout()

    out = os.path.join(IEEE_DIR, f"fig_confusion_{tag}.pdf")
    fig.savefig(out, dpi=DPI, format="pdf")
    fig.savefig(out.replace(".pdf",".png"), dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


# ─── Figure 6: Attention Heatmaps (viridis) ───────────────────────────────────
def fig_attention_heatmaps(tag="full"):
    """Regenerate attention concentration grid with viridis colormap."""
    THUMB_DIR = os.path.join(FIGS, "attention_heatmaps", "clean_thumbs")
    GP_H5     = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
    CKPT      = os.path.join(RESULTS, "checkpoints", "abmil_full", "best.pt")

    if not os.path.exists(THUMB_DIR) or not os.path.exists(CKPT):
        print("  [SKIP] attention heatmap assets not found")
        return

    SLIDES = [
        ("17cc1b48-e2d1-4601-b206-da3a10075589", "LUAD", "High"),
        ("c7d2fc66-baeb-4138-99f3-ddae1100ceda", "LUSC", "High"),
        ("01bea133-7f27-40a6-8d4e-018da17accda", "LUAD", "Moderate"),
        ("dac76377-5786-4bdc-b546-41e9bb196512", "LUSC", "Moderate"),
        ("8974796d-f014-4adc-b508-762cbd62adb5", "LUAD", "Low"),
        ("b5b74bca-6e1f-4964-8b3c-1203ff967bb7", "LUSC", "Low"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CKPT, map_location=device)
    model  = ABMIL(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def make_grid(coords, attn, tile_size=512):
        x, y = coords[:,0], coords[:,1]
        gw = int((x.max()-x.min())//tile_size)+1
        gh = int((y.max()-y.min())//tile_size)+1
        grid = np.zeros((gh,gw), dtype=np.float32)
        cnt  = np.zeros((gh,gw), dtype=np.int32)
        for i,(cx,cy) in enumerate(zip(x,y)):
            gx = int((cx-x.min())//tile_size)
            gy = int((cy-y.min())//tile_size)
            grid[gy,gx] += attn[i]; cnt[gy,gx] += 1
        m = cnt>0; grid[m] /= cnt[m]
        if grid.max()>grid.min():
            grid = (grid-grid.min())/(grid.max()-grid.min())
        return grid

    def blend(thumb, grid, alpha=0.5):
        tw,th = thumb.size
        g = Image.fromarray((grid*255).astype(np.uint8)).resize((tw,th), Image.LANCZOS)
        h = (cm.viridis(np.array(g)/255.0)[:,:,:3]*255).astype(np.uint8)
        return Image.blend(thumb.convert("RGB"), Image.fromarray(h), alpha)

    fig, axes = plt.subplots(3, 2, figsize=(COL2, COL2 * 1.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.02, hspace=0.25, wspace=0.08)

    with h5py.File(GP_H5, "r") as h:
        for i, (fid, label, group) in enumerate(SLIDES):
            row, col = i//2, i%2
            feats = h[fid]["features"][:]
            coords = h[fid]["coords"][:]
            bag = torch.tensor(feats, dtype=torch.float32).to(device)
            with torch.inference_mode():
                logits, attn_t = model(bag)
            attn = attn_t.cpu().numpy()
            prob = torch.softmax(logits,dim=0)[1].item()
            pred = "LUSC" if prob>=0.5 else "LUAD"
            ent  = float(-np.sum((attn+1e-10)/((attn+1e-10).sum()) *
                                  np.log((attn+1e-10)/((attn+1e-10).sum()))))

            grid    = make_grid(coords, attn)
            thumb   = Image.open(os.path.join(THUMB_DIR, f"{fid[:8]}.png"))
            overlay = blend(thumb, grid)

            ax = axes[row][col]
            ax.imshow(overlay)
            correct = pred==label
            ax.set_title(f"True: {label}  Pred: {pred} (p={prob:.2f})\n"
                        f"H={ent:.2f}  N={len(feats)}",
                        fontsize=7, color="green" if correct else "red")
            ax.axis("off")

    # Row labels as figure text (set_ylabel doesn't work on image axes)
    row_labels = ["High\nConcentration", "Moderate\nConcentration", "Low\nConcentration"]
    row_centers = [0.78, 0.47, 0.16]   # vertical centers for each row
    for lbl, yc in zip(row_labels, row_centers):
        fig.text(0.02, yc, lbl,
                 va="center", ha="center", fontsize=8,
                 fontweight="bold", rotation=90)

    # Column headers via fig.text to avoid overlap with suptitle
    fig.text(0.38, 0.96, "LUAD", ha="center", fontsize=9, fontweight="bold")
    fig.text(0.75, 0.96, "LUSC", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("ABMIL Attention Heatmaps — Concentration Analysis",
                 fontsize=9, y=1.00)

    out = os.path.join(IEEE_DIR, f"fig_attention_heatmaps_{tag}.pdf")
    fig.savefig(out, dpi=DPI, format="pdf")
    fig.savefig(out.replace(".pdf",".png"), dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    tag = "full"
    print(f"Generating IEEE-compliant figures (tag={tag}, DPI={DPI})...\n")
    print(f"Output: {IEEE_DIR}\n")

    print("1. ROC curves...")
    fig_roc_curves(tag)

    print("2. Learning curve...")
    fig_learning_curve(tag)

    print("3. Bootstrap CI forest plot...")
    fig_bootstrap_ci(tag)

    print("4. Training curves...")
    fig_training_curves(tag)

    print("5. Confusion matrices...")
    fig_confusion_matrices(tag)

    print("6. Attention heatmaps (viridis)...")
    fig_attention_heatmaps(tag)

    print(f"\nAll IEEE figures saved to: {IEEE_DIR}/")
    print("Both PDF and PNG versions generated.")
    print("Use PDF versions in LaTeX for best quality.")


if __name__ == "__main__":
    main()
