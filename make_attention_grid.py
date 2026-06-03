"""
make_attention_grid.py
----------------------
Creates a clean 3x2 attention heatmap grid:
  Row 1: High concentration   (LUAD | LUSC)
  Row 2: Moderate concentration
  Row 3: Low concentration
"""

import os, sys, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import h5py

sys.path.insert(0, os.path.join(os.environ.get("DSAI543_DATA", os.path.expanduser("~/research_data")), "project"))
from models.model import ABMIL

BASE_DIR  = os.environ.get("DSAI543_DATA", os.path.expanduser("~/research_data"))
GP_H5     = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
CKPT_PATH = os.path.join(BASE_DIR, "project", "results",
                         "checkpoints", "abmil_full", "best.pt")
THUMB_DIR = os.path.join(BASE_DIR, "project", "results", "figures",
                         "attention_heatmaps", "clean_thumbs")
OUT_PATH  = os.path.join(BASE_DIR, "project", "results", "figures",
                         "attention_heatmaps", "attention_concentration_grid_v2.png")

# (fid, label, group)
SLIDES = [
    ("17cc1b48-e2d1-4601-b206-da3a10075589", "LUAD", "High"),
    ("c7d2fc66-baeb-4138-99f3-ddae1100ceda", "LUSC", "High"),
    ("01bea133-7f27-40a6-8d4e-018da17accda", "LUAD", "Moderate"),
    ("dac76377-5786-4bdc-b546-41e9bb196512", "LUSC", "Moderate"),
    ("8974796d-f014-4adc-b508-762cbd62adb5", "LUAD", "Low"),
    ("b5b74bca-6e1f-4964-8b3c-1203ff967bb7", "LUSC", "Low"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(CKPT_PATH, map_location=device)
model  = ABMIL(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Model loaded on {device}")


def make_grid(coords, attn, tile_size=512):
    x, y   = coords[:, 0], coords[:, 1]
    gw     = int((x.max() - x.min()) // tile_size) + 1
    gh     = int((y.max() - y.min()) // tile_size) + 1
    grid   = np.zeros((gh, gw), dtype=np.float32)
    cnt    = np.zeros((gh, gw), dtype=np.int32)
    for i, (cx, cy) in enumerate(zip(x, y)):
        gx = int((cx - x.min()) // tile_size)
        gy = int((cy - y.min()) // tile_size)
        grid[gy, gx] += attn[i]
        cnt[gy, gx]  += 1
    mask = cnt > 0
    grid[mask] /= cnt[mask]
    if grid.max() > grid.min():
        grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid


def blend_heatmap(thumb, grid, alpha=0.5):
    tw, th   = thumb.size
    grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
    grid_arr = np.array(grid_img) / 255.0
    heatmap  = (cm.jet(grid_arr)[:, :, :3] * 255).astype(np.uint8)
    heatmap  = Image.fromarray(heatmap)
    return Image.blend(thumb.convert("RGB"), heatmap, alpha)


def compute_entropy(attn):
    a = attn + 1e-10
    a = a / a.sum()
    return float(-np.sum(a * np.log(a)))


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 16))
row_labels = ["High Concentration", "Moderate Concentration", "Low Concentration"]

with h5py.File(GP_H5, "r") as h:
    for i, (fid, label, group) in enumerate(SLIDES):
        row = i // 2
        col = i % 2

        features = h[fid]["features"][:]
        coords   = h[fid]["coords"][:]

        bag = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.inference_mode():
            logits, attn_t = model(bag)
        attn = attn_t.cpu().numpy()
        prob = torch.softmax(logits, dim=0)[1].item()
        pred = "LUSC" if prob >= 0.5 else "LUAD"
        ent  = compute_entropy(attn)

        grid    = make_grid(coords, attn)
        thumb   = Image.open(os.path.join(THUMB_DIR, f"{fid[:8]}.png"))
        overlay = blend_heatmap(thumb, grid, alpha=0.5)

        ax = axes[row][col]
        ax.imshow(overlay)

        correct = pred == label
        color   = "green" if correct else "red"
        marker  = "✓" if correct else "✗"
        ax.set_title(
            f"True: {label}  |  Pred: {pred} (p={prob:.3f}) {marker}\n"
            f"Entropy: {ent:.2f}  |  Tiles: {len(features)}",
            fontsize=10, color=color
        )
        ax.axis("off")

        # Row label on leftmost column
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=12, fontweight="bold",
                         rotation=90, labelpad=10)

# Column headers
axes[0][0].set_title("LUAD\n" + axes[0][0].get_title(), fontsize=10, color="green")
axes[0][1].set_title("LUSC\n" + axes[0][1].get_title(), fontsize=10, color="green")

# Row labels via text on figure
for r, lbl in enumerate(row_labels):
    fig.text(0.01, 1 - (r + 0.5) / 3, lbl,
             va="center", ha="left", fontsize=11,
             fontweight="bold", rotation=90,
             transform=fig.transFigure)

fig.suptitle("ABMIL Attention Heatmaps — Concentration Analysis\n"
             "Red/Yellow = High Attention  |  Blue = Low Attention",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
