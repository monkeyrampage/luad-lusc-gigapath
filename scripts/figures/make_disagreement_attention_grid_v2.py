"""
make_disagreement_attention_grid_v2.py
---------------------------------------
IEEE-compliant 7x3 disagreement attention grid.
Fixes: viridis colormap, no row gaps, larger fonts, 300 DPI.
Layout: Original | ABMIL Attention | Gated ABMIL Attention
"""

import os, sys, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from PIL import Image
import h5py

rcParams.update({
    "font.family":    "serif",
    "font.size":      8,
    "axes.titlesize": 8,
    "savefig.dpi":    300,
    "figure.dpi":     300,
})

sys.path.insert(0, os.path.join(os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data")), "project"))
from models.model import ABMIL, GatedABMIL

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
GP_H5      = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
THUMB_DIR  = os.path.join(BASE_DIR, "project", "results", "figures",
                          "disagreement_thumbnails", "clean")
OUT_PATH   = os.path.join(BASE_DIR, "project", "results", "figures",
                          "disagreement_attention_grid_v2.png")
CKPT_ABMIL = os.path.join(BASE_DIR, "project", "results",
                           "checkpoints", "abmil_full", "best.pt")
CKPT_GATED = os.path.join(BASE_DIR, "project", "results",
                           "checkpoints", "gated_abmil_full", "best.pt")

SLIDES = [
    ("1365c2c0-7231-487c-9ede-5036116ff6fa", "LUAD", "ABMIL"),
    ("1ec9435b-6056-4d34-802d-a4d20208f60e", "LUAD", "ABMIL"),
    ("fbcae59c-816d-44c0-899c-067562d575f0", "LUAD", "ABMIL"),
    ("331bb33f-0dd0-42f1-b547-1f42a2af15ff", "LUAD", "Gated"),
    ("3dfc4d6e-c337-494c-8e59-ffadd137dbeb", "LUAD", "Gated"),
    ("933463fd-242e-4884-b149-07af70445139", "LUAD", "Gated"),
    ("eed858e8-ea64-40d8-a080-9551941ae1b5", "LUSC", "Gated"),
]

ABMIL_COLOR = "#2196F3"
GATED_COLOR = "#9C27B0"
TILE_SIZE   = 512


def load_model(cls, ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = cls(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def get_attention(model, features, device):
    bag = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.inference_mode():
        logits, attn = model(bag)
    prob = torch.softmax(logits, dim=0)[1].item()
    pred = "LUSC" if prob >= 0.5 else "LUAD"
    return attn.cpu().numpy(), prob, pred


def make_attn_grid(coords, attn):
    x, y = coords[:, 0], coords[:, 1]
    gw   = int((x.max() - x.min()) // TILE_SIZE) + 1
    gh   = int((y.max() - y.min()) // TILE_SIZE) + 1
    grid = np.zeros((gh, gw), dtype=np.float32)
    cnt  = np.zeros((gh, gw), dtype=np.int32)
    for i, (cx, cy) in enumerate(zip(x, y)):
        gx = int((cx - x.min()) // TILE_SIZE)
        gy = int((cy - y.min()) // TILE_SIZE)
        grid[gy, gx] += attn[i]
        cnt[gy, gx]  += 1
    mask = cnt > 0
    grid[mask] /= cnt[mask]
    if grid.max() > grid.min():
        grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid


def blend(thumb, grid, alpha=0.5):
    tw, th   = thumb.size
    grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
    grid_arr = np.array(grid_img) / 255.0
    heatmap  = (cm.viridis(grid_arr)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(thumb.convert("RGB"), Image.fromarray(heatmap), alpha)


def main():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abmil_model = load_model(ABMIL,      CKPT_ABMIL, device)
    gated_model = load_model(GatedABMIL, CKPT_GATED, device)
    print(f"Models loaded on {device}")

    n_rows = len(SLIDES)
    fig, axes = plt.subplots(n_rows, 3,
                              figsize=(7.16, n_rows * 1.9),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.05})

    # Column headers on row 0
    col_titles = ["Original", "ABMIL Attention", "Gated ABMIL Attention"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    with h5py.File(GP_H5, "r") as h:
        for row, (fid, true_label, winner) in enumerate(SLIDES):
            print(f"  Processing {fid[:8]} ({true_label}, winner={winner})")

            features = h[fid]["features"][:]
            coords   = h[fid]["coords"][:]

            a_attn, a_prob, a_pred = get_attention(abmil_model, features, device)
            g_attn, g_prob, g_pred = get_attention(gated_model, features, device)

            a_grid    = make_attn_grid(coords, a_attn)
            g_grid    = make_attn_grid(coords, g_attn)
            thumb     = Image.open(os.path.join(THUMB_DIR, f"{fid[:8]}.png"))
            a_overlay = blend(thumb, a_grid)
            g_overlay = blend(thumb, g_grid)

            border_color = ABMIL_COLOR if winner == "ABMIL" else GATED_COLOR

            # Col 0: original
            axes[row][0].imshow(thumb)
            axes[row][0].set_ylabel(
                f"True: {true_label}  [{winner} wins]",
                fontsize=7, fontweight="bold", color=border_color,
                labelpad=3
            )
            axes[row][0].axis("off")

            # Col 1: ABMIL
            axes[row][1].imshow(a_overlay)
            correct = a_pred == true_label
            axes[row][1].set_title(
                f"{'OK' if correct else 'X'} {a_pred} (p={a_prob:.3f})",
                fontsize=7, color="green" if correct else "red", pad=2
            )
            axes[row][1].axis("off")

            # Col 2: Gated
            axes[row][2].imshow(g_overlay)
            correct = g_pred == true_label
            axes[row][2].set_title(
                f"{'OK' if correct else 'X'} {g_pred} (p={g_prob:.3f})",
                fontsize=7, color="green" if correct else "red", pad=2
            )
            axes[row][2].axis("off")

    fig.suptitle(
        "Disagreement Cases: ABMIL vs. Gated ABMIL Attention Maps\n"
        "Blue label = ABMIL correct  |  Purple label = Gated correct  |  "
        "Viridis: Dark = Low, Bright = High Attention",
        fontsize=8, y=1.005
    )

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
