"""
attention_heatmap.py
--------------------
Generates attention heatmaps for ABMIL model on test slides.
Maps per-tile attention weights back to spatial coordinates
and overlays as a heatmap on the slide thumbnail.

Outputs:
  results/figures/attention_heatmaps/
    <fid>_<label>_<pred>_heatmap.png   — heatmap overlay
    attention_grid.png                  — multi-slide grid figure

Usage:
  python3 attention_heatmap.py --tag full --n-slides 6
  python3 attention_heatmap.py --tag full --fids <fid1> <fid2>
"""

import argparse
import csv
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from models.model import ABMIL

BASE_DIR    = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
GP_H5       = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
CKPT_PATH   = os.path.join(BASE_DIR, "project", "results",
                            "checkpoints", "abmil_full", "best.pt")
SPLITS_DIR  = os.path.join(BASE_DIR, "project", "splits")
OUT_DIR     = os.path.join(BASE_DIR, "project", "results",
                           "figures", "attention_heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

TILE_SIZE   = 512    # extraction tile size at level 0


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ABMIL(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint (val AUC={ckpt['val_auc']:.4f}, epoch={ckpt['epoch']})")
    return model


def get_attention(model, features, device):
    """Run ABMIL forward pass, return attention weights (N_tiles,)."""
    bag    = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.inference_mode():
        logits, attn = model(bag)
    prob = torch.softmax(logits, dim=0)[1].item()
    pred = "LUSC" if prob >= 0.5 else "LUAD"
    return attn.cpu().numpy(), prob, pred


def make_heatmap(coords, attn_weights, tile_size=TILE_SIZE):
    """
    Build a 2D attention heatmap from tile coordinates and weights.
    Returns a PIL Image of the heatmap.
    """
    if len(coords) == 0:
        return None

    # Normalize coordinates to grid
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Grid dimensions
    grid_w = int((x_max - x_min) // tile_size) + 1
    grid_h = int((y_max - y_min) // tile_size) + 1

    grid     = np.zeros((grid_h, grid_w), dtype=np.float32)
    grid_cnt = np.zeros((grid_h, grid_w), dtype=np.int32)

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        gx = int((x - x_min) // tile_size)
        gy = int((y - y_min) // tile_size)
        grid[gy, gx]     += attn_weights[i]
        grid_cnt[gy, gx] += 1

    # Average where multiple tiles overlap
    mask = grid_cnt > 0
    grid[mask] /= grid_cnt[mask]

    # Normalize to [0,1]
    if grid.max() > grid.min():
        grid = (grid - grid.min()) / (grid.max() - grid.min())

    return grid


def render_heatmap_overlay(thumbnail, grid, alpha=0.55, cmap="jet"):
    """
    Overlay attention heatmap on thumbnail.
    Returns PIL Image.
    """
    thumb_w, thumb_h = thumbnail.size

    # Resize grid to thumbnail size
    grid_img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    grid_img = grid_img.resize((thumb_w, thumb_h), Image.LANCZOS)
    grid_arr = np.array(grid_img) / 255.0

    # Apply colormap
    colormap  = cm.get_cmap(cmap)
    heatmap   = (colormap(grid_arr)[:, :, :3] * 255).astype(np.uint8)
    heatmap   = Image.fromarray(heatmap, mode="RGB")

    # Blend
    thumb_rgb = thumbnail.convert("RGB")
    blended   = Image.blend(thumb_rgb, heatmap, alpha=alpha)

    return blended


def get_slide_thumbnail(fid, size=(600, 600)):
    """Download slide thumbnail from GDC."""
    import requests
    url = f"https://api.gdc.cancer.gov/data/{fid}"
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            # Write to temp file and open with openslide
            import tempfile, openslide
            with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as f:
                tmp_path = f.name
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            slide = openslide.OpenSlide(tmp_path)
            thumb = slide.get_thumbnail(size)
            slide.close()
            os.unlink(tmp_path)
            return thumb
    except Exception as e:
        print(f"  [WARN] Could not get thumbnail: {e}")
        return None


def process_slide(fid, label, model, device, download_thumb=True):
    """Process one slide — get attention weights and generate heatmap."""
    import h5py

    with h5py.File(GP_H5, "r") as h:
        if fid not in h:
            print(f"  [SKIP] {fid[:8]} not in H5")
            return None
        features = h[fid]["features"][:]
        coords   = h[fid]["coords"][:]

    print(f"  {fid[:8]}: {len(features)} tiles, label={label}")

    attn_weights, prob, pred = get_attention(model, features, device)

    print(f"  Pred: {pred} (p={prob:.4f}), "
          f"attn range: [{attn_weights.min():.6f}, {attn_weights.max():.6f}]")

    grid = make_heatmap(coords, attn_weights)
    if grid is None:
        return None

    # Get thumbnail
    if download_thumb:
        print(f"  Downloading thumbnail...")
        thumb = get_slide_thumbnail(fid)
    else:
        thumb = None

    if thumb is None:
        # Create blank thumbnail sized by grid
        thumb = Image.new("RGB", (600, 600), (240, 240, 240))

    overlay = render_heatmap_overlay(thumb, grid)

    return {
        "fid":          fid,
        "label":        label,
        "pred":         pred,
        "prob":         prob,
        "attn_weights": attn_weights,
        "grid":         grid,
        "thumb":        thumb,
        "overlay":      overlay,
        "n_tiles":      len(features),
        "correct":      pred == label,
    }


def save_single_heatmap(result, show_colorbar=True):
    """Save individual heatmap figure with side-by-side original + overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original thumbnail
    axes[0].imshow(result["thumb"])
    axes[0].set_title(f"Original\nTrue: {result['label']}", fontsize=11)
    axes[0].axis("off")

    # Attention heatmap only
    im = axes[1].imshow(result["grid"], cmap="jet", interpolation="bilinear")
    axes[1].set_title(f"Attention Map\n{result['n_tiles']} tiles", fontsize=11)
    axes[1].axis("off")
    if show_colorbar:
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(result["overlay"])
    correct_str = "✓ Correct" if result["correct"] else "✗ Wrong"
    axes[2].set_title(
        f"Overlay\nPred: {result['pred']} (p={result['prob']:.3f}) {correct_str}",
        fontsize=11,
        color="green" if result["correct"] else "red"
    )
    axes[2].axis("off")

    fig.suptitle(f"ABMIL Attention — {result['fid'][:8]}", fontsize=13)
    fig.tight_layout()

    fname  = f"{result['fid'][:8]}_{result['label']}_pred{result['pred']}.png"
    out    = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def save_attention_grid(results, n_cols=3):
    """Save multi-slide grid of overlays."""
    n      = len(results)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for i, result in enumerate(results):
        ax = axes_flat[i]
        ax.imshow(result["overlay"])
        correct_str = "✓" if result["correct"] else "✗"
        color       = "green" if result["correct"] else "red"
        ax.set_title(
            f"{result['fid'][:8]}\n"
            f"True: {result['label']}  Pred: {result['pred']} "
            f"(p={result['prob']:.3f}) {correct_str}",
            fontsize=9, color=color
        )
        ax.axis("off")

    # Hide empty axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("ABMIL Attention Heatmaps — Test Set", fontsize=14)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "attention_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrid saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",           type=str, default="full")
    parser.add_argument("--n-slides",      type=int, default=6,
                        help="Number of slides to visualize")
    parser.add_argument("--fids",          nargs="+", default=None,
                        help="Specific file IDs to visualize")
    parser.add_argument("--no-download",   action="store_true",
                        help="Skip thumbnail download (use blank background)")
    parser.add_argument("--include-wrong", action="store_true",
                        help="Include misclassified slides")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(CKPT_PATH, device)

    # Load test split
    suffix   = f"_{args.tag}" if args.tag else ""
    test_csv = os.path.join(SPLITS_DIR, f"test{suffix}.csv")
    with open(test_csv) as f:
        all_rows = list(csv.DictReader(f))

    # Select slides
    if args.fids:
        rows = [r for r in all_rows if r["file_id"] in args.fids]
    else:
        # Pick balanced set: LUAD and LUSC, mix correct/incorrect
        import h5py
        with h5py.File(GP_H5, "r") as h:
            available = set(h.keys())

        rows = [r for r in all_rows if r["file_id"] in available]

        # Sample balanced
        luad_rows = [r for r in rows if r["label"] == "LUAD"]
        lusc_rows = [r for r in rows if r["label"] == "LUSC"]

        import random
        random.seed(42)
        n_each   = args.n_slides // 2
        selected = (random.sample(luad_rows, min(n_each, len(luad_rows))) +
                    random.sample(lusc_rows, min(n_each, len(lusc_rows))))
        rows     = selected

    print(f"\nProcessing {len(rows)} slides...")
    results = []
    for row in rows:
        print(f"\nSlide {len(results)+1}/{len(rows)}:")
        result = process_slide(
            row["file_id"], row["label"], model, device,
            download_thumb=not args.no_download
        )
        if result:
            save_single_heatmap(result)
            results.append(result)

    if results:
        save_attention_grid(results)

    print(f"\nDone. {len(results)} heatmaps saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
