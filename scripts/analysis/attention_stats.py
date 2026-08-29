"""
attention_stats.py
------------------
Computes attention concentration metrics for all test slides.
Metrics:
  - entropy        : low = concentrated, high = diffuse
  - gini           : high = concentrated (few tiles dominate)
  - top10_pct      : fraction of attention in top 10% of tiles
  - max_attn       : single highest attention weight

Saves ranked CSV and picks:
  - 2 high concentration  (1 LUAD, 1 LUSC) — model is focused
  - 2 moderate            (1 LUAD, 1 LUSC)
  - 2 low concentration   (1 LUAD, 1 LUSC) — diffuse attention

Usage:
  python -m scripts.analysis.attention_stats --tag full
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.model import ABMIL

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
GP_H5      = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
CKPT_PATH  = os.path.join(BASE_DIR, "project", "results",
                           "checkpoints", "abmil_full", "best.pt")
SPLITS_DIR = os.path.join(BASE_DIR, "project", "splits")
OUT_DIR    = os.path.join(BASE_DIR, "project", "results", "figures", "attention_heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = ABMIL(input_dim=1536, hidden_dim=512, n_classes=2).to(device)
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


def entropy(attn):
    """Shannon entropy — low = concentrated."""
    a = attn + 1e-10
    a = a / a.sum()
    return float(-np.sum(a * np.log(a)))


def gini(attn):
    """Gini coefficient — high = concentrated."""
    a = np.sort(attn)
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) / (n * np.sum(a))) - (n + 1) / n)


def top_k_pct(attn, k=0.10):
    """Fraction of total attention in top k% of tiles."""
    n_top = max(1, int(len(attn) * k))
    top   = np.sort(attn)[-n_top:].sum()
    return float(top / attn.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="full")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(CKPT_PATH, device)

    suffix   = f"_{args.tag}" if args.tag else ""
    test_csv = os.path.join(SPLITS_DIR, f"test{suffix}.csv")
    with open(test_csv) as f:
        all_rows = list(csv.DictReader(f))

    import h5py
    results = []

    print(f"Processing {len(all_rows)} slides...")
    with h5py.File(GP_H5, "r") as h:
        for i, row in enumerate(all_rows):
            fid   = row["file_id"]
            label = row["label"]
            if fid not in h:
                continue

            features = h[fid]["features"][:]
            attn, prob, pred = get_attention(model, features, device)

            results.append({
                "file_id":    fid,
                "patient_id": row["patient_id"],
                "label":      label,
                "pred":       pred,
                "correct":    pred == label,
                "prob":       round(prob, 4),
                "n_tiles":    len(features),
                "entropy":    round(entropy(attn), 4),
                "gini":       round(gini(attn), 4),
                "top10_pct":  round(top_k_pct(attn, 0.10), 4),
                "max_attn":   round(float(attn.max()), 6),
            })

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(all_rows)}")

    # Sort by entropy (ascending = most concentrated first)
    results.sort(key=lambda x: x["entropy"])

    # Save full ranked CSV
    csv_path = os.path.join(OUT_DIR, f"attention_stats_{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nFull stats saved: {csv_path}")

    # Print summary stats
    entropies = [r["entropy"] for r in results]
    print(f"\nEntropy stats across {len(results)} slides:")
    print(f"  Min:    {min(entropies):.4f}  (most concentrated)")
    print(f"  Median: {np.median(entropies):.4f}")
    print(f"  Max:    {max(entropies):.4f}  (most diffuse)")

    # Pick 6 representative slides: 2 high, 2 moderate, 2 low concentration
    # High = low entropy (top), Moderate = middle, Low = high entropy (bottom)
    def pick_one(candidates, label):
        return next((r for r in candidates if r["label"] == label and r["correct"]), None)

    n = len(results)
    high_pool = results[:n//4]                          # bottom 25% entropy
    mid_pool  = results[n//3: 2*n//3]                  # middle third
    low_pool  = results[3*n//4:]                        # top 25% entropy

    selected = {}
    for level, pool in [("high", high_pool),
                        ("moderate", mid_pool),
                        ("low", low_pool)]:
        for label in ["LUAD", "LUSC"]:
            pick = pick_one(pool, label)
            if pick:
                selected[f"{level}_{label}"] = pick

    print(f"\n{'='*70}")
    print(f"Selected slides for heatmap visualization:")
    print(f"{'='*70}")
    print(f"{'Key':<20} {'FID':>10} {'Label':>6} {'Pred':>6} "
          f"{'Correct':>8} {'Prob':>6} {'Entropy':>8} {'Gini':>6} "
          f"{'Top10%':>7} {'Tiles':>6}")
    print("-" * 85)

    fids_for_heatmap = []
    for key, r in selected.items():
        print(f"{key:<20} {r['file_id'][:8]:>10} {r['label']:>6} "
              f"{r['pred']:>6} {'✓' if r['correct'] else '✗':>8} "
              f"{r['prob']:>6.3f} {r['entropy']:>8.4f} "
              f"{r['gini']:>6.4f} {r['top10_pct']:>7.4f} "
              f"{r['n_tiles']:>6}")
        fids_for_heatmap.append(r["file_id"])

    # Save selection
    sel_path = os.path.join(OUT_DIR, f"selected_slides_{args.tag}.json")
    with open(sel_path, "w") as f:
        json.dump({"selected": selected, "fids": fids_for_heatmap}, f, indent=2)
    print(f"\nSelection saved: {sel_path}")

    print(f"\nRun heatmaps with:")
    print(f"python -m scripts.analysis.attention_heatmap --tag {args.tag} --fids " +
          " ".join(fids_for_heatmap))


if __name__ == "__main__":
    main()
