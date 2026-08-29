"""
eval_checkpoint.py
------------------
Recompute held-out TCGA test metrics directly from a trained checkpoint.
Trains nothing and writes nothing.

Usage:
  python -m scripts.evaluation.eval_checkpoint --model abmil --tag full
"""

import argparse
import os

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from scripts.training.train import run_epoch, get_paths, RESULTS_DIR
from data.dataset import get_gigapath_loaders
from models.model import get_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="abmil")
    ap.add_argument("--tag",   default="full")
    a = ap.parse_args()

    device = torch.device("cpu")
    run_name = f"{a.model}_{a.tag}"
    ckpt_path = os.path.join(RESULTS_DIR, "checkpoints", run_name, "best.pt")

    ckpt = torch.load(ckpt_path, map_location=device)
    cargs = ckpt["args"]                       # the exact args this checkpoint was trained with
    print(f"Loaded {ckpt_path}")
    print(f"  stored val_auc={ckpt.get('val_auc'):.4f}  epoch={ckpt.get('epoch')}")
    print(f"  hidden={cargs['hidden_dim']} dropout={cargs['dropout']} max_tiles={cargs['max_tiles']}")

    paths = get_paths(a.tag)
    # same call train.py makes; test_loader is what train.py scores on
    _, _, test_loader = get_gigapath_loaders(
        paths["train"], paths["val"], paths["test"],
        batch_size=1, num_workers=0, max_tiles=cargs["max_tiles"],
    )

    model = get_model(a.model, input_dim=1536, hidden_dim=cargs["hidden_dim"],
                      n_classes=2, dropout=cargs["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # the repo's own test forward pass — identical to train.py
    loss, auc, acc, probs, labels = run_epoch(model, test_loader, None, device, train=False)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    print(f"\n=== {a.model} TCGA test (recomputed from checkpoint) ===")
    print(f"  AUC: {auc:.4f}")
    print(f"  Acc: {acc:.4f}")
    print(f"  F1:  {f1_score(labels, preds, zero_division=0):.4f}")
    print(f"  CM:  {confusion_matrix(labels, preds).tolist()}")


if __name__ == "__main__":
    main()
