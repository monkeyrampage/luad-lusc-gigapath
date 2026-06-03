"""
train.py
--------
Training loop for LUAD/LUSC classification.
Supports all three models: abmil, gated_abmil, meanpool_mlp.

Usage:
  # Mini experiment with ABMIL
  python3 train.py --model abmil --tag mini

  # Gated ABMIL
  python3 train.py --model gated_abmil --tag mini

  # Mean pool baseline
  python3 train.py --model meanpool_mlp --tag mini

  # Full dataset
  python3 train.py --model abmil --tag full --train-csv splits/train.csv ...
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from data.dataset import get_gigapath_loaders, collate_bags
from models.model import get_model, count_parameters

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.environ.get("DSAI543_DATA", os.path.expanduser("~/research_data")), "project")
SPLITS_DIR  = os.path.join(BASE_DIR, "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def get_paths(tag):
    suffix = f"_{tag}" if tag else ""
    return {
        "train": os.path.join(SPLITS_DIR, f"train{suffix}.csv"),
        "val":   os.path.join(SPLITS_DIR, f"val{suffix}.csv"),
        "test":  os.path.join(SPLITS_DIR, f"test{suffix}.csv"),
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    all_labels, all_probs, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss()

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for bags, labels in loader:
            labels = labels.to(device)
            batch_loss = 0.0

            for bag, label in zip(bags, labels):
                bag   = bag.to(device)
                logits, _ = model(bag)
                loss  = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                batch_loss += loss

                prob = torch.softmax(logits, dim=0)[1].item()
                all_probs.append(prob)
                all_labels.append(label.item())

            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss.item()

    avg_loss = total_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs)
    preds    = [1 if p >= 0.5 else 0 for p in all_probs]
    acc      = accuracy_score(all_labels, preds)

    return avg_loss, auc, acc, all_probs, all_labels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dirs
    run_name = f"{args.model}_{args.tag}" if args.tag else args.model
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints", run_name)
    log_dir  = os.path.join(RESULTS_DIR, "logs", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Splits
    paths = get_paths(args.tag)
    print(f"\nSplits: {paths['train']}")

    # Dataloaders
    train_loader, val_loader, test_loader = get_gigapath_loaders(
        paths["train"], paths["val"], paths["test"],
        batch_size=1, num_workers=args.num_workers,
        max_tiles=args.max_tiles,
    )

    # Model
    model = get_model(args.model, input_dim=1536, hidden_dim=args.hidden_dim,
                      n_classes=2, dropout=args.dropout).to(device)
    print(f"\nModel: {args.model} | params={count_parameters(model):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_val_auc  = 0.0
    patience_left = args.patience
    history       = []

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience}, lr={args.lr})...\n")
    print(f"{'Ep':>4} {'T-Loss':>8} {'T-AUC':>7} {'T-Acc':>7} "
          f"{'V-Loss':>8} {'V-AUC':>7} {'V-Acc':>7} {'Time':>7}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_auc, tr_acc, _, _ = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_auc, va_acc, _, _ = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch:>4} {tr_loss:>8.4f} {tr_auc:>7.4f} {tr_acc:>7.4f} "
              f"{va_loss:>8.4f} {va_auc:>7.4f} {va_acc:>7.4f} {elapsed:>6.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_auc": tr_auc, "train_acc": tr_acc,
            "val_loss":   va_loss, "val_auc":   va_auc, "val_acc":   va_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save best
        if va_auc > best_val_auc:
            best_val_auc  = va_auc
            patience_left = args.patience
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "epoch":      epoch,
                "model":      args.model,
                "state_dict": model.state_dict(),
                "val_auc":    va_auc,
                "val_acc":    va_acc,
                "args":       vars(args),
            }, ckpt_path)
            print(f"     ↑ Best val AUC={va_auc:.4f} — saved checkpoint")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # Save training history
    hist_path = os.path.join(log_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    print(f"\nLoading best checkpoint (val AUC={best_val_auc:.4f})...")
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    te_loss, te_auc, te_acc, te_probs, te_labels = run_epoch(model, test_loader, None, device, train=False)
    print(f"\n{'='*40}")
    print(f"Test AUC:  {te_auc:.4f}")
    print(f"Test Acc:  {te_acc:.4f}")
    print(f"{'='*40}")

    # Save test results — nested format consistent with train_classical.py
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    te_preds = [1 if p >= 0.5 else 0 for p in te_probs]
    results = {
        "model":        args.model,
        "tag":          args.tag,
        "epochs_run":   len(history),
        "args":         vars(args),
        "val": {
            "auc": best_val_auc,
        },
        "test": {
            "auc":       te_auc,
            "acc":       te_acc,
            "precision": float(precision_score(te_labels, te_preds, zero_division=0)),
            "recall":    float(recall_score(te_labels, te_preds, zero_division=0)),
            "f1":        float(f1_score(te_labels, te_preds, zero_division=0)),
            "cm":        confusion_matrix(te_labels, te_preds).tolist(),
            "probs":     te_probs,
            "labels":    te_labels,
        },
    }
    res_path = os.path.join(log_dir, "results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLogs:       {log_dir}/")
    print(f"Checkpoint: {ckpt_dir}/best.pt")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="abmil",
                        choices=["abmil", "gated_abmil", "meanpool_mlp"])
    parser.add_argument("--tag",         type=str,   default="mini",
                        help="Split tag: 'mini' or '' for full")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--wd",          type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.25)
    parser.add_argument("--hidden-dim",  type=int,   default=512)
    parser.add_argument("--max-tiles",   type=int,   default=500)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
