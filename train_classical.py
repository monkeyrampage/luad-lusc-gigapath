"""
train_classical.py
------------------
Trains classical feature baselines on HOG+LBP+GLCM embeddings.
  1. RBF-SVM  (RBF-SVM baseline)
  2. Classical MLP

Usage:
  python3 train_classical.py --tag mini
  python3 train_classical.py --tag mini --model svm
  python3 train_classical.py --tag mini --model mlp
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from data.dataset import load_split_csv

BASE_DIR    = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
SPLITS_DIR  = os.path.join(BASE_DIR, "project", "splits")
RESULTS_DIR = os.path.join(BASE_DIR, "project", "results")
CL_H5       = os.path.join(BASE_DIR, "embeddings", "classical_embeddings.h5")


def load_classical_split(split_csv):
    import h5py
    rows = load_split_csv(split_csv)
    X, y, fids = [], [], []
    with h5py.File(CL_H5, "r") as h:
        available = set(h.keys())
        for r in rows:
            fid = r["file_id"]
            if fid not in available:
                continue
            feats = h[fid]["features"][:]
            X.append(feats.mean(axis=0))
            y.append(int(r["label_int"]))
            fids.append(fid)
    return np.stack(X), np.array(y), fids


def get_paths(tag):
    suffix = f"_{tag}" if tag else ""
    return {
        "train": os.path.join(SPLITS_DIR, f"train{suffix}.csv"),
        "val":   os.path.join(SPLITS_DIR, f"val{suffix}.csv"),
        "test":  os.path.join(SPLITS_DIR, f"test{suffix}.csv"),
    }


def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        "auc":       float(roc_auc_score(y_true, probs)),
        "acc":       float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "cm":        confusion_matrix(y_true, preds).tolist(),
        "probs":     probs.tolist(),
        "labels":    y_true.tolist(),
    }


# ─── SVM ──────────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, args):
    print("\n── RBF-SVM ─────────────────────────────")
    t0   = time.time()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=args.C, gamma=args.gamma,
                       probability=True, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    print(f"  Fit time: {time.time()-t0:.1f}s")

    results = {}
    for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        probs          = pipe.predict_proba(X)[:, 1]
        results[split] = compute_metrics(y, probs)
        print(f"  {split}: AUC={results[split]['auc']:.4f}  "
              f"Acc={results[split]['acc']:.4f}  "
              f"F1={results[split]['f1']:.4f}")
    return results


# ─── Classical MLP ────────────────────────────────────────────────────────────

class ClassicalMLP(nn.Module):
    def __init__(self, input_dim=404, hidden_dim=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n── Classical MLP ({device}) ─────────────")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    def to_tensor(X, y):
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

    Xtr, ytr = to_tensor(X_train, y_train)
    Xva, yva = to_tensor(X_val,   y_val)
    Xte, yte = to_tensor(X_test,  y_test)

    model     = ClassicalMLP(input_dim=X_train.shape[1],
                             dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_auc, patience_left, best_state = 0.0, args.patience, None
    history = []

    print(f"{'Ep':>4} {'Loss':>8} {'T-AUC':>7} {'V-AUC':>7}")
    print("-" * 32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr.to(device)), ytr.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            tr_probs = torch.softmax(model(Xtr.to(device)), dim=1)[:, 1].cpu().numpy()
            va_probs = torch.softmax(model(Xva.to(device)), dim=1)[:, 1].cpu().numpy()

        tr_auc = roc_auc_score(y_train, tr_probs)
        va_auc = roc_auc_score(y_val,   va_probs)
        print(f"{epoch:>4} {loss.item():>8.4f} {tr_auc:>7.4f} {va_auc:>7.4f}")
        history.append({"epoch": epoch, "loss": loss.item(),
                        "train_auc": tr_auc, "val_auc": va_auc})

        if va_auc > best_val_auc:
            best_val_auc  = va_auc
            patience_left = args.patience
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    results = {}
    with torch.inference_mode():
        for split, X, y_np in [("val",  Xva, y_val),
                                ("test", Xte, y_test)]:
            probs          = torch.softmax(model(X.to(device)), dim=1)[:, 1].cpu().numpy()
            results[split] = compute_metrics(y_np, probs)
            print(f"  {split}: AUC={results[split]['auc']:.4f}  "
                  f"Acc={results[split]['acc']:.4f}  "
                  f"F1={results[split]['f1']:.4f}")

    return results, history


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",      type=str,   default="mini")
    parser.add_argument("--model",    type=str,   default="both",
                        choices=["svm", "mlp", "both"])
    parser.add_argument("--C",        type=float, default=1.0)
    parser.add_argument("--gamma",    type=str,   default="scale")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--wd",       type=float, default=1e-4)
    parser.add_argument("--dropout",  type=float, default=0.25)
    parser.add_argument("--patience", type=int,   default=15)
    args = parser.parse_args()

    paths = get_paths(args.tag)
    print("Loading classical features...")
    t0 = time.time()
    X_tr, y_tr, _ = load_classical_split(paths["train"])
    X_va, y_va, _ = load_classical_split(paths["val"])
    X_te, y_te, _ = load_classical_split(paths["test"])
    print(f"  Train={X_tr.shape} Val={X_va.shape} Test={X_te.shape} "
          f"({time.time()-t0:.1f}s)")

    all_results = {}

    if args.model in ("svm", "both"):
        all_results["svm"] = train_svm(X_tr, y_tr, X_va, y_va, X_te, y_te, args)

    if args.model in ("mlp", "both"):
        res, _ = train_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, args)
        all_results["classical_mlp"] = res

    # Save
    for name, res in all_results.items():
        log_dir = os.path.join(RESULTS_DIR, "logs", f"{name}_{args.tag}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "results.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved {name} → {log_dir}/results.json")

    # Summary table
    print(f"\n{'='*55}")
    print(f"{'Model':<20} {'Val AUC':>9} {'Test AUC':>9} {'Test Acc':>9}")
    print("-" * 55)
    for name, res in all_results.items():
        print(f"{name:<20} {res['val']['auc']:>9.4f} "
              f"{res['test']['auc']:>9.4f} {res['test']['acc']:>9.4f}")


if __name__ == "__main__":
    main()
