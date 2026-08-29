"""
learning_curve.py
-----------------
Trains ABMIL and BoVW+SVM on increasing training set sizes.
Val and test sets are fixed (full splits).
Reports test AUC at each training size.

Models:
  - ABMIL (GigaPath embeddings)
  - MeanPool MLP (GigaPath embeddings)
  - BoVW + SVM (Classical features — best classical baseline)

Usage:
  python3 learning_curve.py --tag full
  python3 learning_curve.py --tag full --models abmil bovw
"""

import argparse
import csv
import json
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from models.model import ABMIL, MeanPoolMLP

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
GP_H5      = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
CL_H5      = os.path.join(BASE_DIR, "embeddings", "classical_embeddings.h5")
SPLITS_DIR = os.path.join(BASE_DIR, "project", "splits")
OUT_DIR    = os.path.join(BASE_DIR, "project", "results", "figures")
LOG_DIR    = os.path.join(BASE_DIR, "project", "results", "logs", "learning_curve")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# Training sizes: patients per class
SIZES = [25, 50, 75, 100, 150, 200, 250, 300, "full"]
SEED  = 42


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_split(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def subsample_rows(rows, n_per_class, seed=SEED):
    """Sample n_per_class patients per label, return slide rows."""
    rng = random.Random(seed)
    patient_to_rows = defaultdict(list)
    for r in rows:
        patient_to_rows[r["patient_id"]].append(r)

    luad_pats = [p for p, rs in patient_to_rows.items()
                 if rs[0]["label"] == "LUAD"]
    lusc_pats = [p for p, rs in patient_to_rows.items()
                 if rs[0]["label"] == "LUSC"]

    rng.shuffle(luad_pats)
    rng.shuffle(lusc_pats)

    selected_pats = set(luad_pats[:n_per_class] + lusc_pats[:n_per_class])
    return [r for r in rows if r["patient_id"] in selected_pats]


def load_gp_features(rows, h5_path, max_tiles=500, train=True):
    """Load GigaPath bags for a list of rows."""
    import h5py
    bags, labels = [], []
    with h5py.File(h5_path, "r") as h:
        for r in rows:
            fid = r["file_id"]
            if fid not in h:
                continue
            feats = h[fid]["features"][:]
            if train and len(feats) > max_tiles:
                idx   = np.random.choice(len(feats), max_tiles, replace=False)
                feats = feats[idx]
            bags.append(feats)
            labels.append(int(r["label_int"]))
    return bags, np.array(labels)


def load_classical_meanpool(rows, h5_path):
    """Load mean-pooled classical features."""
    import h5py
    X, y = [], []
    with h5py.File(h5_path, "r") as h:
        for r in rows:
            fid = r["file_id"]
            if fid not in h:
                continue
            feats = h[fid]["features"][:].mean(axis=0)
            X.append(feats)
            y.append(int(r["label_int"]))
    return np.stack(X), np.array(y)


def load_classical_tiles(rows, h5_path, max_tiles=200):
    """Load per-tile classical features (for BoVW)."""
    import h5py
    slide_tiles, y = [], []
    with h5py.File(h5_path, "r") as h:
        for r in rows:
            fid = r["file_id"]
            if fid not in h:
                continue
            feats = h[fid]["features"][:]
            if len(feats) > max_tiles:
                idx   = np.random.choice(len(feats), max_tiles, replace=False)
                feats = feats[idx]
            slide_tiles.append(feats)
            y.append(int(r["label_int"]))
    return slide_tiles, np.array(y)


def compute_metrics(y_true, probs):
    preds = (np.array(probs) >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "acc": float(accuracy_score(y_true, preds)),
        "f1":  float(f1_score(y_true, preds, zero_division=0)),
    }


# ─── ABMIL training ───────────────────────────────────────────────────────────

def train_abmil_once(train_rows, val_rows, test_rows, model_cls,
                     device, epochs=50, patience=10, lr=1e-4):
    """Train one ABMIL model, return test metrics."""
    tr_bags, tr_y = load_gp_features(train_rows, GP_H5, train=True)
    va_bags, va_y = load_gp_features(val_rows,   GP_H5, train=False)
    te_bags, te_y = load_gp_features(test_rows,  GP_H5, train=False)

    model     = model_cls(input_dim=1536, hidden_dim=512,
                          n_classes=2, dropout=0.25).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    def run_epoch(bags, labels, train=True):
        model.train() if train else model.eval()
        all_probs, all_labels, total_loss = [], [], 0.0
        ctx = torch.enable_grad() if train else torch.inference_mode()
        with ctx:
            for bag_np, label in zip(bags, labels):
                bag    = torch.tensor(bag_np, dtype=torch.float32).to(device)
                label_t = torch.tensor([label], dtype=torch.long).to(device)
                logits, _ = model(bag)
                loss   = criterion(logits.unsqueeze(0), label_t)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                prob = torch.softmax(logits, dim=0)[1].item()
                all_probs.append(prob)
                all_labels.append(label)
                total_loss += loss.item()
        auc = roc_auc_score(all_labels, all_probs)
        return total_loss / len(bags), auc, all_probs, all_labels

    best_val_auc, patience_left, best_state = 0.0, patience, None

    for epoch in range(1, epochs + 1):
        _, tr_auc, _, _ = run_epoch(tr_bags, tr_y, train=True)
        scheduler.step()
        _, va_auc, _, _ = run_epoch(va_bags, va_y, train=False)

        if va_auc > best_val_auc:
            best_val_auc  = va_auc
            patience_left = patience
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    _, _, te_probs, te_labels = run_epoch(te_bags, te_y, train=False)
    return compute_metrics(te_labels, te_probs), best_val_auc


# ─── BoVW training ────────────────────────────────────────────────────────────

def train_bovw_once(train_rows, val_rows, test_rows, K=256):
    tr_tiles, tr_y = load_classical_tiles(train_rows, CL_H5)
    va_tiles, va_y = load_classical_tiles(val_rows,   CL_H5)
    te_tiles, te_y = load_classical_tiles(test_rows,  CL_H5)

    all_tr = np.concatenate(tr_tiles)
    scaler = StandardScaler()
    all_tr_scaled = scaler.fit_transform(all_tr)

    kmeans = MiniBatchKMeans(n_clusters=K, random_state=SEED,
                             batch_size=4096, max_iter=100, n_init=3)
    kmeans.fit(all_tr_scaled)

    def encode(slide_list):
        X = []
        for tiles in slide_list:
            scaled  = scaler.transform(tiles)
            codes   = kmeans.predict(scaled)
            hist, _ = np.histogram(codes, bins=K, range=(0, K), density=True)
            X.append(hist)
        return np.stack(X)

    X_tr = encode(tr_tiles)
    X_va = encode(va_tiles)
    X_te = encode(te_tiles)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=SEED)),
    ])
    pipe.fit(X_tr, tr_y)

    te_probs = pipe.predict_proba(X_te)[:, 1]
    return compute_metrics(te_y, te_probs)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",     type=str,  default="full")
    parser.add_argument("--models",  nargs="+",
                        default=["abmil", "meanpool", "bovw"],
                        choices=["abmil", "meanpool", "bovw"])
    parser.add_argument("--repeats", type=int,  default=3,
                        help="Repeat each size N times with different seeds")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Repeats per size: {args.repeats}\n")

    suffix     = f"_{args.tag}" if args.tag else ""
    train_rows = load_split(os.path.join(SPLITS_DIR, f"train{suffix}.csv"))
    val_rows   = load_split(os.path.join(SPLITS_DIR, f"val{suffix}.csv"))
    test_rows  = load_split(os.path.join(SPLITS_DIR, f"test{suffix}.csv"))

    # Count max patients per class
    luad_pats = set(r["patient_id"] for r in train_rows if r["label"] == "LUAD")
    lusc_pats = set(r["patient_id"] for r in train_rows if r["label"] == "LUSC")
    max_per_class = min(len(luad_pats), len(lusc_pats))
    print(f"Max patients per class: {max_per_class}")

    results = {m: {} for m in args.models}

    for size in SIZES:
        n = max_per_class if size == "full" else int(size)
        if n > max_per_class:
            print(f"Skipping size {n} (exceeds {max_per_class})")
            continue

        print(f"\n{'='*55}")
        print(f"Training size: {n} patients/class ({n*2} total patients)")

        for model_name in args.models:
            aucs, accs, f1s = [], [], []

            for rep in range(args.repeats):
                seed     = SEED + rep * 100
                sub_rows = subsample_rows(train_rows, n, seed=seed)
                n_slides = len(sub_rows)
                t0       = time.time()

                print(f"  [{model_name}] rep={rep+1} slides={n_slides}...",
                      end=" ", flush=True)

                try:
                    if model_name == "abmil":
                        metrics, _ = train_abmil_once(
                            sub_rows, val_rows, test_rows,
                            ABMIL, device)
                    elif model_name == "meanpool":
                        metrics, _ = train_abmil_once(
                            sub_rows, val_rows, test_rows,
                            MeanPoolMLP, device)
                    elif model_name == "bovw":
                        metrics = train_bovw_once(
                            sub_rows, val_rows, test_rows)

                    aucs.append(metrics["auc"])
                    accs.append(metrics["acc"])
                    f1s.append(metrics["f1"])
                    print(f"AUC={metrics['auc']:.4f}  ({time.time()-t0:.0f}s)")

                except Exception as e:
                    print(f"ERROR: {e}")

            if aucs:
                results[model_name][str(n)] = {
                    "n_per_class":  n,
                    "auc_mean":     float(np.mean(aucs)),
                    "auc_std":      float(np.std(aucs)),
                    "auc_all":      aucs,
                    "acc_mean":     float(np.mean(accs)),
                    "f1_mean":      float(np.mean(f1s)),
                }
                print(f"  [{model_name}] size={n}: "
                      f"AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Save results
    out_json = os.path.join(LOG_DIR, f"learning_curve_{args.tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot
    plot_learning_curve(results, args.tag)


def plot_learning_curve(results, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MODEL_STYLES = {
        "abmil":    {"color": "#2196F3", "label": "ABMIL (GigaPath)",        "marker": "o", "ls": "-"},
        "meanpool": {"color": "#4CAF50", "label": "MeanPool MLP (GigaPath)", "marker": "s", "ls": "-"},
        "bovw":     {"color": "#FFC107", "label": "BoVW + SVM (Classical)",  "marker": "^", "ls": "--"},
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, size_results in results.items():
        if not size_results:
            continue
        style  = MODEL_STYLES.get(model_name, {})
        sizes  = sorted([int(k) for k in size_results.keys()])
        means  = [size_results[str(s)]["auc_mean"] for s in sizes]
        stds   = [size_results[str(s)]["auc_std"]  for s in sizes]
        slides = [size_results[str(s)]["n_per_class"] * 2 for s in sizes]

        ax.plot(sizes, means,
                color=style.get("color", "gray"),
                marker=style.get("marker", "o"),
                linestyle=style.get("ls", "-"),
                lw=2, markersize=7,
                label=style.get("label", model_name))
        ax.fill_between(sizes,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15,
                        color=style.get("color", "gray"))

    ax.set_xlabel("Training Patients per Class", fontsize=12)
    ax.set_ylabel("Test AUC", fontsize=12)
    ax.set_title("Learning Curve: Test AUC vs Training Set Size", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.02])

    # Add slide count on top x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    xticks = ax.get_xticks()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"~{int(x*2)}" for x in xticks], fontsize=9)
    ax2.set_xlabel("Approx. Training Slides", fontsize=10)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"learning_curve_{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
