"""
splits.py
---------
Generates patient-level train/val/test splits from labels.csv.
Supports mini-experiment mode via --max-patients-per-class.

Outputs:
  project/splits/train.csv
  project/splits/val.csv
  project/splits/test.csv
  project/splits/split_stats.json

Usage:
  # Full dataset (70/15/15)
  python3 splits.py

  # Mini experiment: 100 train / 50 val / 50 test patients per class
  python3 splits.py --max-patients-per-class 100 --val-patients 50 --test-patients 50 --tag mini
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

BASE_DIR   = os.environ.get("DSAI543_DATA", os.path.expanduser("~/research_data"))
LABELS_CSV = os.path.join(BASE_DIR, "embeddings", "labels.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "project", "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)


def load_labels(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def build_patient_map(rows):
    patient_to_slides = defaultdict(list)
    for r in rows:
        patient_to_slides[r["patient_id"]].append(r)

    luad_patients, lusc_patients = [], []
    for pid, slides in patient_to_slides.items():
        labels   = [s["label"] for s in slides]
        majority = max(set(labels), key=labels.count)
        if majority == "LUAD":
            luad_patients.append(pid)
        else:
            lusc_patients.append(pid)

    return patient_to_slides, luad_patients, lusc_patients


def split_patients(rows, train_ratio, val_ratio, seed,
                   max_train=None, max_val=None, max_test=None):
    random.seed(seed)
    _, luad_pats, lusc_pats = build_patient_map(rows)

    random.shuffle(luad_pats)
    random.shuffle(lusc_pats)

    def do_split(patients):
        n = len(patients)
        if max_train is not None:
            n_train = min(max_train, n)
            n_val   = min(max_val,   n - n_train)
            n_test  = min(max_test,  n - n_train - n_val)
        else:
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            n_test  = n - n_train - n_val
        return (
            patients[:n_train],
            patients[n_train:n_train + n_val],
            patients[n_train + n_val:n_train + n_val + n_test],
        )

    luad_tr, luad_va, luad_te = do_split(luad_pats)
    lusc_tr, lusc_va, lusc_te = do_split(lusc_pats)

    train_pids = set(luad_tr + lusc_tr)
    val_pids   = set(luad_va + lusc_va)
    test_pids  = set(luad_te + lusc_te)

    train_rows = [r for r in rows if r["patient_id"] in train_pids]
    val_rows   = [r for r in rows if r["patient_id"] in val_pids]
    test_rows  = [r for r in rows if r["patient_id"] in test_pids]

    return train_rows, val_rows, test_rows


def write_split(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def split_stats(rows, name):
    return {
        "split":    name,
        "slides":   len(rows),
        "LUAD":     sum(1 for r in rows if r["label"] == "LUAD"),
        "LUSC":     sum(1 for r in rows if r["label"] == "LUSC"),
        "patients": len(set(r["patient_id"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",                   type=int,   default=42)
    parser.add_argument("--train",                  type=float, default=0.70)
    parser.add_argument("--val",                    type=float, default=0.15)
    parser.add_argument("--test",                   type=float, default=0.15)
    parser.add_argument("--max-patients-per-class", type=int,   default=None,
                        help="Train patients per class (mini mode)")
    parser.add_argument("--val-patients",           type=int,   default=50,
                        help="Val patients per class in mini mode")
    parser.add_argument("--test-patients",          type=int,   default=50,
                        help="Test patients per class in mini mode")
    parser.add_argument("--tag",                    type=str,   default="",
                        help="Output filename suffix e.g. 'mini'")
    args = parser.parse_args()

    print(f"Loading {LABELS_CSV}...")
    rows = load_labels(LABELS_CSV)
    n_patients = len(set(r["patient_id"] for r in rows))
    print(f"  {len(rows)} slides | {n_patients} unique patients")
    print(f"  LUAD: {sum(1 for r in rows if r['label']=='LUAD')} | "
          f"LUSC: {sum(1 for r in rows if r['label']=='LUSC')}")

    mini = args.max_patients_per_class is not None
    if mini:
        print(f"\nMini mode: {args.max_patients_per_class} train / "
              f"{args.val_patients} val / {args.test_patients} test per class")
        train, val, test = split_patients(
            rows, args.train, args.val, args.seed,
            max_train=args.max_patients_per_class,
            max_val=args.val_patients,
            max_test=args.test_patients,
        )
    else:
        print(f"\nFull mode: {args.train:.0%}/{args.val:.0%}/{args.test:.0%} "
              f"(seed={args.seed})")
        train, val, test = split_patients(rows, args.train, args.val, args.seed)

    # Leakage check
    train_pids = set(r["patient_id"] for r in train)
    val_pids   = set(r["patient_id"] for r in val)
    test_pids  = set(r["patient_id"] for r in test)
    assert not (train_pids & val_pids),  "Leakage: train/val"
    assert not (train_pids & test_pids), "Leakage: train/test"
    assert not (val_pids   & test_pids), "Leakage: val/test"
    print("  ✓ No patient leakage")

    # Write splits
    tag = f"_{args.tag}" if args.tag else ""
    for split_rows, name in [(train, "train"), (val, "val"), (test, "test")]:
        write_split(split_rows, os.path.join(SPLITS_DIR, f"{name}{tag}.csv"))

    # Print table
    stats = [split_stats(train, "train"), split_stats(val, "val"), split_stats(test, "test")]
    print(f"\n{'Split':<8} {'Slides':>8} {'LUAD':>8} {'LUSC':>8} {'Patients':>10}")
    print("-" * 46)
    for s in stats:
        print(f"{s['split']:<8} {s['slides']:>8} {s['LUAD']:>8} "
              f"{s['LUSC']:>8} {s['patients']:>10}")

    # Save stats JSON
    stats_path = os.path.join(SPLITS_DIR, f"split_stats{tag}.json")
    with open(stats_path, "w") as f:
        json.dump({"seed": args.seed, "tag": args.tag,
                   "mini": mini, "splits": stats}, f, indent=2)

    print(f"\nSaved to {SPLITS_DIR}/")
    for name in ["train", "val", "test"]:
        print(f"  {name}{tag}.csv")
    print(f"  split_stats{tag}.json")


if __name__ == "__main__":
    main()
