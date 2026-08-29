"""
cptac_download_and_extract.py
------------------------------
Downloads CPTAC-LUAD and CPTAC-LSCC tumor slides from TCIA pathdb,
extracts GigaPath embeddings, and saves to HDF5.

Uses the same extraction pipeline as process_batch.py:
  - Level 1 (native 10x), 224x224 tiles
  - Background filter: mean < 220
  - GigaPath 1536-dim embeddings
  - Delete SVS after embedding (save-as-you-go)

Outputs:
  embeddings/cptac_gigapath_embeddings.h5
  embeddings/cptac_labels.csv

Usage:
  python3 cptac_download_and_extract.py --query-only   # just count slides
  python3 cptac_download_and_extract.py --download-only # download without extracting
  python3 cptac_download_and_extract.py                 # full pipeline
  python3 cptac_download_and_extract.py --max-slides 10 # test with 10 slides
"""

import argparse
import csv
import logging
import os
import time
import requests
import numpy as np
import h5py
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cptac_extraction.log"),
        logging.StreamHandler()
    ]
)

BASE_DIR   = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
EMBED_DIR  = os.path.join(BASE_DIR, "embeddings")
TMP_DIR    = os.path.join(BASE_DIR, "cptac_tmp")
GP_H5      = os.path.join(EMBED_DIR, "cptac_gigapath_v2.h5")
CL_H5      = os.path.join(EMBED_DIR, "cptac_classical_embeddings.h5")
LABELS_CSV = os.path.join(EMBED_DIR, "cptac_labels_v2.csv")
os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(TMP_DIR,   exist_ok=True)


# ─── Query TCIA pathdb ────────────────────────────────────────────────────────

def query_cptac_slides():
    """Query CPTAC-LUAD and CPTAC-LSCC tumor slides from pathdb."""
    from tcia_utils import pathdb
    import pandas as pd

    logging.info("Querying CPTAC-LUAD...")
    luad = pathdb.getImages("CPTAC-LUAD", format="df")
    logging.info(f"  Total LUAD images: {len(luad)}")

    logging.info("Querying CPTAC-LSCC...")
    lusc = pathdb.getImages("CPTAC-LSCC", format="df")
    logging.info(f"  Total LUSC images: {len(lusc)}")

    # Filter tumor slides only (slideId ending in odd number = tumor)
    # Based on CPTAC convention: _21 = tumor, _26 = normal
    luad_tumor = luad[luad["imageId"].str.endswith("-21")].copy()
    lusc_tumor = lusc[lusc["imageId"].str.endswith("-21")].copy()

    luad_tumor["label"]     = "LUAD"
    luad_tumor["label_int"] = 0
    lusc_tumor["label"]     = "LUSC"
    lusc_tumor["label_int"] = 1

    all_slides = pd.concat([luad_tumor, lusc_tumor], ignore_index=True)

    logging.info(f"\nFiltered tumor slides:")
    logging.info(f"  LUAD: {len(luad_tumor)} slides ({luad_tumor['subjectId'].nunique()} patients)")
    logging.info(f"  LUSC: {len(lusc_tumor)} slides ({lusc_tumor['subjectId'].nunique()} patients)")
    logging.info(f"  Total: {len(all_slides)} slides")
    logging.info(f"  Pixel size: {all_slides['physicalPixelSizeX'].unique()} μm/px")

    return all_slides


# ─── Download ─────────────────────────────────────────────────────────────────

def download_slide(url, dest_path, chunk_size=1024*1024):
    """Download a slide SVS file from TCIA."""
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
    return downloaded


# ─── Embedding extraction ─────────────────────────────────────────────────────

def load_gigapath(device):
    import timm, torch
    from torchvision import transforms
    model = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
    ).to(device).eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    return model, transform


def extract_embeddings(svs_path, gp_model, transform, device):
    """Extract GigaPath embeddings from SVS at level 1 (10x)."""
    import openslide
    import torch
    from PIL import Image

    slide = openslide.OpenSlide(str(svs_path))
    ds    = slide.level_downsamples[1]
    l1_w, l1_h = slide.level_dimensions[1]

    coords, tiles = [], []
    for x in range(0, l1_w, 224):
        for y in range(0, l1_h, 224):
            tile = slide.read_region(
                (int(x * ds), int(y * ds)), 1, (224, 224)
            ).convert("RGB")
            if np.mean(np.array(tile)) < 220:
                coords.append([x, y])
                tiles.append(np.array(tile))
    slide.close()

    if not coords:
        return None, None

    # GigaPath inference
    g_features = []
    with torch.no_grad():
        for i in range(0, len(tiles), 128):
            batch = torch.stack([
                transform(Image.fromarray(t)) for t in tiles[i:i+128]
            ]).to(device)
            g_features.append(gp_model(batch).cpu().numpy())

    g_all = np.concatenate(g_features, axis=0)
    return g_all, np.array(coords, dtype=np.int32)


def save_to_h5(h5_path, slide_id, features, coords):
    """Append slide embeddings to HDF5."""
    with h5py.File(h5_path, "a") as f:
        if slide_id in f:
            return
        grp = f.create_group(slide_id)
        grp.create_dataset("features", data=features, dtype="float32")
        grp.create_dataset("coords",   data=coords,   dtype="int32")


def already_done(slide_id):
    """Check if slide is already in H5."""
    if not os.path.exists(GP_H5):
        return False
    with h5py.File(GP_H5, "r") as f:
        return slide_id in f


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-only",    action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--max-slides",    type=int, default=0,
                        help="Max slides to process (0=all)")
    parser.add_argument("--batch-size",    type=int, default=128)
    args = parser.parse_args()

    # Query slides
    slides = query_cptac_slides()

    if args.query_only:
        print(slides[["subjectId","label","imageUrl"]].to_string())
        return

    if args.max_slides > 0:
        slides = slides.head(args.max_slides)
        logging.info(f"Limited to {args.max_slides} slides for testing")

    # Load model
    if not args.download_only:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading GigaPath on {device}...")
        gp_model, transform = load_gigapath(device)

    # Labels CSV
    label_rows = []

    # Process each slide
    total   = len(slides)
    done    = 0
    skipped = 0
    failed  = 0

    for idx, row in slides.iterrows():
        slide_id  = row["imageId"]
        subject   = row["subjectId"]
        label     = row["label"]
        label_int = row["label_int"]
        url       = row["imageUrl"]

        if already_done(slide_id):
            logging.info(f"[{done+skipped+failed+1}/{total}] SKIP {slide_id[:16]} already in H5")
            skipped += 1
            label_rows.append({
                "slide_id":   slide_id,
                "subject_id": subject,
                "label":      label,
                "label_int":  label_int,
            })
            continue

        svs_path = os.path.join(TMP_DIR, f"{slide_id}.svs")
        t0       = time.time()

        try:
            # Download
            logging.info(f"[{done+skipped+failed+1}/{total}] Downloading {subject} ({label})...")
            downloaded = download_slide(url, svs_path)
            size_mb    = downloaded / 1e6
            logging.info(f"  Downloaded: {size_mb:.0f} MB")

            if args.download_only:
                done += 1
                continue

            # Extract
            logging.info(f"  Extracting embeddings...")
            features, coords = extract_embeddings(svs_path, gp_model, transform, device)

            if features is None:
                logging.warning(f"  No tissue found — skipping")
                failed += 1
                os.remove(svs_path)
                continue

            # Save
            save_to_h5(GP_H5, slide_id, features, coords)
            label_rows.append({
                "slide_id":   slide_id,
                "subject_id": subject,
                "label":      label,
                "label_int":  label_int,
            })

            # Delete SVS
            os.remove(svs_path)
            elapsed = time.time() - t0
            logging.info(f"  Done | Tiles: {len(features)} | {elapsed:.1f}s")
            done += 1

        except Exception as e:
            logging.error(f"  FAILED: {e}")
            if os.path.exists(svs_path):
                os.remove(svs_path)
            failed += 1

    # Save labels CSV
    if label_rows:
        with open(LABELS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=label_rows[0].keys())
            w.writeheader()
            w.writerows(label_rows)
        logging.info(f"\nLabels saved: {LABELS_CSV}")

    # Summary
    logging.info(f"\n{'='*50}")
    logging.info(f"Done:    {done}")
    logging.info(f"Skipped: {skipped}")
    logging.info(f"Failed:  {failed}")
    if os.path.exists(GP_H5):
        with h5py.File(GP_H5, "r") as f:
            logging.info(f"H5 keys: {len(f.keys())}")


if __name__ == "__main__":
    main()
