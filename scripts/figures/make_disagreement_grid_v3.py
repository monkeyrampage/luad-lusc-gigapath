"""
make_disagreement_grid_v3.py
-----------------------------
IEEE-compliant disagreement thumbnail grid.
7 slides, 2 rows x 4 cols layout.
300 DPI, larger fonts, clean captions.

Usage:
  python -m scripts.figures.make_disagreement_grid_v3
"""

import os
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN_DIR = os.path.join(_PROJECT_ROOT, "results", "figures", "disagreement_thumbnails", "clean")
OUT_PATH  = os.path.join(_PROJECT_ROOT, "results", "figures", "disagreement_grid_v3.png")

SLIDES = [
    ("1365c2c0", "LUAD", 0.0001, 0.7733, "ABMIL",  773),
    ("1ec9435b", "LUAD", 0.0760, 0.9958, "ABMIL", 3182),
    ("fbcae59c", "LUAD", 0.1365, 0.9787, "ABMIL", 2028),
    ("331bb33f", "LUAD", 0.5353, 0.0010, "Gated", 4253),
    ("3dfc4d6e", "LUAD", 0.6657, 0.0204, "Gated", 1597),
    ("933463fd", "LUAD", 0.9669, 0.0014, "Gated", 7335),
    ("eed858e8", "LUSC", 0.0343, 0.9393, "Gated", 3042),
]

THUMB_SIZE   = 420
CAPTION_H    = 100
PAD          = 18
COLS         = 4
ROWS         = 2
BG           = (255, 255, 255)
CAP_BG       = (245, 245, 245)
BORDER_W     = 6
ABMIL_COLOR  = (33,  150, 243)
GATED_COLOR  = (156,  39, 176)
GREEN        = (0,  140,   0)
RED          = (200,   0,   0)
DARK         = (30,   30,  30)

CELL_W  = THUMB_SIZE
CELL_H  = THUMB_SIZE + CAPTION_H
TITLE_H = 36
GRID_W  = COLS * CELL_W + (COLS + 1) * PAD
GRID_H  = ROWS * CELL_H + (ROWS + 1) * PAD + TITLE_H

grid = Image.new("RGB", (GRID_W, GRID_H), BG)
draw = ImageDraw.Draw(grid)

# Title
draw.text((PAD, 8),
          "Disagreement Cases: ABMIL vs. Gated ABMIL  |  Blue border = ABMIL correct  |  Purple border = Gated correct",
          fill=DARK)

for i, (fid, true_label, abmil_p, gated_p, winner, n_tiles) in enumerate(SLIDES):
    row = i // COLS
    col = i % COLS
    x0  = PAD + col * (CELL_W + PAD)
    y0  = TITLE_H + PAD + row * (CELL_H + PAD)

    img_path = os.path.join(CLEAN_DIR, f"{fid}.png")
    if not os.path.exists(img_path):
        print(f"  [SKIP] {fid}")
        continue

    img = Image.open(img_path).convert("RGB")
    img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    grid.paste(img, (x0, y0))

    border_color = ABMIL_COLOR if winner == "ABMIL" else GATED_COLOR
    draw.rectangle([x0, y0, x0 + THUMB_SIZE - 1, y0 + THUMB_SIZE - 1],
                   outline=border_color, width=BORDER_W)

    # Caption
    cap_y = y0 + THUMB_SIZE
    draw.rectangle([x0, cap_y, x0 + CELL_W, cap_y + CAPTION_H], fill=CAP_BG)
    draw.line([x0, cap_y, x0 + CELL_W, cap_y], fill=border_color, width=2)

    abmil_pred  = "LUAD" if abmil_p < 0.5 else "LUSC"
    gated_pred  = "LUAD" if gated_p < 0.5 else "LUSC"
    abmil_color = GREEN if abmil_pred == true_label else RED
    gated_color = GREEN if gated_pred == true_label else RED

    draw.text((x0 + 8, cap_y + 6),
              f"True: {true_label}   Tiles: {n_tiles}", fill=DARK)
    draw.text((x0 + 8, cap_y + 28),
              f"ABMIL: {abmil_p:.3f}  ->  {abmil_pred}", fill=abmil_color)
    draw.text((x0 + 8, cap_y + 50),
              f"Gated: {gated_p:.3f}  ->  {gated_pred}", fill=gated_color)
    draw.text((x0 + 8, cap_y + 72),
              f"Correct: {winner}", fill=border_color)

grid.save(OUT_PATH, dpi=(300, 300))
print(f"Saved: {OUT_PATH}")
