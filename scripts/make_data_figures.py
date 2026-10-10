"""Step 2: look at the data before modelling.

Writes two figures to docs/figures/:
  class_grid.png      - 8 images per class, drawn from as many different
                        videos as possible (caption: video_frame, split)
  same_name_pairs.png - every filename that exists in two class folders,
                        shown side by side with both folder labels

Usage: python scripts/make_data_figures.py
"""
import collections
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cholec-tinytools"
OUT = ROOT / "docs" / "figures"
CLASSES = ["clipper", "grasper", "hook", "scissor"]
SCALE = 2  # 128x86 -> 256x172 so details are visible
FONT = ImageFont.load_default(size=13)


def all_paths():
    return sorted(DATA.glob("*/*/*.png"))


def tile(path, caption):
    img = Image.open(path).convert("RGB")
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + 18), "white")
    canvas.paste(img, (0, 0))
    ImageDraw.Draw(canvas).text((3, img.height + 2), caption, fill="black", font=FONT)
    return canvas


def grid(rows, row_labels, label_w=90, pad=4):
    tw, th = rows[0][0].size
    ncol = max(len(r) for r in rows)
    W = label_w + ncol * (tw + pad)
    H = len(rows) * (th + pad)
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    for i, (r, lab) in enumerate(zip(rows, row_labels)):
        y = i * (th + pad)
        d.text((4, y + th // 2 - 8), lab, fill="black", font=FONT)
        for j, t in enumerate(r):
            out.paste(t, (label_w + j * (tw + pad), y))
    return out


def class_grid(paths, per_class=8, seed=0):
    rng = random.Random(seed)
    rows = []
    for cls in CLASSES:
        by_video = collections.defaultdict(list)
        for p in paths:
            if p.parent.name == cls:
                by_video[p.name.split("_")[0]].append(p)
        # round-robin over videos so one long video doesn't fill the row
        vids = sorted(by_video)
        rng.shuffle(vids)
        for v in vids:
            rng.shuffle(by_video[v])
        picks = []
        while len(picks) < per_class and any(by_video.values()):
            for v in vids:
                if by_video[v] and len(picks) < per_class:
                    picks.append(by_video[v].pop())
        rows.append([tile(p, f"{p.stem} ({p.parts[-3][:5]})") for p in picks])
    return grid(rows, CLASSES)


def pair_grid(paths):
    by_name = collections.defaultdict(list)
    for p in paths:
        by_name[p.name].append(p)
    pairs = sorted((v for v in by_name.values() if len(v) > 1), key=lambda v: v[0].name)
    tiles = []
    for v in pairs:
        for p in v:
            tiles.append(tile(p, f"{p.stem}  {p.parts[-3][:5]}/{p.parent.name}"))
    # 4 pairs per row -> 8 tiles per row; each pair occupies adjacent tiles
    rows = [tiles[i:i + 8] for i in range(0, len(tiles), 8)]
    return grid(rows, [""] * len(rows), label_w=0, pad=6), len(pairs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = all_paths()
    class_grid(paths).save(OUT / "class_grid.png")
    g, n = pair_grid(paths)
    g.save(OUT / "same_name_pairs.png")
    print(f"{len(paths)} images; wrote class_grid.png and same_name_pairs.png ({n} pairs)")


if __name__ == "__main__":
    main()
