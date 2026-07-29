"""Run a trained checkpoint on loose fundus images.

    python -m src.predict --ckpt checkpoints/best.pt --images path/to/img.png path/to/dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .engine import pick_device
from .labels import ALL_LABELS, DISEASE_LABELS, LABEL_FULL_NAME
from .model import load_checkpoint

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect(paths) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS)
        elif p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict RFMiD findings for images")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--json-out", default="")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    files = collect(args.images)
    if not files:
        raise SystemExit("no images found")

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()
    tf = build_transforms(ckpt.get("img_size", 224), train=False)
    thresholds = ckpt.get("thresholds") or [0.5] * len(ALL_LABELS)

    results = []
    for f in files:
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x))
            if args.tta:
                p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])))) / 2
        probs = p[0].float().cpu().tolist()

        by_label = dict(zip(ALL_LABELS, probs))
        diseases = sorted(((n, by_label[n]) for n in DISEASE_LABELS),
                          key=lambda kv: -kv[1])[:args.topk]
        flagged = [n for n in DISEASE_LABELS
                   if by_label[n] >= thresholds[ALL_LABELS.index(n)]]

        print(f"\n{f.name}")
        print(f"  Disease_Risk (abnormal): {by_label['Disease_Risk']:.3f}")
        print(f"  top-{args.topk} findings:")
        for name, v in diseases:
            mark = "*" if name in flagged else " "
            print(f"   {mark} {name:<6} {v:.3f}  {LABEL_FULL_NAME.get(name, '')}")
        results.append({"file": str(f), "probs": by_label, "flagged": flagged})

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
