"""Score a checkpoint on an official RFMiD split and dump per-image predictions.

    python -m src.evaluate --ckpt checkpoints/best.pt --data data/rfmid --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from .data import RFMiDDataset
from .engine import evaluate, pick_device
from .labels import ALL_LABELS
from .metrics import format_report
from .model import load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a RFMiD checkpoint")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--data", default="data/rfmid")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--out", default="reports")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    img_size = ckpt.get("img_size", 224)
    print(f"loaded {args.ckpt} (backbone={ckpt.get('backbone')}, img_size={img_size}, "
          f"trained epoch={ckpt.get('epoch')}) on {device}")

    ds = RFMiDDataset(args.data, args.split, img_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
    m = evaluate(model, loader, device, tta_hflip=args.tta)
    print(format_report(m, f"{args.split} split (n={len(ds)})"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(m["probs"], columns=ALL_LABELS)
    pred.insert(0, "ID", m["ids"])
    pred_path = out_dir / f"predictions_{args.split}.csv"
    pred.to_csv(pred_path, index=False)

    summary = {k: v for k, v in m.items() if k not in ("probs", "targets", "ids")}
    json_path = out_dir / f"metrics_{args.split}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {pred_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
