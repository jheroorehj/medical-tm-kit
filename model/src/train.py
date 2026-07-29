"""Fine-tune ConvNeXt V2-Base on RFMiD (multi-label, 46 heads).

    python -m src.train --data data/rfmid --epochs 20 --batch-size 16 --amp

Saves ``checkpoints/best.pt`` (highest RIADD final score on the official
validation split) and ``checkpoints/last.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import RFMiDDataset
from .engine import cosine_warmup, evaluate, pick_device, train_one_epoch
from .labels import ALL_LABELS
from .losses import build_loss, pos_weight_from_counts
from .metrics import best_thresholds, format_report
from .model import ModelConfig, build_model, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RFMiD multi-label fine-tuning")
    p.add_argument("--data", default="data/rfmid", help="root holding the official splits")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--backbone", default="convnextv2_base")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="head / peak LR")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--loss", default="asl", choices=["asl", "bce"])
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    p.add_argument("--tta", action="store_true", help="hflip TTA at eval time")
    p.add_argument("--no-fov-crop", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0, help="debug: cap train images")
    p.add_argument("--limit-val", type=int, default=0, help="debug: cap val images")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  amp={use_amp}")

    fov = not args.no_fov_crop
    train_ds = RFMiDDataset(args.data, "train", args.img_size, train=True, fov_crop=fov)
    val_ds = RFMiDDataset(args.data, "val", args.img_size, train=False, fov_crop=fov)
    pos = train_ds.pos_counts()
    n_train_full = len(train_ds)

    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, n_train_full)))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    print(f"train={len(train_ds)}  val={len(val_ds)}  classes={len(ALL_LABELS)}")

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = build_model(ModelConfig(
        backbone=args.backbone, drop_path_rate=args.drop_path, img_size=args.img_size
    )).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = build_loss(
        args.loss, pos_weight_from_counts(pos, n_train_full).to(device)
    ).to(device)

    # Discriminative LR: the pretrained trunk moves slower than the fresh head.
    head_names = {n for n, _ in model.named_parameters() if n.startswith("head.")}
    groups = [
        {"params": [p for n, p in model.named_parameters() if n in head_names],
         "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if n not in head_names],
         "lr": args.lr * args.backbone_lr_mult},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = cosine_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                               device, scaler, args.grad_accum, epoch=epoch)
        m = evaluate(model, val_loader, device, criterion, tta_hflip=args.tta)
        print(f"[epoch {epoch}] train_loss {loss:.4f} | val_loss {m['loss']:.4f} | "
              f"A {m['score_a']:.4f}  B {m['score_b']:.4f}  FINAL {m['final_score']:.4f}",
              flush=True)

        history.append({k: m[k] for k in
                        ("loss", "score_a", "score_b", "final_score", "map_multi", "mauc_multi")}
                       | {"epoch": epoch, "train_loss": loss})

        # A degenerate split (e.g. --limit-val, or a subset where every image
        # is abnormal) makes the binary AUC — and therefore the final score —
        # undefined. Fall back to Score B so checkpointing still works.
        selection = m["final_score"] if np.isfinite(m["final_score"]) else m["score_b"]
        if np.isfinite(selection) and selection > best:
            best = selection
            thr = best_thresholds(m["targets"], m["probs"])
            save_checkpoint(out_dir / "best.pt", model, extra={
                "epoch": epoch,
                "val_final_score": best,
                "val_score_a": m["score_a"],
                "val_score_b": m["score_b"],
                "thresholds": thr.tolist(),
                "args": vars(args),
            })
            print(f"  -> new best {best:.4f}, saved {out_dir / 'best.pt'}")
            print(format_report(m, f"validation @ epoch {epoch}"))

    save_checkpoint(out_dir / "last.pt", model, extra={"epoch": args.epochs, "args": vars(args)})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\ndone. best val score = {best:.4f}")
    print(f"weights: {out_dir/'last.pt'}"
          + (f"  |  {out_dir/'best.pt'}" if (out_dir / "best.pt").exists() else ""))


if __name__ == "__main__":
    main()
