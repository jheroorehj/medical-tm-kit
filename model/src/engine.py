"""Train / evaluate loops shared by the CLI entry points."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import riadd_score


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, scheduler,
                    device, scaler=None, grad_accum: int = 1,
                    max_norm: float = 1.0, log_every: int = 20, epoch: int = 0) -> float:
    model.train()
    total, seen = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(x), y) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total += loss.item() * grad_accum * x.size(0)
        seen += x.size(0)
        if log_every and step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch} step {step:>4}/{len(loader)}  "
                  f"loss {total / max(seen, 1):.4f}  lr {lr:.2e}  "
                  f"{seen / max(time.time() - t0, 1e-9):.1f} img/s", flush=True)

    return total / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, criterion=None,
             tta_hflip: bool = False) -> dict:
    model.eval()
    probs, targets, ids = [], [], []
    loss_sum, seen = 0.0, 0

    for x, y, batch_ids in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y_dev).item() * x.size(0)
            seen += x.size(0)
        p = torch.sigmoid(logits)
        if tta_hflip:
            p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])))) / 2
        probs.append(p.float().cpu().numpy())
        targets.append(y.numpy())
        ids.extend(batch_ids)

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    out = riadd_score(targets, probs)
    out["loss"] = loss_sum / seen if seen else float("nan")
    out["probs"] = probs
    out["targets"] = targets
    out["ids"] = ids
    return out
