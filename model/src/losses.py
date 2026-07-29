"""Losses for long-tailed multi-label classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al., ICCV 2021).

    Down-weights easy negatives and hard-clips very-easy ones. RFMiD is
    extremely negative-dominated (most of the 45 disease columns are 0 for
    almost every image), which is exactly the regime ASL was designed for.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = 1 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        with torch.no_grad():
            pt = p * targets + p_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            weight = (1 - pt) ** gamma
        return -(loss * weight).sum(dim=1).mean()


class WeightedBCE(nn.Module):
    """BCE-with-logits using per-class ``pos_weight``, capped to stay stable."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def pos_weight_from_counts(pos: np.ndarray, n: int, cap: float = 20.0) -> torch.Tensor:
    """``(neg / pos)`` per class, clamped — classes with 0 positives get 1.0."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = n - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


def build_loss(name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    if name == "asl":
        return AsymmetricLoss()
    if name == "bce":
        return WeightedBCE(pos_weight)
    raise ValueError(f"unknown loss {name!r} (expected 'asl' or 'bce')")
