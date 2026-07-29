"""ConvNeXt V2-Base multi-label classifier for RFMiD.

Backbone weights come from the Hugging Face Hub through timm
(``timm/convnextv2_base.fcmae_ft_in22k_in1k``, 87.7M params). The head is a
plain 46-way linear layer producing independent logits — one sigmoid per
label, since RFMiD images carry several findings at once.

License note: the ConvNeXt V2 *code* is MIT, but Meta's released ImageNet
pre-trained weights are CC BY-NC 4.0. The timm model card says apache-2.0;
the upstream repo (facebookresearch/ConvNeXt-V2) says non-commercial. Treat
them as non-commercial and cite the model license alongside the dataset's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .labels import ALL_LABELS, NUM_CLASSES

DEFAULT_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"

# Handy alternatives, all pretrained in timm.
BACKBONE_PRESETS = {
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",       # 88M  — rank 1
    "convnextv2_base_384": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",       # 28M  — fast baseline
    "convnextv2_large": "convnextv2_large.fcmae_ft_in22k_in1k",     # 198M
    "swinv2_base": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "effnetv2_m": "tf_efficientnetv2_m.in21k_ft_in1k",              # ensemble partner
}


@dataclass
class ModelConfig:
    backbone: str = DEFAULT_BACKBONE
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    drop_rate: float = 0.2          # head dropout
    drop_path_rate: float = 0.2     # stochastic depth
    img_size: int = 224


def resolve_backbone(name: str) -> str:
    return BACKBONE_PRESETS.get(name, name)


def build_model(cfg: ModelConfig | None = None, **kwargs) -> nn.Module:
    """Create the multi-label model, downloading HF weights on first call."""
    import timm

    cfg = cfg or ModelConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)

    model = timm.create_model(
        resolve_backbone(cfg.backbone),
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    model.cfg = cfg  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def predict_probs(model: nn.Module, x: torch.Tensor, tta_hflip: bool = False) -> torch.Tensor:
    """Sigmoid probabilities, optionally averaged with a horizontal-flip pass."""
    model.eval()
    logits = model(x)
    probs = torch.sigmoid(logits)
    if tta_hflip:
        probs = (probs + torch.sigmoid(model(torch.flip(x, dims=[3])))) / 2
    return probs


def save_checkpoint(path, model: nn.Module, extra: dict | None = None) -> None:
    """Write a self-describing ``.pt`` — weights plus everything needed to reload."""
    cfg = getattr(model, "cfg", ModelConfig())
    payload = {
        "state_dict": model.state_dict(),
        "backbone": resolve_backbone(cfg.backbone),
        "num_classes": cfg.num_classes,
        "img_size": cfg.img_size,
        "labels": ALL_LABELS,
        "task": "rfmid-multilabel",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from a ``.pt`` written by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(
        backbone=ckpt.get("backbone", DEFAULT_BACKBONE),
        num_classes=ckpt.get("num_classes", NUM_CLASSES),
        pretrained=False,
        img_size=ckpt.get("img_size", 224),
    )
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
