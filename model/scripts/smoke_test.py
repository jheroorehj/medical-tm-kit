"""End-to-end sanity check — no dataset download required.

    python -m scripts.smoke_test

1. Pulls ConvNeXt V2-Base weights from the Hugging Face Hub via timm.
2. Runs a forward pass on a synthetic fundus image through the real
   preprocessing pipeline (FOV crop -> square pad -> resize -> normalise).
3. Computes the RIADD metric bundle on random predictions (must land near 0.5).
4. Saves ``checkpoints/smoke_convnextv2_base.pt``, reloads it, and checks the
   outputs match bit-for-bit.
5. Traces TorchScript and verifies the traced module agrees.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.data import build_transforms  # noqa: E402
from src.engine import pick_device  # noqa: E402
from src.labels import ALL_LABELS, NUM_CLASSES  # noqa: E402
from src.losses import build_loss, pos_weight_from_counts  # noqa: E402
from src.metrics import format_report, riadd_score, to_riadd_view  # noqa: E402
from src.model import ModelConfig, build_model, load_checkpoint, save_checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "checkpoints"


def synthetic_fundus(size=(1424, 2144), seed=0) -> Image.Image:
    """A letterboxed disc on black — mimics the raw RFMiD framing."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = min(h, w) // 2 - 10
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 70, 30))
    draw.ellipse([cx + r // 2 - 60, cy - 60, cx + r // 2 + 60, cy + 60], fill=(230, 190, 90))
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-12, 12, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def step(n: int, msg: str) -> None:
    print(f"\n[{n}] {msg}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device("auto")
    print(f"torch {torch.__version__} | device {device}")

    step(1, "building ConvNeXt V2-Base from Hugging Face (timm)")
    t0 = time.time()
    model = build_model(ModelConfig(backbone="convnextv2_base", img_size=224)).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"    timm/convnextv2_base.fcmae_ft_in22k_in1k")
    print(f"    params {params/1e6:.2f}M | head -> {NUM_CLASSES} logits | {time.time()-t0:.1f}s")

    step(2, "preprocessing + forward pass")
    raw = synthetic_fundus()
    tf = build_transforms(224, train=False)
    x = tf(raw).unsqueeze(0).to(device)
    print(f"    raw {raw.size} -> tensor {tuple(x.shape)} "
          f"(range {x.min():.2f}..{x.max():.2f})")
    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits)[0].cpu()
    print(f"    logits {tuple(logits.shape)} in {(time.time()-t0)*1e3:.0f} ms")
    assert logits.shape == (1, NUM_CLASSES)
    top = torch.topk(probs, 5)
    print("    top-5 (untrained head, expect noise):")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"      {ALL_LABELS[i]:<14} {v:.3f}")

    step(3, "train-mode augmentation + loss + backward")
    tf_train = build_transforms(224, train=True)
    xb = torch.stack([tf_train(raw) for _ in range(2)]).to(device)
    yb = torch.zeros(2, NUM_CLASSES, device=device)
    yb[:, 0] = 1        # Disease_Risk
    yb[0, 1] = 1        # DR
    yb[1, 3] = 1        # MH
    for name in ("asl", "bce"):
        pw = pos_weight_from_counts(np.full(NUM_CLASSES, 50.0), 1920).to(device)
        loss = build_loss(name, pw).to(device)(model.train()(xb), yb)
        loss.backward()
        print(f"    {name:<4} loss {loss.item():.4f}  grad_ok="
              f"{model.get_classifier().weight.grad is not None}")
        model.zero_grad(set_to_none=True)
    model.eval()

    step(4, "RIADD metric on random predictions (expect ~0.5)")
    rng = np.random.default_rng(0)
    y = (rng.random((640, NUM_CLASSES)) < 0.08).astype(np.float32)
    y[:, 0] = (rng.random(640) < 0.79).astype(np.float32)
    p = rng.random((640, NUM_CLASSES))
    _, y_ml = to_riadd_view(y)
    print(f"    46 raw columns -> {y_ml.shape[1]} scored categories (27 + OTHER)")
    print(format_report(riadd_score(y, p), "random baseline")
          .split("  per-class AUC")[0])

    step(5, "checkpoint round-trip (.pt)")
    ckpt_path = OUT / "smoke_convnextv2_base.pt"
    save_checkpoint(ckpt_path, model, extra={"note": "untrained head, smoke test"})
    reloaded, meta = load_checkpoint(ckpt_path, map_location=device)
    reloaded.to(device).eval()
    with torch.no_grad():
        delta = (reloaded(x) - logits).abs().max().item()
    print(f"    saved {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)")
    print(f"    keys: {sorted(k for k in meta if k != 'state_dict')}")
    print(f"    reload max|delta| = {delta:.2e}")
    assert delta == 0.0

    step(6, "TorchScript trace")
    ts_path = OUT / "smoke_convnextv2_base_ts.pt"
    with torch.no_grad():
        traced = torch.jit.trace(model, x)
    traced.save(str(ts_path))
    with torch.no_grad():
        ts_delta = (torch.jit.load(str(ts_path))(x) - logits).abs().max().item()
    print(f"    saved {ts_path.name} ({ts_path.stat().st_size/1e6:.1f} MB)")
    print(f"    traced max|delta| = {ts_delta:.2e}")
    assert ts_delta < 1e-4

    print("\nALL CHECKS PASSED")
    print(f"  {ckpt_path}")
    print(f"  {ts_path}")
    print("\nnext: python -m scripts.download_data --root data/rfmid")
    print("      python -m src.train --data data/rfmid --epochs 20 --batch-size 16 --amp")


if __name__ == "__main__":
    main()
