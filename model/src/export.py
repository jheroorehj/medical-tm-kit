"""Export the model to deployable artifacts.

    python -m src.export --ckpt checkpoints/best.pt --formats pt torchscript onnx

* ``pt``          — ``state_dict`` + labels + config (reload with ``load_checkpoint``)
* ``torchscript`` — traced ``.pt``, loads with ``torch.jit.load`` and no source code
* ``onnx``        — for onnxruntime / onnxruntime-web deployment

If ``--ckpt`` is omitted the ImageNet-pretrained backbone is exported with a
randomly initialised head — useful to verify the toolchain before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .labels import ALL_LABELS
from .model import ModelConfig, build_model, load_checkpoint, save_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export RFMiD model artifacts")
    p.add_argument("--ckpt", default="", help="trained .pt; empty = pretrained backbone only")
    p.add_argument("--backbone", default="convnextv2_base")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--name", default="rfmid_convnextv2_base")
    p.add_argument("--formats", nargs="+", default=["pt", "torchscript"],
                   choices=["pt", "torchscript", "onnx"])
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        model, ckpt = load_checkpoint(args.ckpt)
        img_size = ckpt.get("img_size", args.img_size)
        print(f"loaded {args.ckpt}")
    else:
        model = build_model(ModelConfig(backbone=args.backbone, img_size=args.img_size))
        img_size = args.img_size
        print(f"built {args.backbone} from HF pretrained weights (untrained head)")

    model.eval()
    example = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        out = model(example)
    print(f"forward OK: {tuple(example.shape)} -> {tuple(out.shape)}")

    written = []
    if "pt" in args.formats:
        path = out_dir / f"{args.name}.pt"
        save_checkpoint(path, model)
        written.append(path)

    if "torchscript" in args.formats:
        path = out_dir / f"{args.name}_ts.pt"
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        traced.save(str(path))
        reloaded = torch.jit.load(str(path))
        assert torch.allclose(reloaded(example), out, atol=1e-4)
        written.append(path)

    if "onnx" in args.formats:
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise SystemExit("ONNX export needs the exporter package: pip install onnx")
        path = out_dir / f"{args.name}.onnx"
        torch.onnx.export(
            model, example, str(path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
        )
        written.append(path)
        try:
            import numpy as np
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input": example.numpy()})[0]
            print(f"onnxruntime parity: max|delta| = "
                  f"{np.abs(ort_out - out.numpy()).max():.2e}")
        except ImportError:
            print("onnxruntime not installed — skipped the parity check")

    labels_path = out_dir / f"{args.name}_labels.txt"
    labels_path.write_text("\n".join(ALL_LABELS), encoding="utf-8")
    written.append(labels_path)

    print("\nwrote:")
    for p in written:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
