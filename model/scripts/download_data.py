"""Fetch the official RFMiD splits from the Hugging Face Hub.

    python -m scripts.download_data --root data/rfmid

Dataset: ctmedtech/RFMID — 3,200 images (1,920 / 640 / 640), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import SPLITS, download_rfmid  # noqa: E402
from src.labels import ALL_LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download RFMiD")
    ap.add_argument("--root", default="data/rfmid")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--csv-only", action="store_true", help="labels without images")
    ap.add_argument("--max-images", type=int, default=0,
                    help="debug: fetch only the first N images per split "
                         "(IDs 1..N, which is what --limit-train/--limit-val consume)")
    args = ap.parse_args()

    root = Path(args.root)
    if args.csv_only or args.max_images:
        from huggingface_hub import hf_hub_download

        def grab(remote: str) -> Path:
            dest = root / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(
                    Path(hf_hub_download("ctmedtech/RFMID", remote,
                                         repo_type="dataset")).read_bytes())
            return dest

        for name in args.splits:
            spec = SPLITS[name]
            print(f"{name}: {grab(spec.csv)}")
            for i in range(1, args.max_images + 1):
                grab(f"{spec.images}/{i}.png")
            if args.max_images:
                print(f"       + images 1..{args.max_images}")
    else:
        print(f"downloading {args.splits} into {root} (~3,200 PNGs, this takes a while)")
        download_rfmid(root, tuple(args.splits))

    print("\nclass balance:")
    for name in args.splits:
        csv = root / SPLITS[name].csv
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        pos = df[ALL_LABELS].sum()
        empty = int((pos == 0).sum())
        print(f"  {name:<5} n={len(df):<5} Disease_Risk={int(pos['Disease_Risk']):<5} "
              f"top: " + ", ".join(f"{k}={int(v)}" for k, v in
                                   pos.drop('Disease_Risk').nlargest(5).items())
              + f"  ({empty} classes with 0 positives)")


if __name__ == "__main__":
    main()
