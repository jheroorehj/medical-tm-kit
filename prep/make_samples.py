#!/usr/bin/env python3
"""
make_samples.py — 내장 샘플 이미지 생성 (데모 안정성 확보)

왜 필요한가:
  강의 10-3 "프로젝트 실패 3대 요인" 1위가 **작동 안 하는 데모**입니다.
  같은 절에서 대비책으로 요구한 것이
  "이미지 업로드가 실패하면 내장 샘플 이미지가 자동으로 올라가게" 입니다.

  이 스크립트는 홀드아웃(또는 지정한 폴더)에서 클래스별로 이미지를 골라
  web/samples.js 에 **base64 data URI 로 내장**합니다. 파일 경로를 쓰지 않으므로
  Vercel 배포 후에도, 인터넷이 끊겨도, 발표장 노트북이 바뀌어도 동작합니다.

  덤으로 판독 화면에 "샘플로 바로 시험" 버튼이 생겨서
  발표 중 파일 탐색기를 열 필요가 없습니다.

사용:
  python3 make_samples.py                     # 홀드아웃에서 클래스별 1장
  python3 make_samples.py --per-class 2       # 클래스별 2장
  python3 make_samples.py --from ../data/prepared/train --size 320
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2

try:
    import yaml
except ImportError:
    sys.exit("PyYAML이 필요합니다:  pip install pyyaml")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="내장 샘플 생성 (data URI)")
    ap.add_argument("--config", type=Path, default=Path("prep.config.yaml"))
    ap.add_argument("--from", dest="src", type=Path, default=None,
                    help="샘플을 고를 폴더 (기본: prepared/holdout, 없으면 train)")
    ap.add_argument("--out", type=Path, default=Path("../web/samples.js"))
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--size", type=int, default=320, help="내장 이미지 한 변 크기(px)")
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    cfg = {}
    if args.config.exists():
        with args.config.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    out_root = Path(cfg.get("paths", {}).get("out", "../data/prepared"))

    src = args.src
    if src is None:
        for cand in (out_root / "holdout", out_root / "train"):
            if cand.is_dir():
                src = cand
                break
    if src is None or not src.is_dir():
        sys.exit(f"샘플을 고를 폴더가 없습니다. --from 으로 지정하세요.\n"
                 f"  (찾아본 곳: {(out_root / 'holdout').resolve()})")

    print(f"▸ 원본: {src.resolve()}")

    samples = []
    total_bytes = 0
    for cls_dir in sorted(d for d in src.iterdir() if d.is_dir()):
        files = sorted(p for p in cls_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXT
                       and not p.name.startswith(("aug_", "recap_")))
        if not files:
            print(f"  {cls_dir.name:<16} 이미지 없음 — 건너뜁니다")
            continue

        # 앞쪽에서 균등하게 뽑습니다 (재현 가능하도록 무작위를 쓰지 않습니다)
        step = max(1, len(files) // args.per_class)
        picked = files[::step][: args.per_class]

        for i, path in enumerate(picked):
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            side = min(h, w)
            sq = img[(h - side) // 2:(h - side) // 2 + side,
                     (w - side) // 2:(w - side) // 2 + side]
            sq = cv2.resize(sq, (args.size, args.size), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", sq, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue
            b64 = base64.b64encode(enc.tobytes()).decode("ascii")
            label = cls_dir.name if args.per_class == 1 else f"{cls_dir.name} {i + 1}"
            samples.append({"label": label, "classId": cls_dir.name,
                            "src": f"data:image/jpeg;base64,{b64}"})
            total_bytes += len(b64)
            print(f"  {cls_dir.name:<16} {path.name}  ({len(b64) // 1024} KB)")

    if not samples:
        sys.exit("샘플을 만들지 못했습니다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(samples, ensure_ascii=False, indent=2)
    args.out.write_text(
        "/* samples.js — prep/make_samples.py 가 생성했습니다. 직접 수정하지 마세요.\n"
        " *\n"
        " * 내장 샘플 이미지입니다. 업로드가 실패하면 자동으로 이 중 첫 번째가\n"
        " * 올라가고, 판독 화면에 '샘플로 바로 시험' 버튼이 생깁니다.\n"
        " * data URI 로 내장되어 있어 배포 환경·오프라인에서도 동작합니다.\n"
        " *\n"
        f" * 생성 개수: {len(samples)}장\n"
        " */\n"
        f"window.MTK_SAMPLES = {body};\n",
        encoding="utf-8")

    print(f"\n✓ {args.out.resolve()}")
    print(f"  {len(samples)}장 · 약 {total_bytes // 1024} KB")
    print()
    print("  index.html 이 samples.js 를 자동으로 불러옵니다 — 추가 설정은 없습니다.")
    print("  ⚠️ 샘플이 너무 크면 첫 로딩이 느려집니다. 총 500KB 이하를 권합니다.")
    print("     (--size 240 또는 --quality 70 으로 줄일 수 있습니다)")


if __name__ == "__main__":
    main()
