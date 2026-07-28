#!/usr/bin/env python3
"""
fetch_model.py — Teachable Machine 모델 파일을 내려받아 함께 배포하기

왜 필요한가:
  강의 9-3: "배포 시에는 모델 URL을 매번 입력하는 방식보다,
             모델 파일을 함께 배포하는 방식이 안정적"

  발표장에서 TM 클라우드가 느리거나 네트워크가 불안하면 모델 로드에 실패합니다.
  모델 파일을 web/model/ 에 두면 우리 사이트에서 함께 서빙되므로
  외부 의존이 하나 줄어듭니다. (실패 3대 요인 1위 = 작동 안 하는 데모)

사용:
  python3 fetch_model.py https://teachablemachine.withgoogle.com/models/aX9k2Lp7/

그다음 웹앱 설정 탭의 주분류기 URL 칸에 다음을 넣으세요:
  ./model/

또는 project.config.js 의 models.primary.url 을 './model/' 로 고정하세요.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def fetch(url: str, dst: Path) -> int:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"내려받기 실패 ({e.code}): {url}\n"
                 f"  TM에서 'Upload my model' 을 눌렀는지 확인하세요.")
    except Exception as e:                                    # noqa: BLE001
        sys.exit(f"내려받기 실패: {url}\n  {e}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="TM 모델 파일 내려받기")
    ap.add_argument("url", help="TM 모델 URL (끝의 / 포함)")
    ap.add_argument("--out", type=Path, default=Path("../web/model"))
    args = ap.parse_args()

    base = args.url.strip()
    if base.endswith("model.json"):
        base = base[: -len("model.json")]
    if not base.endswith("/"):
        base += "/"

    print(f"▸ 모델 URL: {base}")
    print(f"▸ 저장 위치: {args.out.resolve()}\n")

    # 1) model.json — weights 파일 목록이 이 안에 들어 있습니다
    n = fetch(base + "model.json", args.out / "model.json")
    print(f"  model.json      {n // 1024:>5} KB")

    n = fetch(base + "metadata.json", args.out / "metadata.json")
    print(f"  metadata.json   {n // 1024:>5} KB")

    meta = json.loads((args.out / "metadata.json").read_text(encoding="utf-8"))
    labels = meta.get("labels", [])

    model = json.loads((args.out / "model.json").read_text(encoding="utf-8"))
    paths: list[str] = []
    for group in model.get("weightsManifest", []):
        paths += group.get("paths", [])
    if not paths:
        paths = ["weights.bin"]                # 구버전 폴백

    total = 0
    for p in paths:
        n = fetch(base + p, args.out / p)
        total += n
        print(f"  {p:<15} {n // 1024:>5} KB")

    print(f"\n✓ 완료 — 클래스 {len(labels)}개: {', '.join(labels)}")
    print(f"  총 {(total) // 1024} KB")
    print()
    print("  웹앱 설정 탭의 주분류기 URL 칸에 다음을 넣으세요:")
    print("      ./model/")
    print()
    print("  또는 project.config.js 에 고정:")
    print("      primary: { url: './model/', label: '주분류기' },")
    print()
    print("  ★ 클래스 이름이 project.config.js 의 id 와 일치하는지 확인하세요:")
    for l in labels:
        print(f"      - {l}")


if __name__ == "__main__":
    main()
