#!/usr/bin/env python3
"""
make_testdata.py — 스모크 테스트용 합성 데이터 생성

실제 데이터가 도착하기 전에 파이프라인 전체가 동작하는지 확인하기 위한 것입니다.
의미 있는 의료 데이터가 아니라, 아래 성질을 일부러 갖도록 만든 더미입니다.

  · 환자당 여러 장 (group_regex 로 원본 단위 분할이 되는지 확인)
  · 클래스별 장수 불균형 (불균형 경고가 뜨는지 확인)
  · 완전 중복 파일 1장씩 (dedup 이 작동하는지 확인)
  · 구석에 번인된 환자 ID 텍스트 (edge_trim 이 지우는지 확인)
  · 검은 여백 (auto_border_crop 이 잘라내는지 확인)

사용:
  python3 make_testdata.py
  python3 pipeline.py --clean
  python3 augment.py --factor 3 --webcam-degrade
  python3 negatives.py --synthetic 60 --degrade 60

실제 데이터를 넣기 전에 data/raw 를 비우세요.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

# 클래스명 → 환자 수. web/project.config.js 의 기본 클래스와 맞춰 두었습니다.
SPECS = {"normal": 9, "suspect": 5, "abnormal": 7}


def main() -> None:
    ap = argparse.ArgumentParser(description="스모크 테스트용 합성 데이터 생성")
    ap.add_argument("--out", type=Path, default=Path("../data/raw"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = args.out

    for cls, n_patients in SPECS.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)

        for p in range(n_patients):
            # 검은 여백이 있는 세로 영상 — auto_border_crop 테스트용
            base = np.zeros((520, 400, 3), np.uint8)
            tone = rng.randrange(60, 200)
            cv2.circle(base, (200, 260), 150, (tone, tone, tone), -1)

            # 같은 환자의 여러 촬영본 — 원본 단위 분할 테스트용
            for v in range(rng.randint(2, 3)):
                img = base.copy()
                # 구석 번인 텍스트 — edge_trim 테스트용
                cv2.putText(img, f"PT{p:03d}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                lesion = rng.randrange(90, 240)
                cv2.circle(img, (180 + v * 12, 240), 28, (lesion, lesion, lesion), -1)
                cv2.imwrite(str(d / f"patient{p:03d}_view{v}.png"), img)

        # 완전 중복 1장 — dedup 테스트용
        first = sorted(d.glob("patient*.png"))[0]
        cv2.imwrite(str(d / "dup_copy.png"), cv2.imread(str(first)))

    counts = {c: len(list((root / c).glob('*.png'))) for c in SPECS}
    print(f"▸ 생성 완료: {root.resolve()}")
    for c, n in counts.items():
        print(f"    {c:<12} {n:>3}장 (환자 {SPECS[c]}명 + 중복 1장)")
    print()
    print("  다음:  python3 pipeline.py --clean")
    print()
    print("  ⚠️ 이 데이터는 원이 그려진 더미입니다. dHash 가 거의 동일하므로")
    print("     dedup 이 과다 제거 경고를 냅니다 — 그게 정상 동작 확인입니다.")
    print("     실제 데이터를 넣을 때는 data/raw 를 먼저 비우세요.")


if __name__ == "__main__":
    main()
