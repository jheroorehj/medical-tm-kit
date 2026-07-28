#!/usr/bin/env python3
"""
from_csv.py — CSV 라벨을 클래스 폴더로 정리 (+ 클래스 병합)

왜 필요한가:
  공개 의료 데이터셋은 폴더가 아니라 CSV로 라벨이 주어지는 경우가 많습니다.
    · HAM10000  — metadata CSV 의 dx 컬럼이 클래스
    · APTOS 2019 — train.csv 의 diagnosis 컬럼이 0~4 등급
  Teachable Machine 은 폴더째로 드래그해 넣는 게 가장 편하므로, CSV를 보고
  이미지를 클래스 폴더로 나눠 주는 작업이 먼저 필요합니다.

  강의 10-2에서 "Claude Code에게 시키면 되는 실전 팁"으로 소개된 작업을
  재현 가능한 스크립트로 고정한 것입니다. 매번 다시 짜지 않아도 됩니다.

같이 해 주는 일:
  · ★ 클래스 병합 — 5등급을 2클래스로 줄이는 문제 재정의 (강의 7-2 전략)
  · ★ 언더샘플링 — 병합 후 적은 쪽 숫자에 맞춰 균형 (강의 7-2)
  · 라벨 신뢰도 필터 — 예: HAM10000 에서 histo 확진 샘플만 사용 (강의 6-2)
  · 클래스별 개수 리포트 (강의 3-3: "데이터 받으면 가장 먼저 할 일 = 개수 세기")

사용 예 — APTOS 2019 (0~4 등급 → 2클래스):
  python3 from_csv.py \\
    --csv ../data/aptos/train.csv \\
    --images ../data/aptos/train_images \\
    --id-col id_code --label-col diagnosis \\
    --ext .png \\
    --map "0,1=normal" --map "2,3,4=abnormal" \\
    --balance

사용 예 — HAM10000 (7클래스 중 상위 2개만, histo 확진만):
  python3 from_csv.py \\
    --csv ../data/ham/HAM10000_metadata.csv \\
    --images ../data/ham/images \\
    --id-col image_id --label-col dx --ext .jpg \\
    --map "nv=normal" --map "mel=abnormal" \\
    --filter "dx_type=histo" \\
    --group-col lesion_id \\
    --balance

  ★ --group-col 을 주면 같은 병변(환자)의 여러 사진이 흩어지지 않도록
    그룹 키를 파일명 앞에 붙입니다. 그러면 pipeline.py 의 group_regex 로
    환자 단위 분할이 가능해집니다 (강의 6차시 데이터 누수 방지).

출력: <out>/<클래스명>/<그룹키>_<원본이름>.<ext>
그다음:  python3 pipeline.py --clean
"""

from __future__ import annotations

import argparse
import csv as csvmod
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_map(specs: list[str]) -> dict[str, str]:
    """'0,1=normal' → {'0': 'normal', '1': 'normal'}"""
    out: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            sys.exit(f"--map 형식이 잘못되었습니다: {spec}\n  예: --map \"0,1=normal\"")
        left, cls = spec.split("=", 1)
        cls = cls.strip()
        if not cls:
            sys.exit(f"--map 의 클래스명이 비었습니다: {spec}")
        for v in left.split(","):
            v = v.strip()
            if v:
                out[v] = cls
    return out


def parse_filters(specs: list[str]) -> list[tuple[str, set[str]]]:
    """'dx_type=histo,consensus' → ('dx_type', {'histo','consensus'})"""
    out = []
    for spec in specs:
        if "=" not in spec:
            sys.exit(f"--filter 형식이 잘못되었습니다: {spec}\n  예: --filter \"dx_type=histo\"")
        col, vals = spec.split("=", 1)
        out.append((col.strip(), {v.strip() for v in vals.split(",") if v.strip()}))
    return out


def find_image(images_dir: Path, stem: str, ext: str | None) -> Path | None:
    """CSV의 id 값으로 실제 파일을 찾습니다. 확장자가 CSV에 포함된 경우도 처리합니다."""
    # id 값에 이미 확장자가 붙어 있는 경우
    direct = images_dir / stem
    if direct.is_file():
        return direct
    if ext:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    for e in IMAGE_EXT:
        p = images_dir / f"{stem}{e}"
        if p.is_file():
            return p
    return None


def safe(s: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(s))


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV 라벨 → 클래스 폴더 정리")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="이미지가 들어 있는 폴더")
    ap.add_argument("--out", type=Path, default=Path("../data/raw"))
    ap.add_argument("--id-col", required=True, help="파일명(또는 파일명 stem)이 담긴 컬럼")
    ap.add_argument("--label-col", required=True, help="클래스 라벨 컬럼")
    ap.add_argument("--ext", default=None, help="이미지 확장자 (예: .png). 생략하면 자동 탐색")
    ap.add_argument("--map", action="append", default=[], metavar="VALUES=CLASS",
                    help='라벨값 → 클래스 매핑. 병합 가능. 예: --map "0,1=normal"')
    ap.add_argument("--filter", action="append", default=[], metavar="COL=VALUES",
                    help='해당 컬럼이 이 값들일 때만 사용. 예: --filter "dx_type=histo"')
    ap.add_argument("--group-col", default=None,
                    help="환자/병변 식별 컬럼. 파일명 앞에 붙여 환자 단위 분할을 가능하게 합니다")
    ap.add_argument("--balance", action="store_true",
                    help="가장 적은 클래스 수에 맞춰 언더샘플링 (강의 7-2 전략)")
    ap.add_argument("--limit", type=int, default=0, help="클래스당 최대 장수 (0=제한 없음)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--move", action="store_true", help="복사 대신 이동 (디스크 절약)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true", help="출력 폴더를 먼저 비웁니다")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV를 찾을 수 없습니다: {args.csv.resolve()}")
    if not args.images.is_dir():
        sys.exit(f"이미지 폴더를 찾을 수 없습니다: {args.images.resolve()}")

    mapping = parse_map(args.map)
    filters = parse_filters(args.filter)
    rng = random.Random(args.seed)

    # ── CSV 읽기 ──
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csvmod.DictReader(f))
    if not rows:
        sys.exit("CSV에 데이터가 없습니다.")

    cols = list(rows[0].keys())
    for need in [args.id_col, args.label_col] + [c for c, _ in filters] \
            + ([args.group_col] if args.group_col else []):
        if need not in cols:
            sys.exit(f"컬럼 '{need}' 이 CSV에 없습니다.\n  사용 가능한 컬럼: {', '.join(cols)}")

    print(f"▸ CSV   : {args.csv.resolve()}  ({len(rows)}행)")
    print(f"▸ 이미지: {args.images.resolve()}")
    print(f"▸ 출력  : {args.out.resolve()}")

    # ── 원본 라벨 분포 (강의 3-3: 받으면 먼저 센다) ──
    raw_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        raw_counts[str(r[args.label_col]).strip()] += 1
    print("\n▸ 원본 라벨 분포")
    for k in sorted(raw_counts, key=lambda x: -raw_counts[x]):
        share = raw_counts[k] / len(rows) * 100
        print(f"    {k:<24} {raw_counts[k]:>6}장  ({share:5.1f}%)")

    if not mapping:
        print("\n⚠️ --map 을 주지 않았습니다. 위 라벨값을 보고 매핑을 지정하세요.")
        print('   예:  --map "0,1=normal" --map "2,3,4=abnormal"')
        return

    imbalance = max(raw_counts.values()) / max(1, min(raw_counts.values()))
    if imbalance > 10:
        print(f"\n⚠️ 클래스 불균형이 {imbalance:.0f}배입니다. --balance 로 언더샘플링을 권합니다.")

    # ── 선별 ──
    buckets: dict[str, list[tuple[Path, str]]] = defaultdict(list)   # cls → [(src, group)]
    skipped_filter = skipped_map = missing = 0

    for r in rows:
        if any(str(r[c]).strip() not in vals for c, vals in filters):
            skipped_filter += 1
            continue
        label = str(r[args.label_col]).strip()
        cls = mapping.get(label)
        if cls is None:
            skipped_map += 1
            continue
        src = find_image(args.images, str(r[args.id_col]).strip(), args.ext)
        if src is None:
            missing += 1
            continue
        group = safe(r[args.group_col]) if args.group_col else safe(src.stem)
        buckets[cls].append((src, group))

    if not buckets:
        sys.exit("조건에 맞는 이미지가 없습니다. --map / --filter / --ext 를 확인하세요.")

    print("\n▸ 매핑 결과")
    for cls in sorted(buckets):
        print(f"    {cls:<24} {len(buckets[cls]):>6}장")
    if skipped_filter:
        print(f"    (필터 제외 {skipped_filter}행)")
    if skipped_map:
        print(f"    (매핑 없음 {skipped_map}행 — --map 에 포함되지 않은 라벨)")
    if missing:
        print(f"    ⚠️ 이미지 파일을 못 찾음 {missing}행 — --ext 를 확인하세요")

    # ── 언더샘플링 / 상한 ──
    # 그룹 단위로 잘라 같은 환자가 쪼개지지 않게 합니다
    target = min(len(v) for v in buckets.values()) if args.balance else None
    if args.limit:
        target = min(target, args.limit) if target else args.limit

    if target:
        for cls, items in buckets.items():
            if len(items) <= target:
                continue
            by_group: dict[str, list] = defaultdict(list)
            for it in items:
                by_group[it[1]].append(it)
            groups = sorted(by_group)
            rng.shuffle(groups)
            picked, n = [], 0
            for g in groups:
                if n >= target:
                    break
                picked += by_group[g]
                n += len(by_group[g])
            print(f"    {cls:<24} {len(items)} → {len(picked)}장으로 축소")
            buckets[cls] = picked

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았습니다.")
        return

    # ── 복사 ──
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    print("\n▸ 복사 중…")
    total = 0
    for cls, items in buckets.items():
        dst_dir = args.out / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src, group in items:
            # 그룹 키를 앞에 붙여 pipeline.py 의 group_regex 가 잡을 수 있게 합니다
            dst = dst_dir / f"{group}__{safe(src.name)}"
            if dst.exists():
                continue
            if args.move:
                shutil.move(str(src), dst)
            else:
                shutil.copy2(src, dst)
            total += 1
        print(f"    {cls:<24} {len(items):>6}장")

    print(f"\n✓ 완료 — {total}장을 {args.out.resolve()} 에 정리했습니다")
    print()
    print("  다음 단계:")
    print("   1) prep.config.yaml 의 split.group_regex 를 다음으로 설정하세요:")
    print("        group_regex: '^([^_]+)__'      # 파일명 앞의 그룹 키를 잡습니다")
    print("   2) python3 pipeline.py --clean")
    print()
    print("  ★ project.config.js 의 dataset 블록에 출처·라이선스·라벨링 방식을")
    print("    적어 두세요. 대회 필수 표기 항목입니다.")


if __name__ == "__main__":
    main()
