#!/usr/bin/env python3
"""
pipeline.py — Teachable Machine 앞단 전처리 파이프라인

TM은 특징 추출기가 고정되어 있고 입력이 224x224로 고정입니다. 우리가 손댈 수 있는
가장 큰 레버는 "무엇을 224 안에 담아 보낼지"입니다. 이 스크립트가 그 일을 합니다.

  1. 검은 여백/장비 테두리 자동 제거      ← 224 예산을 배경에 낭비하지 않음
  2. 가장자리 트림                        ← 번인된 환자 정보 텍스트 제거 효과
  3. 중앙 ROI 크롭                        ← 추론 시 ROI와 동일하게 맞춤
  4. CLAHE 대비 보정                      ← 저대비 병변 경계를 살림
  5. 224 정사각 정규화 + 3채널화
  6. dHash 중복 제거                      ← 부풀려진 정확도의 흔한 원인
  7. ★ 원본(환자) 단위 홀드아웃 분할      ← TM의 랜덤 분할 누출을 우회

출력:
  data/prepared/
  ├── train/<class>/…      ← 이것만 TM에 업로드합니다
  ├── holdout/<class>/…    ← TM에 절대 넣지 마세요. 웹앱 '성능 평가' 탭에 넣습니다
  ├── manifest.csv
  └── report.md

사용:
  python3 pipeline.py --config prep.config.yaml
  python3 pipeline.py --config prep.config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML이 필요합니다:  pip install pyyaml")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# ══ 설정 ════════════════════════════════════════════════════════════════

DEFAULTS = {
    "paths": {"raw": "../data/raw", "out": "../data/prepared"},
    "classes": [],
    "split": {"holdout_ratio": 0.20, "seed": 20260728, "group_regex": None},
    "preprocess": {
        "size": 224, "auto_border_crop": True, "border_threshold": 12,
        "edge_trim": 0.02, "center_crop": 0.85, "square_mode": "crop",
        "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid": 8},
        "force_rgb": True, "out_format": "png", "jpg_quality": 95,
    },
    "dedup": {"enabled": True, "hamming_threshold": 3},
    "balance": {"max_per_class": 400, "equalize": False},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif v is not None or k not in out:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    if path is None:
        return DEFAULTS
    if not path.exists():
        sys.exit(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return deep_merge(DEFAULTS, yaml.safe_load(f) or {})


# ══ 이미지 처리 ═════════════════════════════════════════════════════════

def load_image(path: Path) -> np.ndarray | None:
    """16bit / 그레이스케일 / 알파 채널을 모두 8bit BGR로 정규화해 읽습니다."""
    # 한글 경로에서도 안전하도록 numpy 버퍼 경유로 읽습니다
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None

    if img.dtype != np.uint8:
        # 16bit DICOM 변환본 등 — 실제 값 범위로 정규화합니다
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def auto_border_crop(img: np.ndarray, threshold: int) -> np.ndarray:
    """
    검은(또는 거의 검은) 여백을 잘라냅니다.
    X-ray의 콜리메이터 여백, 내시경의 원형 시야 밖 영역, 스캔 여백에 효과적입니다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    # 지나친 크롭 방지 — 원본의 20% 미만으로 줄어들면 신뢰하지 않습니다
    if (y1 - y0) < img.shape[0] * 0.2 or (x1 - x0) < img.shape[1] * 0.2:
        return img
    return img[y0:y1, x0:x1]


def edge_trim(img: np.ndarray, ratio: float) -> np.ndarray:
    """가장자리를 비율만큼 잘라냅니다. 번인 텍스트가 구석에 있는 경우가 많습니다."""
    if ratio <= 0:
        return img
    h, w = img.shape[:2]
    dy, dx = int(h * ratio), int(w * ratio)
    if h - 2 * dy < 16 or w - 2 * dx < 16:
        return img
    return img[dy:h - dy, dx:w - dx]


def center_crop(img: np.ndarray, ratio: float) -> np.ndarray:
    """중앙 정사각 ROI. 추론 시 inferRoi와 같은 값을 써야 합니다."""
    if ratio >= 1.0:
        return img
    h, w = img.shape[:2]
    side = int(min(h, w) * ratio)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def apply_clahe(img: np.ndarray, clip: float, tiles: int) -> np.ndarray:
    """
    LAB 색공간의 L 채널에만 CLAHE를 적용합니다.
    BGR 각 채널에 따로 걸면 색이 틀어지므로 반드시 휘도 채널만 손댑니다.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_square(img: np.ndarray, size: int, mode: str) -> np.ndarray:
    h, w = img.shape[:2]
    if mode == "letterbox":
        scale = size / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    # crop — 짧은 변 기준 중앙 정사각
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = img[y0:y0 + side, x0:x0 + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(sq, (size, size), interpolation=interp)


def preprocess(img: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["auto_border_crop"]:
        img = auto_border_crop(img, cfg["border_threshold"])
    img = edge_trim(img, cfg["edge_trim"])
    img = center_crop(img, cfg["center_crop"])
    if cfg["clahe"]["enabled"]:
        img = apply_clahe(img, cfg["clahe"]["clip_limit"], cfg["clahe"]["tile_grid"])
    return to_square(img, cfg["size"], cfg["square_mode"])


def dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """차분 해시 — 리사이즈·압축·약한 밝기 변화에 강한 근중복 탐지용."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def save_image(img: np.ndarray, path: Path, fmt: str, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


# ══ 수집과 분할 ═════════════════════════════════════════════════════════

@dataclass
class Sample:
    src: Path
    cls: str
    group: str
    hash: int = 0
    split: str = "train"
    out: Path | None = None


@dataclass
class Report:
    per_class: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    duplicates: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)


def group_key(name: str, regex: str | None) -> str:
    """원본(환자) 단위 분할을 위한 그룹 키를 파일명에서 추출합니다."""
    if not regex:
        return name
    m = re.search(regex, name)
    if not m:
        return name
    return m.group(1) if m.groups() else m.group(0)


def discover(raw: Path, classes: list[str], regex: str | None,
             report: Report) -> dict[str, list[Sample]]:
    if not raw.exists():
        sys.exit(f"raw 폴더가 없습니다: {raw.resolve()}\n"
                 f"클래스별 하위 폴더를 만들고 이미지를 넣으세요.")

    found = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if classes:
        for c in classes:
            if c not in found:
                report.warnings.append(f"설정한 클래스 폴더가 없습니다: {c}")
        targets = [c for c in classes if c in found]
    else:
        targets = found

    if not targets:
        sys.exit(f"{raw.resolve()} 아래에 클래스 폴더가 없습니다.")

    out: dict[str, list[Sample]] = {}
    for cls in targets:
        files = [p for p in sorted((raw / cls).rglob("*"))
                 if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        out[cls] = [Sample(src=p, cls=cls, group=group_key(p.stem, regex)) for p in files]
        if not files:
            report.warnings.append(f"클래스 '{cls}'에 이미지가 없습니다.")
    return out


def split_by_group(samples: list[Sample], ratio: float, rng: random.Random) -> None:
    """
    ★ 원본(환자) 단위로 분할합니다.
    이미지 단위로 무작위 분할하면 같은 환자의 다른 촬영본이 train과 holdout에
    흩어져 성능이 과대평가됩니다. 이 함수가 그것을 막습니다.
    """
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_group[s.group].append(s)

    groups = sorted(by_group.keys())
    rng.shuffle(groups)

    target = int(round(len(samples) * ratio))
    holdout_groups: set[str] = set()
    count = 0
    for g in groups:
        if count >= target:
            break
        holdout_groups.add(g)
        count += len(by_group[g])

    # 전부 홀드아웃으로 가는 사고 방지 (그룹이 1개뿐인 경우)
    if len(holdout_groups) == len(groups) and len(groups) > 1:
        holdout_groups.discard(groups[0])

    for s in samples:
        s.split = "holdout" if s.group in holdout_groups else "train"


def dedup(samples: list[Sample], threshold: int, report: Report) -> list[Sample]:
    """
    dHash 기반 근중복 제거.

    ★ 반드시 같은 클래스 안에서만 비교합니다.
      의료영상에서 정상과 이상은 시각적으로 매우 비슷합니다 (같은 부위, 같은
      촬영 프로토콜). 클래스를 넘나들며 dHash를 비교하면 라벨이 다른 정상 소견을
      "이상 소견의 중복"으로 오인해 지워 버립니다. 클래스 하나가 통째로 사라질
      수 있습니다.

    같은 클래스 안에서 제거하는 것으로도 목적은 달성됩니다 — 노리는 것은
    "같은 이미지가 train과 holdout에 동시에 있어 정확도가 부풀려지는 것"이고,
    그런 쌍은 항상 같은 클래스에 속합니다.
    """
    kept: list[Sample] = []
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.cls].append(s)

    for cls, items in by_class.items():
        seen: list[tuple[int, Sample]] = []
        removed = 0
        for s in items:
            dup_of = None
            for h, other in seen:
                if hamming(h, s.hash) <= threshold:
                    dup_of = other
                    break
            if dup_of is not None:
                report.duplicates.append((str(s.src), str(dup_of.src)))
                removed += 1
                continue
            seen.append((s.hash, s))
            kept.append(s)

        # 과도한 제거는 임계값이 데이터에 맞지 않다는 신호입니다
        if items and removed / len(items) > 0.35:
            report.warnings.append(
                f"클래스 '{cls}': 중복 제거로 {removed}/{len(items)}장"
                f"({removed / len(items) * 100:.0f}%)이 삭제되었습니다. "
                f"dedup.hamming_threshold({threshold})가 이 데이터에 너무 공격적입니다. "
                f"구조가 균일한 의료영상은 서로 다른 개체끼리도 dHash가 비슷하므로 "
                f"0~1로 낮추거나 dedup.enabled: false 로 끄고 확인하세요."
            )
    return kept


# ══ 리포트 ══════════════════════════════════════════════════════════════

def write_report(out: Path, cfg: dict, report: Report, samples: list[Sample]) -> None:
    lines: list[str] = []
    A = lines.append

    A("# 전처리 리포트")
    A("")
    A(f"- 생성 시각: `{__import__('datetime').datetime.now().isoformat(timespec='seconds')}`")
    A(f"- 출력 경로: `{out.resolve()}`")
    A(f"- 입력 크기: {cfg['preprocess']['size']}×{cfg['preprocess']['size']}")
    A(f"- 중앙 ROI: `center_crop = {cfg['preprocess']['center_crop']}`  "
      f"← **web/project.config.js의 `decision.inferRoi`를 같은 값으로 맞추세요**")
    A(f"- CLAHE: {'사용' if cfg['preprocess']['clahe']['enabled'] else '미사용'}")
    A(f"- 그룹 정규식: `{cfg['split']['group_regex']}`")
    A("")

    A("## 클래스별 장수")
    A("")
    A("| 클래스 | train | holdout | 합계 | 원본 그룹 |")
    A("|---|---:|---:|---:|---:|")
    total_tr = total_ho = 0
    for cls in sorted(report.per_class):
        tr = report.per_class[cls]["train"]
        ho = report.per_class[cls]["holdout"]
        total_tr += tr
        total_ho += ho
        A(f"| `{cls}` | {tr} | {ho} | {tr + ho} | {report.groups.get(cls, 0)} |")
    A(f"| **합계** | **{total_tr}** | **{total_ho}** | **{total_tr + total_ho}** | |")
    A("")

    # 불균형 경고
    counts = {c: report.per_class[c]["train"] for c in report.per_class}
    if counts:
        lo, hi = min(counts.values()), max(counts.values())
        if lo > 0 and hi / lo > 2.0:
            A(f"> ⚠️ **클래스 불균형** — 최다 {hi}장 / 최소 {lo}장 (비율 {hi/lo:.1f}배). "
              f"TM은 불균형을 보정하지 않습니다. `balance.equalize: true`를 켜거나 "
              f"적은 클래스의 증강 배수를 늘리세요.")
            A("")

    if any(report.per_class[c]["train"] < 40 for c in report.per_class):
        A("> ⚠️ **표본 부족** — 학습 장수가 40장 미만인 클래스가 있습니다. "
          "`augment.py`로 증강하되, 증강은 원본의 다양성을 늘리지 못한다는 점을 기억하세요.")
        A("")

    if report.duplicates:
        A(f"## 중복 제거: {len(report.duplicates)}건")
        A("")
        A("같은 이미지가 train과 holdout에 동시에 있으면 정확도가 부풀려집니다. "
          "제거된 목록은 `manifest.csv`의 `duplicate` 행에서 확인하세요.")
        A("")
        for a, b in report.duplicates[:15]:
            A(f"- `{Path(a).name}` ≈ `{Path(b).name}`")
        if len(report.duplicates) > 15:
            A(f"- … 외 {len(report.duplicates) - 15}건")
        A("")

    if report.unreadable:
        A(f"## 읽기 실패: {len(report.unreadable)}건")
        A("")
        for p in report.unreadable[:15]:
            A(f"- `{p}`")
        A("")

    if report.warnings:
        A("## 경고")
        A("")
        for w in report.warnings:
            A(f"- {w}")
        A("")

    A("## 다음 단계")
    A("")
    A("```bash")
    A("# 1) 학습 데이터 증강 (holdout은 절대 증강하지 않습니다)")
    A("python3 augment.py --config prep.config.yaml --factor 4 --webcam-degrade")
    A("")
    A("# 2) 무효/기타 클래스 생성 — softmax가 억지 답을 내는 문제를 데이터로 막습니다")
    A("python3 negatives.py --config prep.config.yaml --synthetic 150 --degrade 150")
    A("```")
    A("")
    A("3) `data/prepared/train/` 의 **클래스별 폴더를 Teachable Machine에 업로드**합니다.")
    A("4) `data/prepared/holdout/` 은 **TM에 넣지 말고**, 웹앱 `성능 평가` 탭에 드래그합니다.")
    A("")
    A("> **번인 텍스트 확인**: 의료영상에는 환자 정보가 픽셀에 새겨진 경우가 있습니다. "
      "`edge_trim`이 구석은 잘라내지만 완전하지 않습니다. 업로드 전에 몇 장을 직접 열어 "
      "확인하세요. 웹앱의 `근거 분석`(오클루전 히트맵)으로 모델이 텍스트를 보고 있는지 검증할 수 있습니다.")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(out: Path, samples: list[Sample], report: Report) -> None:
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "class", "split", "group", "source", "output", "dhash"])
        for s in samples:
            w.writerow(["ok", s.cls, s.split, s.group, str(s.src),
                        str(s.out) if s.out else "", f"{s.hash:016x}"])
        for src, dup_of in report.duplicates:
            w.writerow(["duplicate", "", "", "", src, dup_of, ""])
        for p in report.unreadable:
            w.writerow(["unreadable", "", "", "", p, "", ""])


# ══ 메인 ════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="TM 앞단 전처리 파이프라인")
    ap.add_argument("--config", type=Path, default=Path("prep.config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 계획만 출력")
    ap.add_argument("--clean", action="store_true", help="출력 폴더를 먼저 비웁니다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw"])
    out = Path(cfg["paths"]["out"])
    pcfg = cfg["preprocess"]
    rng = random.Random(cfg["split"]["seed"])
    report = Report()

    print(f"▸ raw   : {raw.resolve()}")
    print(f"▸ out   : {out.resolve()}")

    by_class = discover(raw, cfg["classes"], cfg["split"]["group_regex"], report)
    for cls, items in by_class.items():
        groups = len({s.group for s in items})
        report.groups[cls] = groups
        print(f"  · {cls:<14} {len(items):>5}장  (원본 그룹 {groups}개)")
        if groups == len(items) and len(items) > 20 and not cfg["split"]["group_regex"]:
            report.warnings.append(
                f"클래스 '{cls}': 그룹 정규식이 없어 파일 하나를 하나의 원본으로 취급합니다. "
                f"환자당 여러 장이 있는 데이터셋이면 split.group_regex를 지정하세요 (데이터 누출 위험)."
            )

    # 균형 조정 — 원본 그룹 단위로 잘라 누출을 막습니다
    max_per = cfg["balance"]["max_per_class"]
    if cfg["balance"]["equalize"]:
        smallest = min(len(v) for v in by_class.values() if v)
        max_per = min(max_per, smallest) if max_per else smallest
    if max_per:
        for cls, items in by_class.items():
            if len(items) <= max_per:
                continue
            groups = sorted({s.group for s in items})
            rng.shuffle(groups)
            chosen: set[str] = set()
            n = 0
            for g in groups:
                if n >= max_per:
                    break
                chosen.add(g)
                n += sum(1 for s in items if s.group == g)
            before = len(items)
            by_class[cls] = [s for s in items if s.group in chosen]
            print(f"  · {cls:<14} {before} → {len(by_class[cls])}장으로 축소 (max_per_class)")

    # 분할
    for cls, items in by_class.items():
        split_by_group(items, cfg["split"]["holdout_ratio"], rng)

    if args.dry_run:
        print("\n[dry-run] 다음과 같이 처리됩니다:")
        for cls, items in by_class.items():
            tr = sum(1 for s in items if s.split == "train")
            print(f"  {cls:<14} train {tr:>5} / holdout {len(items) - tr:>5}")
        return

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 전처리 + 해시
    processed: list[Sample] = []
    cache: dict[Path, np.ndarray] = {}
    total = sum(len(v) for v in by_class.values())
    done = 0

    print("\n▸ 전처리 중…")
    for cls, items in by_class.items():
        for s in items:
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total}", end="\r", flush=True)
            img = load_image(s.src)
            if img is None:
                report.unreadable.append(str(s.src))
                continue
            try:
                proc = preprocess(img, pcfg)
            except Exception as e:                      # noqa: BLE001
                report.unreadable.append(f"{s.src} ({e})")
                continue
            s.hash = dhash(proc)
            cache[s.src] = proc
            processed.append(s)
    print()

    # 중복 제거 — 전체 풀에서 한 번에 (train/holdout 교차 중복까지 잡습니다)
    if cfg["dedup"]["enabled"]:
        before = len(processed)
        processed = dedup(processed, cfg["dedup"]["hamming_threshold"], report)
        if before != len(processed):
            print(f"▸ 중복 제거: {before - len(processed)}장")

    # 저장
    print("▸ 저장 중…")
    ext = "jpg" if pcfg["out_format"] == "jpg" else "png"
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for s in processed:
        counters[(s.split, s.cls)] += 1
        idx = counters[(s.split, s.cls)]
        s.out = out / s.split / s.cls / f"{s.cls}_{idx:05d}.{ext}"
        save_image(cache[s.src], s.out, ext, pcfg["jpg_quality"])
        report.per_class[s.cls][s.split] += 1

    # 클래스가 통째로 사라졌는지 확인 — 조용히 넘어가면 학습 자체가 잘못됩니다
    for cls in by_class:
        tr = report.per_class[cls]["train"]
        ho = report.per_class[cls]["holdout"]
        if tr + ho == 0:
            report.warnings.append(
                f"클래스 '{cls}'에 남은 이미지가 없습니다. 중복 제거 임계값이나 "
                f"원본 데이터를 확인하세요."
            )
        elif tr == 0:
            report.warnings.append(f"클래스 '{cls}'의 학습 데이터가 0장입니다.")
        elif ho == 0:
            report.warnings.append(
                f"클래스 '{cls}'의 홀드아웃이 0장입니다. 원본 그룹이 너무 적어 "
                f"평가할 수 없습니다."
            )

    write_manifest(out, processed, report)
    write_report(out, cfg, report, processed)

    print("\n✓ 완료")
    for cls in sorted(by_class):
        tr = report.per_class[cls]["train"]
        ho = report.per_class[cls]["holdout"]
        flag = '  ⚠️ 비어 있음' if tr + ho == 0 else ''
        print(f"  {cls:<14} train {tr:>5} / holdout {ho:>5}{flag}")

    if report.warnings:
        print(f"\n  ⚠️ 경고 {len(report.warnings)}건 — report.md 를 확인하세요")
        for w in report.warnings[:3]:
            print(f"     · {w[:110]}")
    print(f"\n  리포트: {(out / 'report.md').resolve()}")
    print(f"  ★ train/ 만 TM에 업로드하세요. holdout/ 은 웹앱 평가 탭에 넣습니다.")


if __name__ == "__main__":
    main()
