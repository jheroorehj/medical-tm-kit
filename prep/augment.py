#!/usr/bin/env python3
"""
augment.py — 데이터 증강 (Teachable Machine은 증강 기능이 없습니다)

TM에는 데이터 증강이 아예 없습니다. 회전·밝기·노이즈를 전부 외부에서 만들어
올려야 합니다. 이 스크립트가 그 일을 합니다.

두 계열의 증강을 제공합니다.

  ① 일반 증강 — 회전, 스케일, 밝기/대비, 감마, 노이즈, 약한 블러
     흔한 것이고 대부분의 팀이 여기까지 합니다.

  ② ★ 웹캠 열화 증강 (--webcam-degrade)
     추론이 웹캠으로 이뤄질 거라면, 학습 데이터에 웹캠 열화를 미리 입힙니다.
     JPEG 압축 아티팩트, 모션 블러, 조명 그라디언트, 반사 하이라이트,
     원근 왜곡, 모아레(화면 촬영), 색온도 시프트, 비네팅.

     대부분의 팀이 놓치는 부분입니다. 공개 데이터셋으로 학습한 모델을 웹캠에
     들이대면 정확도가 무너지는 이유가 바로 이 도메인 갭입니다.
     학습 도메인을 추론 도메인 쪽으로 미리 끌어오면 실사용 정확도가 올라갑니다.

     더 강한 수는 recapture.py — 실제 모니터에 띄워 실제 웹캠으로 되찍습니다.

주의:
  · holdout/ 은 절대 증강하지 않습니다. 증강본으로 평가하면 의미가 없습니다.
  · 좌우 반전(--flip)은 해부학적으로 허용될 때만 켜세요.
    심장 위치, 좌우 장기 구분이 의미 있는 영상에서는 반전이 라벨을 망칩니다.

사용:
  python3 augment.py --config prep.config.yaml --factor 4 --webcam-degrade
  python3 augment.py --config prep.config.yaml --factor 6 --webcam-degrade --only abnormal
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML이 필요합니다:  pip install pyyaml")

AUG_PREFIX = "aug"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ══ 일반 증강 ═══════════════════════════════════════════════════════════

def rotate_zoom(img: np.ndarray, deg: float, zoom: float = 1.0) -> np.ndarray:
    """
    회전 + 확대를 한 번의 아핀 변환으로 처리합니다.
    회전만 하면 네 귀퉁이에 검은 삼각형이 생기고, 모델이 그 검은 영역을
    학습 신호로 오인합니다. 살짝 확대해 귀퉁이를 채웁니다.
    """
    h, w = img.shape[:2]
    # 회전으로 생기는 빈 영역을 덮을 최소 배율
    rad = abs(np.deg2rad(deg))
    cover = abs(np.cos(rad)) + abs(np.sin(rad))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, max(zoom, cover))
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx * w], [0, 1, dy * h]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def gaussian_noise(img: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    np_rng = np.random.default_rng(rng.randrange(1 << 30))
    noise = np_rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def soft_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


# ══ 웹캠 열화 증강 ══════════════════════════════════════════════════════

def jpeg_artifact(img: np.ndarray, quality: int) -> np.ndarray:
    """웹캠 스트림과 캔버스 캡처는 JPEG 압축을 거칩니다. 그 아티팩트를 학습시킵니다."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    """손떨림. 방향성 커널로 구현합니다."""
    length = max(3, length | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (length, length))
    s = kernel.sum()
    if s <= 0:
        return img
    return cv2.filter2D(img, -1, kernel / s)


def illumination_gradient(img: np.ndarray, strength: float, angle: float) -> np.ndarray:
    """
    실내 조명은 균일하지 않습니다. 한쪽이 밝고 한쪽이 어두운 그라디언트를 씌웁니다.
    이걸 학습시켜야 조명이 치우친 실제 촬영에서 무너지지 않습니다.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx / w) - 0.5
    yy = (yy / h) - 0.5
    rad = np.deg2rad(angle)
    proj = xx * np.cos(rad) + yy * np.sin(rad)
    field = 1.0 + strength * proj * 2.0
    field = np.clip(field, 0.25, 2.0)[:, :, None]
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


def specular_highlight(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """조명 반사 하이라이트. 화면을 촬영할 때 거의 항상 생깁니다."""
    h, w = img.shape[:2]
    cx = rng.uniform(0.15, 0.85) * w
    cy = rng.uniform(0.15, 0.85) * h
    radius = rng.uniform(0.12, 0.35) * min(h, w)
    strength = rng.uniform(0.15, 0.45)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (radius ** 2)
    blob = np.exp(-d2) * strength
    return np.clip(img.astype(np.float32) + blob[:, :, None] * 255, 0, 255).astype(np.uint8)


def perspective_warp(img: np.ndarray, amount: float, rng: random.Random) -> np.ndarray:
    """웹캠을 정면에서 완벽히 맞추는 사람은 없습니다. 약한 원근 왜곡을 넣습니다."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda: rng.uniform(-amount, amount)          # noqa: E731
    dst = np.float32([
        [w * j(), h * j()],
        [w * (1 + j()), h * j()],
        [w * (1 + j()), h * (1 + j())],
        [w * j(), h * (1 + j())],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def moire(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    ★ 모니터를 카메라로 찍으면 픽셀 격자와 센서 격자가 간섭해 모아레 무늬가 생깁니다.
    X-ray 같은 공개 데이터셋을 화면에 띄워 웹캠으로 판독하는 시나리오라면
    이 증강이 결정적입니다.
    """
    h, w = img.shape[:2]
    freq = rng.uniform(0.35, 1.2)
    angle = rng.uniform(0, 180)
    strength = rng.uniform(0.04, 0.14)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.deg2rad(angle)
    proj = xx * np.cos(rad) + yy * np.sin(rad)
    pattern = np.sin(proj * freq * 2 * np.pi) * strength
    return np.clip(img.astype(np.float32) * (1.0 + pattern[:, :, None]), 0, 255).astype(np.uint8)


def color_temp(img: np.ndarray, shift: float) -> np.ndarray:
    """화이트밸런스 오차. shift>0 이면 따뜻하게(붉게), <0 이면 차갑게(푸르게)."""
    out = img.astype(np.float32)
    out[:, :, 2] *= (1.0 + shift)        # R
    out[:, :, 0] *= (1.0 - shift)        # B
    return np.clip(out, 0, 255).astype(np.uint8)


def vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """저가 웹캠 렌즈는 주변부가 어둡습니다."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.5)
    field = np.clip(1.0 - strength * d ** 2, 0.2, 1.0)[:, :, None]
    return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)


# ══ 증강 조합 ═══════════════════════════════════════════════════════════

def augment_general(img: np.ndarray, rng: random.Random, allow_flip: bool,
                    strength: float) -> tuple[np.ndarray, list[str]]:
    ops: list[str] = []
    out = img

    if rng.random() < 0.85:
        deg = rng.uniform(-15, 15) * strength
        out = rotate_zoom(out, deg, 1.02)
        ops.append(f"rot{deg:+.1f}")

    if rng.random() < 0.5:
        dx, dy = rng.uniform(-0.06, 0.06) * strength, rng.uniform(-0.06, 0.06) * strength
        out = translate(out, dx, dy)
        ops.append("shift")

    if allow_flip and rng.random() < 0.5:
        out = cv2.flip(out, 1)
        ops.append("flip")

    if rng.random() < 0.8:
        alpha = 1.0 + rng.uniform(-0.20, 0.20) * strength
        beta = rng.uniform(-20, 20) * strength
        out = brightness_contrast(out, alpha, beta)
        ops.append("bc")

    if rng.random() < 0.4:
        g = 1.0 + rng.uniform(-0.30, 0.30) * strength
        out = gamma(out, max(0.4, g))
        ops.append("gamma")

    if rng.random() < 0.35:
        out = gaussian_noise(out, rng.uniform(3, 10) * strength, rng)
        ops.append("noise")

    if rng.random() < 0.25:
        out = soft_blur(out, rng.uniform(0.6, 1.4) * strength)
        ops.append("blur")

    return out, ops


def augment_webcam(img: np.ndarray, rng: random.Random,
                   strength: float) -> tuple[np.ndarray, list[str]]:
    """웹캠 열화 증강. 각 효과를 확률적으로 조합합니다."""
    ops: list[str] = []
    out = img

    if rng.random() < 0.55:
        out = perspective_warp(out, 0.035 * strength, rng)
        ops.append("persp")

    if rng.random() < 0.6:
        out = illumination_gradient(out, rng.uniform(0.12, 0.35) * strength,
                                    rng.uniform(0, 360))
        ops.append("illum")

    if rng.random() < 0.3:
        out = specular_highlight(out, rng)
        ops.append("spec")

    if rng.random() < 0.4:
        out = motion_blur(out, int(rng.uniform(3, 9) * strength) | 1, rng.uniform(0, 180))
        ops.append("motion")

    if rng.random() < 0.25:
        out = moire(out, rng)
        ops.append("moire")

    if rng.random() < 0.45:
        out = color_temp(out, rng.uniform(-0.12, 0.12) * strength)
        ops.append("temp")

    if rng.random() < 0.4:
        out = vignette(out, rng.uniform(0.15, 0.4) * strength)
        ops.append("vig")

    if rng.random() < 0.5:
        out = gaussian_noise(out, rng.uniform(4, 12) * strength, rng)
        ops.append("sensor")

    # JPEG는 파이프라인의 마지막에 오는 것이 실제와 같습니다
    if rng.random() < 0.8:
        q = int(rng.uniform(40, 80))
        out = jpeg_artifact(out, q)
        ops.append(f"jpg{q}")

    return out, ops


# ══ 메인 ════════════════════════════════════════════════════════════════

def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="증강 (일반 + 웹캠 열화)")
    ap.add_argument("--config", type=Path, default=Path("prep.config.yaml"))
    ap.add_argument("--factor", type=int, default=4,
                    help="원본 1장당 생성할 증강본 수 (기본 4)")
    ap.add_argument("--webcam-degrade", action="store_true",
                    help="★ 웹캠 열화 증강을 함께 적용 (웹캠 추론이면 반드시 켜세요)")
    ap.add_argument("--flip", action="store_true",
                    help="좌우 반전 허용 (해부학적으로 안전할 때만)")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="증강 강도 배율 (0.5=약하게, 1.5=강하게)")
    ap.add_argument("--only", nargs="*", default=None, help="특정 클래스만 증강")
    ap.add_argument("--target", type=int, default=None,
                    help="클래스별 목표 장수. 지정하면 --factor 대신 클래스마다 배수를 자동 조절해 균형을 맞춥니다")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--clean", action="store_true", help="기존 증강본(aug_*)을 먼저 삭제")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.get("paths", {}).get("out", "../data/prepared"))
    train = out_root / "train"

    if not train.exists():
        sys.exit(f"학습 폴더가 없습니다: {train.resolve()}\n먼저 pipeline.py를 실행하세요.")

    fmt = cfg.get("preprocess", {}).get("out_format", "png")
    quality = cfg.get("preprocess", {}).get("jpg_quality", 95)
    ext = "jpg" if fmt == "jpg" else "png"
    rng = random.Random(args.seed)

    classes = sorted(d.name for d in train.iterdir() if d.is_dir())
    if args.only:
        classes = [c for c in classes if c in args.only]
    if not classes:
        sys.exit("증강할 클래스가 없습니다.")

    print(f"▸ 대상: {train.resolve()}")
    print(f"▸ 웹캠 열화 증강: {'사용' if args.webcam_degrade else '미사용'}"
          f"{'  ← 웹캠으로 추론한다면 반드시 켜세요' if not args.webcam_degrade else ''}")
    print(f"▸ 좌우 반전: {'허용' if args.flip else '금지(기본)'}")
    print()

    # 기존 증강본 정리
    if args.clean:
        removed = 0
        for cls in classes:
            for p in (train / cls).glob(f"{AUG_PREFIX}_*"):
                p.unlink()
                removed += 1
        print(f"▸ 기존 증강본 {removed}장 삭제\n")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"orig": 0, "made": 0})
    op_counter: dict[str, int] = defaultdict(int)

    # 원본 목록을 먼저 확정합니다 (증강본을 다시 증강하지 않도록)
    originals: dict[str, list[Path]] = {}
    for cls in classes:
        originals[cls] = sorted(
            p for p in (train / cls).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXT
            and not p.name.startswith(f"{AUG_PREFIX}_")
        )
        stats[cls]["orig"] = len(originals[cls])

    for cls in classes:
        srcs = originals[cls]
        if not srcs:
            print(f"  {cls:<14} 원본이 없어 건너뜁니다")
            continue

        # --target 이 주어지면 클래스별로 배수를 자동 조절해 균형을 맞춥니다
        if args.target:
            need = max(0, args.target - len(srcs))
            factor = (need + len(srcs) - 1) // len(srcs)
        else:
            factor = args.factor

        made = 0
        for src in srcs:
            buf = np.fromfile(str(src), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue

            for k in range(factor):
                if args.target and len(srcs) + made >= args.target:
                    break
                aug, ops = augment_general(img, rng, args.flip, args.strength)
                if args.webcam_degrade:
                    aug, wops = augment_webcam(aug, rng, args.strength)
                    ops += wops
                for o in ops:
                    op_counter[o.rstrip("0123456789+-.")] += 1

                dst = src.parent / f"{AUG_PREFIX}_{src.stem}_{k:02d}.{ext}"
                if ext == "jpg":
                    ok, enc = cv2.imencode(".jpg", aug, [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    ok, enc = cv2.imencode(".png", aug)
                if ok:
                    enc.tofile(str(dst))
                    made += 1

        stats[cls]["made"] = made
        total = stats[cls]["orig"] + made
        print(f"  {cls:<14} 원본 {stats[cls]['orig']:>4} + 증강 {made:>5} = {total:>5}장")

    print("\n✓ 완료")
    counts = [stats[c]["orig"] + stats[c]["made"] for c in classes if stats[c]["orig"]]
    if counts:
        lo, hi = min(counts), max(counts)
        print(f"  클래스 균형: 최소 {lo} / 최대 {hi}", end="")
        if lo > 0 and hi / lo > 1.5:
            print(f"  ⚠️ {hi/lo:.1f}배 불균형 — `--target {hi}` 으로 맞추세요")
        else:
            print("  (양호)")

    print(f"\n  이제 {train.resolve()} 의 클래스별 폴더를 Teachable Machine에 업로드하세요.")
    print(f"  ★ holdout/ 은 증강하지 않았고 TM에도 올리지 않습니다 — 웹앱 평가 탭 전용입니다.")
    if not args.webcam_degrade:
        print("\n  ⚠️ 웹캠으로 추론할 계획이라면 --webcam-degrade 를 켜고 다시 실행하세요.")
        print("     학습 도메인과 추론 도메인이 어긋나면 웹캠 정확도가 무너집니다.")


if __name__ == "__main__":
    main()
