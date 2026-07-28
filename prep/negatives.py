#!/usr/bin/env python3
"""
negatives.py — 무효 입력('기타') 클래스 생성기

TM의 최대 약점:
  단일 라벨 softmax는 확률의 합이 항상 1입니다. 학습한 어느 클래스와도 닮지
  않은 입력을 넣어도 반드시 무언가를 1등으로 뽑습니다. 웹캠에 손을 흔들면
  "이상 소견 87%"가 나오는 이유가 이것입니다.

해법은 두 가지이고, 둘 다 쓰는 것이 좋습니다.
  ① 구조로 막기 — 게이트키퍼 모델을 앞에 세운다 (웹앱 설정 탭)
  ② ★ 데이터로 막기 — '무효 입력' 클래스를 만들어 함께 학습시킨다  ← 이 스크립트

②는 클래스 하나 추가하는 것으로 끝나는데 실사용 체감 품질을 가장 크게 바꿉니다.
심사위원이 웹캠에 아무거나 비췄을 때 "무효 입력"이 나오는지, "이상 소견 87%"가
나오는지가 여기서 갈립니다.

세 가지 소스를 섞는 것이 가장 강합니다:

  --webcam N    실제 웹캠으로 주변 환경을 찍습니다 (가장 효과적)
                책상, 손, 얼굴, 벽, 천장, 옷 — 실제로 카메라에 들어올 것들
  --synthetic N 합성 퇴화 이미지 (단색, 노이즈, 그라디언트, 격자, 텍스트 모양)
  --degrade N   실제 클래스 이미지를 파괴 (극단적 블러/암부/과노출/균일영역 확대)
                = "대상은 맞는데 촬영이 실패한 프레임"

사용:
  python3 negatives.py --config prep.config.yaml --synthetic 150 --degrade 150
  python3 negatives.py --config prep.config.yaml --webcam 200
  python3 negatives.py --config prep.config.yaml --webcam 150 --synthetic 100 --degrade 100

생성된 폴더를 다른 클래스와 함께 TM에 올리고, web/project.config.js의 classes에
kind: 'invalid' 로 등록하세요.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML이 필요합니다:  pip install pyyaml")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save(img: np.ndarray, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        enc.tofile(str(path))
    return ok


# ══ ① 합성 퇴화 이미지 ══════════════════════════════════════════════════

def synth_solid(size: int, rng: random.Random) -> np.ndarray:
    c = [rng.randrange(0, 256) for _ in range(3)]
    img = np.full((size, size, 3), c, dtype=np.uint8)
    # 완전 균일한 이미지는 실제로 거의 없으므로 약한 노이즈를 더합니다
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def synth_noise(size: int, rng: random.Random) -> np.ndarray:
    g = np.random.default_rng(rng.randrange(1 << 30))
    img = g.integers(0, 256, (size, size, 3), dtype=np.uint8)
    k = rng.choice([1, 3, 5, 9])
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


def synth_gradient(size: int, rng: random.Random) -> np.ndarray:
    a = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    b = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    angle = rng.uniform(0, 360)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rad = np.deg2rad(angle)
    t = (xx * np.cos(rad) + yy * np.sin(rad))
    t = (t - t.min()) / max(1e-6, (t.max() - t.min()))
    img = a[None, None, :] * (1 - t[:, :, None]) + b[None, None, :] * t[:, :, None]
    return img.astype(np.uint8)


def synth_grid(size: int, rng: random.Random) -> np.ndarray:
    """격자·체커보드 — 화면 UI나 타일 벽 같은 인공 패턴을 흉내냅니다."""
    step = rng.choice([8, 12, 16, 24, 32])
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c1 = [rng.randrange(256) for _ in range(3)]
    c2 = [rng.randrange(256) for _ in range(3)]
    for i in range(0, size, step):
        for j in range(0, size, step):
            img[i:i + step, j:j + step] = c1 if ((i // step + j // step) % 2 == 0) else c2
    if rng.random() < 0.5:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(0, 90), 1.2)
        img = cv2.warpAffine(img, m, (size, size), borderMode=cv2.BORDER_REFLECT)
    return img


def synth_textish(size: int, rng: random.Random) -> np.ndarray:
    """
    문서·라벨·모니터 UI처럼 글자가 있는 화면.
    의료영상 대신 서류나 화면이 카메라에 들어오는 경우가 실제로 매우 흔합니다.
    """
    bg = rng.randrange(180, 256)
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    fg = rng.randrange(0, 90)
    lines = rng.randint(4, 11)
    y = rng.randint(10, 30)
    for _ in range(lines):
        h = rng.randint(4, 9)
        w = rng.randint(int(size * 0.25), int(size * 0.85))
        x = rng.randint(4, max(5, size - w - 4))
        cv2.rectangle(img, (x, y), (x + w, y + h), (fg, fg, fg), -1)
        y += h + rng.randint(6, 16)
        if y > size - 12:
            break
    return cv2.GaussianBlur(img, (3, 3), 0)


def synth_blob(size: int, rng: random.Random) -> np.ndarray:
    """유기적인 얼룩 — 손, 피부, 옷 같은 부드러운 저주파 형태를 흉내냅니다."""
    small = np.random.default_rng(rng.randrange(1 << 30)).integers(
        0, 256, (rng.randint(3, 8), rng.randint(3, 8), 3), dtype=np.uint8)
    img = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (15, 15), 0)


SYNTH = [synth_solid, synth_noise, synth_gradient, synth_grid, synth_textish, synth_blob]


# ══ ② 실제 이미지 퇴화 ══════════════════════════════════════════════════

def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    "대상은 맞지만 촬영이 실패한 프레임"을 만듭니다.
    실제 사용에서 가장 흔한 무효 입력이 바로 이것입니다 —
    초점이 안 맞거나, 너무 어둡거나, 대상에 너무 가까이 붙어 균일한 면만 잡힌 경우.
    """
    mode = rng.choice(["blur", "dark", "bright", "zoom", "motion", "occlude"])
    size = img.shape[0]

    if mode == "blur":
        k = rng.choice([21, 31, 41, 51])
        return cv2.GaussianBlur(img, (k, k), 0)

    if mode == "dark":
        f = rng.uniform(0.06, 0.22)
        out = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        g = np.random.default_rng(rng.randrange(1 << 30))
        return np.clip(out + g.normal(0, 6, out.shape), 0, 255).astype(np.uint8)

    if mode == "bright":
        return np.clip(img.astype(np.float32) * rng.uniform(2.4, 4.5) + 40,
                       0, 255).astype(np.uint8)

    if mode == "zoom":
        # 균일한 영역에 극단적으로 확대 — 무엇을 보는지 알 수 없는 상태
        r = rng.uniform(0.05, 0.14)
        side = max(4, int(size * r))
        y = rng.randint(0, size - side)
        x = rng.randint(0, size - side)
        crop = img[y:y + side, x:x + side]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)

    if mode == "motion":
        length = rng.choice([25, 35, 45]) | 1
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5),
                                    rng.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, m, (length, length))
        return cv2.filter2D(img, -1, kernel / max(1e-6, kernel.sum()))

    # occlude — 대상이 손이나 물체에 대부분 가려진 상태
    out = img.copy()
    for _ in range(rng.randint(1, 3)):
        w = rng.randint(int(size * 0.4), int(size * 0.9))
        h = rng.randint(int(size * 0.4), int(size * 0.9))
        x = rng.randint(-w // 3, size - w // 3)
        y = rng.randint(-h // 3, size - h // 3)
        c = [rng.randrange(256) for _ in range(3)]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, -1)
    return cv2.GaussianBlur(out, (7, 7), 0)


# ══ ③ 웹캠 수집 ═════════════════════════════════════════════════════════

def capture_webcam(n: int, size: int, dst: Path, camera: int, interval: float) -> int:
    """
    실제 웹캠으로 주변을 찍습니다. 가장 효과적인 무효 데이터입니다.
    촬영 중 카메라를 계속 움직여 다양한 대상이 들어오게 하세요.
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"  ⚠️ 카메라를 열 수 없습니다 (index={camera}) — 웹캠 수집을 건너뜁니다")
        return 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print()
    print("  ┌─ 웹캠 무효 데이터 수집 ──────────────────────────────────────┐")
    print("  │ 카메라를 천천히 움직여 다양한 대상을 담으세요.               │")
    print("  │   책상 · 손 · 얼굴 · 벽 · 천장 · 옷 · 모니터 · 문서 · 바닥    │")
    print("  │ 판독 대상(의료영상)은 절대 담지 마세요.                      │")
    print("  │ SPACE=시작/일시정지   ESC=종료                               │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()

    saved = 0
    running = False
    last = 0.0
    win = "negatives (SPACE=start/pause, ESC=quit)"

    while saved < n:
        ok, frame = cap.read()
        if not ok:
            break

        view = frame.copy()
        label = f"{saved}/{n}  {'REC' if running else 'PAUSED'}"
        color = (0, 0, 255) if running else (160, 160, 160)
        cv2.putText(view, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imshow(win, view)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key == 32:
            running = not running

        now = time.time()
        if running and (now - last) >= interval:
            last = now
            h, w = frame.shape[:2]
            side = min(h, w)
            sq = frame[(h - side) // 2:(h - side) // 2 + side,
                       (w - side) // 2:(w - side) // 2 + side]
            img = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
            if save(img, dst / f"neg_cam_{saved:05d}.jpg"):
                saved += 1

    cap.release()
    cv2.destroyAllWindows()
    return saved


# ══ 메인 ════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="무효 입력 클래스 생성기")
    ap.add_argument("--config", type=Path, default=Path("prep.config.yaml"))
    ap.add_argument("--class-name", default="invalid",
                    help="생성할 클래스 폴더명 (web/project.config.js의 invalid 클래스 id와 일치시킬 것)")
    ap.add_argument("--synthetic", type=int, default=0, help="합성 퇴화 이미지 장수")
    ap.add_argument("--degrade", type=int, default=0, help="실제 이미지 퇴화 장수")
    ap.add_argument("--webcam", type=int, default=0, help="웹캠 수집 장수")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.35, help="웹캠 수집 간격(초)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--clean", action="store_true", help="기존 무효 클래스를 먼저 비웁니다")
    args = ap.parse_args()

    if not (args.synthetic or args.degrade or args.webcam):
        ap.error("--synthetic / --degrade / --webcam 중 최소 하나를 지정하세요.\n"
                 "권장: --webcam 150 --synthetic 100 --degrade 100")

    cfg = load_config(args.config)
    out_root = Path(cfg.get("paths", {}).get("out", "../data/prepared"))
    train = out_root / "train"
    size = int(cfg.get("preprocess", {}).get("size", 224))
    rng = random.Random(args.seed)

    if not train.exists():
        sys.exit(f"학습 폴더가 없습니다: {train.resolve()}\n먼저 pipeline.py를 실행하세요.")

    dst = train / args.class_name
    if args.clean and dst.exists():
        for p in dst.glob("*"):
            if p.is_file():
                p.unlink()
        print(f"▸ 기존 '{args.class_name}' 폴더를 비웠습니다")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"▸ 출력: {dst.resolve()}\n")

    total = 0

    # ① 합성
    if args.synthetic:
        for i in range(args.synthetic):
            fn = SYNTH[i % len(SYNTH)]
            img = fn(size, rng)
            # 웹캠으로 들어올 것이므로 JPEG 열화를 한 번 입힙니다
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,
                                                 rng.randint(45, 85)])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if save(img, dst / f"neg_syn_{i:05d}.jpg"):
                total += 1
        print(f"  합성 퇴화       {args.synthetic:>5}장")

    # ② 실제 이미지 퇴화
    if args.degrade:
        pool: list[Path] = []
        for d in train.iterdir():
            if not d.is_dir() or d.name == args.class_name:
                continue
            pool += [p for p in d.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXT
                     and not p.name.startswith("aug_")]
        if not pool:
            print("  ⚠️ 퇴화시킬 원본이 없습니다 — --degrade 를 건너뜁니다")
        else:
            rng.shuffle(pool)
            made = 0
            for i in range(args.degrade):
                src = pool[i % len(pool)]
                buf = np.fromfile(str(src), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[0] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                if save(degrade(img, rng), dst / f"neg_deg_{i:05d}.jpg"):
                    made += 1
                    total += 1
            print(f"  실제 이미지 퇴화 {made:>4}장")

    # ③ 웹캠
    if args.webcam:
        got = capture_webcam(args.webcam, size, dst, args.camera, args.interval)
        total += got
        print(f"  웹캠 수집       {got:>5}장")

    n_files = len([p for p in dst.iterdir() if p.is_file()])
    print(f"\n✓ 무효 클래스 '{args.class_name}' — 이번에 {total}장 추가 / 총 {n_files}장")
    print()
    print("  다음 단계:")
    print(f"   1) TM에 '{args.class_name}' 클래스를 추가하고 이 폴더를 업로드합니다.")
    print(f"   2) web/project.config.js의 classes에 다음을 추가합니다:")
    print()
    print(f"        {{ id: '{args.class_name}', label: '무효 입력', kind: 'invalid',")
    print(f"          color: '#A8A49E',")
    print(f"          advice: '판독 대상이 화면에 없습니다. 대상을 가이드 영역에 맞춰 주세요.' }},")
    print()
    print("  ⚠️ 무효 클래스가 다른 클래스보다 지나치게 많으면 모델이 '전부 무효'로")
    print("     기울어집니다. 다른 클래스 평균 장수의 0.8~1.2배 수준으로 맞추세요.")
    print("  ⚠️ 무효 클래스는 train/ 에만 넣습니다. holdout/ 에도 일부 넣고 싶다면")
    print("     별도로 수집해 직접 배치하세요 (여기서 만든 것을 나누면 누출됩니다).")


if __name__ == "__main__":
    main()
