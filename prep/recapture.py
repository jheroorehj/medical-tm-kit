#!/usr/bin/env python3
"""
recapture.py — 재촬영 증강 하네스 (도메인 갭 정면 돌파)

문제:
  공개 의료 데이터셋(X-ray, CT, 병리)은 웹캠으로 찍을 대상이 아닙니다. 그런데
  서비스는 웹캠으로 추론합니다. 그래서 학습 도메인(스캔 원본)과 추론 도메인
  (모니터를 찍은 웹캠 영상)이 완전히 달라지고, 정확도가 무너집니다.

  발표는 "Kaggle 데이터로 92%"인데 심사위원이 웹캠을 켜면 아무거나 찍어도
  같은 답이 나오는 상황이 여기서 생깁니다.

해법:
  학습 데이터를 실제 모니터에 띄우고, 실제 웹캠으로 되찍어 학습셋에 추가합니다.
  그러면 학습 도메인 = 추론 도메인이 됩니다. augment.py의 --webcam-degrade가
  이것을 근사하는 것이고, 이 스크립트는 진짜로 하는 것입니다.

동작:
  1. 웹캠을 열고 화면 영역을 한 번 지정합니다 (캘리브레이션)
  2. 이미지를 전체화면으로 순차 표시하면서 웹캠으로 자동 캡처합니다
  3. 지정한 영역만 잘라 224 정사각으로 저장합니다

준비:
  · 웹캠을 모니터 정면에 삼각대나 거치대로 고정하세요 (손으로 들면 흔들립니다)
  · 화면 반사를 줄이려면 조명을 측면으로 두거나 조금 낮추세요
  · 노트북 내장 캠이면 노트북을 뒤로 젖혀 화면을 향하게 하기 어렵습니다.
    외장 웹캠이 없으면 두 대의 기기(한 대는 표시용)를 쓰는 편이 낫습니다.

사용:
  python3 recapture.py --config prep.config.yaml --limit 80
  python3 recapture.py --config prep.config.yaml --only abnormal --limit 120 --settle 0.25

ESC로 언제든 중단할 수 있고, 그때까지 저장한 것은 유지됩니다.
"""

from __future__ import annotations

import argparse
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
WIN = "recapture"
PREFIX = "recap"


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"카메라를 열 수 없습니다 (index={index}). --camera 로 다른 번호를 시도하세요.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def grab(cap: cv2.VideoCapture, flush: int = 5) -> np.ndarray | None:
    """
    웹캠은 프레임을 버퍼링합니다. 그냥 read()하면 몇 프레임 전의 화면이 잡혀
    "이전 이미지"가 저장되는 사고가 납니다. 버퍼를 비우고 최신 프레임을 씁니다.
    """
    frame = None
    for _ in range(max(1, flush)):
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


def make_fullscreen(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def calibration_pattern(w: int = 1600, h: int = 900) -> np.ndarray:
    """
    화면 영역을 사람이 눈으로 찾기 쉽도록 테두리와 십자선이 있는 패턴을 만듭니다.
    """
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 14)
    side = int(min(w, h) * 0.8)
    x0, y0 = (w - side) // 2, (h - side) // 2
    cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (0, 220, 255), 6)
    cv2.line(img, (w // 2, y0), (w // 2, y0 + side), (0, 220, 255), 2)
    cv2.line(img, (x0, h // 2), (x0 + side, h // 2), (0, 220, 255), 2)
    cv2.putText(img, "CALIBRATION", (x0 + 20, y0 - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2, cv2.LINE_AA)
    return img


def letterbox_to_screen(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """이미지를 화면 비율에 맞춰 검은 배경 위에 최대 크기로 배치합니다."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih) * 0.92          # 화면 가장자리에 약간 여백
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="재촬영 증강 — 모니터에 띄워 웹캠으로 되찍기")
    ap.add_argument("--config", type=Path, default=Path("prep.config.yaml"))
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    ap.add_argument("--only", nargs="*", default=None, help="특정 클래스만")
    ap.add_argument("--limit", type=int, default=80, help="클래스당 최대 재촬영 장수")
    ap.add_argument("--settle", type=float, default=0.18,
                    help="이미지 표시 후 캡처까지 대기 시간(초). 모니터 응답이 느리면 올리세요")
    ap.add_argument("--shots", type=int, default=1,
                    help="이미지 1장당 캡처 횟수 (2 이상이면 미세한 조명/노이즈 변화를 얻습니다)")
    ap.add_argument("--screen-w", type=int, default=1600)
    ap.add_argument("--screen-h", type=int, default=900)
    ap.add_argument("--out", type=Path, default=None,
                    help="출력 경로 (기본: prepared/train/<class>/)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.get("paths", {}).get("out", "../data/prepared"))
    train = out_root / "train"
    size = int(cfg.get("preprocess", {}).get("size", 224))

    if not train.exists():
        sys.exit(f"학습 폴더가 없습니다: {train.resolve()}\n먼저 pipeline.py를 실행하세요.")

    classes = sorted(d.name for d in train.iterdir() if d.is_dir())
    if args.only:
        classes = [c for c in classes if c in args.only]
    if not classes:
        sys.exit("재촬영할 클래스가 없습니다.")

    cap = open_camera(args.camera)
    make_fullscreen(WIN)

    # ── 1. 캘리브레이션 ────────────────────────────────────────────────
    print("=" * 68)
    print(" 캘리브레이션")
    print("=" * 68)
    print(" 1) 전체화면에 노란 사각형 패턴이 표시됩니다.")
    print(" 2) 웹캠이 그 화면을 정면으로 담도록 위치를 고정하세요.")
    print(" 3) 준비되면 아무 키나 누르세요 (ESC=취소).")
    print()

    pattern = calibration_pattern(args.screen_w, args.screen_h)
    while True:
        cv2.imshow(WIN, pattern)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("취소했습니다.")
        if key != 255:
            break

    frame = grab(cap, flush=8)
    if frame is None:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit("카메라에서 프레임을 받지 못했습니다.")

    cv2.destroyWindow(WIN)
    print(" 4) 캡처된 웹캠 영상에서 '노란 사각형 안쪽'을 드래그로 지정하고 ENTER를 누르세요.")
    print("    (c 키로 취소)")
    roi = cv2.selectROI("select screen area", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("select screen area")

    x, y, w, h = [int(v) for v in roi]
    if w < 20 or h < 20:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit("영역이 지정되지 않았습니다. 다시 실행하세요.")
    print(f"    지정된 영역: x={x} y={y} w={w} h={h}\n")

    # ── 2. 재촬영 루프 ─────────────────────────────────────────────────
    make_fullscreen(WIN)
    total_saved = 0
    aborted = False

    for cls in classes:
        if aborted:
            break
        src_dir = train / cls
        # 증강본과 기존 재촬영본은 제외하고 원본만 씁니다
        files = sorted(
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXT
            and not p.name.startswith(("aug_", f"{PREFIX}_"))
        )[: args.limit]

        if not files:
            print(f"  {cls:<14} 원본이 없어 건너뜁니다")
            continue

        dst_dir = (args.out / cls) if args.out else src_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, path in enumerate(files, 1):
            buf = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue

            cv2.imshow(WIN, letterbox_to_screen(img, args.screen_w, args.screen_h))
            # 표시가 실제로 갱신되도록 이벤트 루프를 돌립니다
            key = cv2.waitKey(max(1, int(args.settle * 1000))) & 0xFF
            if key == 27:
                aborted = True
                break
            time.sleep(0.02)

            for s in range(args.shots):
                frame = grab(cap, flush=3)
                if frame is None:
                    continue
                crop = frame[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                # 짧은 변 기준 중앙 정사각 → size
                ch, cw = crop.shape[:2]
                side = min(ch, cw)
                sq = crop[(ch - side) // 2:(ch - side) // 2 + side,
                          (cw - side) // 2:(cw - side) // 2 + side]
                final = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)

                dst = dst_dir / f"{PREFIX}_{path.stem}_{s:02d}.jpg"
                ok, enc = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    enc.tofile(str(dst))
                    saved += 1
                    total_saved += 1

            print(f"  {cls:<14} {i}/{len(files)}  저장 {saved}", end="\r", flush=True)

        print(f"  {cls:<14} {len(files)}장 표시 → {saved}장 저장" + " " * 12)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✓ 재촬영 완료 — 총 {total_saved}장{' (중단됨)' if aborted else ''}")
    if total_saved:
        print("\n  이 이미지들은 학습 도메인 = 추론 도메인 이라는 성질을 갖습니다.")
        print("  원본 스캔 이미지와 함께 TM에 올리면, 웹캠 추론 정확도가 크게 올라갑니다.")
        print("\n  ⚠️ 재촬영본은 train/ 에만 넣으세요. holdout/ 에 넣으면 평가가 오염됩니다.")
        print("  ⚠️ 재촬영본 비중이 너무 높으면 원본 스캔 판독 성능이 떨어집니다.")
        print("     원본 : 재촬영 = 대략 2:1 ~ 1:1 사이를 권장합니다.")


if __name__ == "__main__":
    main()
