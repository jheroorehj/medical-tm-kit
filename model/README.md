# RFMiD 멀티라벨 안저 질환 분류 — ConvNeXt V2-Base

RFMiD(Retinal Fundus Multi-Disease Image Dataset) 46개 라벨을 **멀티라벨**로 분류하고,
**RIADD(ISBI-2021) 공식 지표**로 평가합니다. 백본은 Hugging Face Hub의
`timm/convnextv2_base.fcmae_ft_in22k_in1k` (87.7M) — RFMiD 벤치마크에서 단일 모델 최고 macro-AUC **0.8711** 기록 모델입니다.

## 빠른 시작

```bash
pip install -r requirements.txt
```

데이터 없이 모델부터 확인 (HF 다운로드 → 순전파 → 손실/역전파 → 지표 → `.pt` 저장·재로드 → TorchScript):

```bash
python -m scripts.smoke_test
```

데이터 받고 학습:

```bash
python -m scripts.download_data --root data/rfmid
```
```bash
python -m src.train --data data/rfmid --epochs 20 --batch-size 16 --amp
```
```bash
python -m src.evaluate --ckpt checkpoints/best.pt --data data/rfmid --split test --tta
```
```bash
python -m src.export --ckpt checkpoints/best.pt --formats pt torchscript onnx
```

## 구성

| 파일 | 역할 |
|---|---|
| [labels.py](src/labels.py) | 46개 라벨 정의 + RIADD 28분류 그룹핑(27개 + `OTHER`) |
| [data.py](src/data.py) | 공식 split 로더, 원형 FOV 크롭, 증강 |
| [model.py](src/model.py) | ConvNeXtV2 빌드, `.pt` 저장/로드 |
| [losses.py](src/losses.py) | Asymmetric Loss(기본), pos_weight BCE |
| [metrics.py](src/metrics.py) | RIADD Score A/B/Final, 클래스별 AUC·AP, F1 임계값 |
| [engine.py](src/engine.py) | 학습/평가 루프, AMP, cosine+warmup |
| [train.py](src/train.py) · [evaluate.py](src/evaluate.py) · [predict.py](src/predict.py) · [export.py](src/export.py) | CLI |
| [scripts/download_data.py](scripts/download_data.py) · [scripts/smoke_test.py](scripts/smoke_test.py) | 데이터 · 검증 |

## 데이터

`ctmedtech/RFMID` (HF, **CC BY 4.0**) — 총 **8.2 GB**, PNG 3,200장 평균 2.5 MB.
공식 분할이 그대로 들어 있어 우리 수치를 리더보드와 같은 테스트셋 위에서 비교할 수 있습니다.

| split | n | Disease_Risk 양성 | 상위 클래스 |
|---|---|---|---|
| train | 1,920 | 1,519 | DR 376 · MH 317 · ODC 282 · TSLN 186 · DN 138 |
| val | 640 | 506 | DR 132 · MH 102 · ODC 72 · TSLN 65 |
| test | 640 | 506 | DR 124 · MH 104 · ODC 91 · TSLN 53 |

train에서 양성이 0인 클래스가 2개(`ODPM`, `HR`), val에서 6개입니다. 롱테일이 심해 기본 손실을 ASL로 잡았습니다.

전체를 받기 전에 파이프라인만 확인하려면:

```bash
python -m scripts.download_data --root data/rfmid --max-images 24
```

## 평가 지표 — RIADD 공식

| | 정의 | 리더보드 최고 |
|---|---|---|
| **Score A** | `Disease_Risk` ROC-AUC | **0.9636** |
| **Score B** | 28개 카테고리 `(mAP + mAUC)/2` | **0.7873** |
| **Final** | `(A + B)/2` | **0.8754** |

28개 = 상위 27개 질환 + `OTHER`(나머지 18개 희귀 질환의 OR). 모델은 46개 로짓을 내고,
[metrics.py](src/metrics.py)의 `to_riadd_view()`가 채점 시점에 28차원으로 접습니다 —
`OTHER` 확률은 18개 희귀 클래스 확률의 max입니다.

해당 split에 양성이 하나도 없는 클래스는 AUC/AP가 정의되지 않아 macro 평균에서 제외하고,
**몇 개를 제외했는지 `n_skipped`로 함께 출력**합니다. 리포트에 그대로 옮겨 적을 수 있는 형태입니다.

## 설계 메모

**멀티라벨을 정면으로 다룹니다.** 46개 독립 시그모이드 헤드입니다. `Disease_Risk`를
별도 모델로 빼지 않고 0번 헤드로 함께 학습해, 이진 스크리닝 AUC(Score A)와 질환별 확률을
**모델 하나에서** 동시에 얻습니다.

**전처리는 카메라 3종 혼합을 겨냥합니다.** RFMiD는 화각·해상도가 제각각이라
[`crop_fov()`](src/data.py)로 검은 여백을 잘라 원형 FOV만 남긴 뒤 정사각 패딩합니다.
리사이즈 전에 스케일을 맞추는 게 목적이라, 추가 center-crop은 넣지 않았습니다 —
주변부 병변(TSLN, DR 출혈)이 잘려나가기 때문입니다.

**손실은 ASL이 기본입니다.** 45개 질환 컬럼 대부분이 거의 모든 이미지에서 0이라
음성이 압도적입니다. `--loss bce`로 클래스별 `pos_weight` BCE로 바꿀 수 있습니다.

**학습률은 헤드와 백본을 분리합니다.** 사전학습 백본은 `--backbone-lr-mult`(기본 0.1)만큼
느리게 움직입니다.

## 백본 교체

`--backbone` 프리셋: `convnextv2_base`(기본) · `convnextv2_base_384` · `convnextv2_tiny` ·
`convnextv2_large` · `swinv2_base` · `effnetv2_m`. timm 모델명을 직접 넣어도 됩니다.

앙상블은 백본만 바꿔 두 번 학습한 뒤 확률을 평균하면 됩니다 —
RFMiD SOTA(6모델 스태킹, macro-AUC 0.8800)와 단일 ConvNeXtV2-Base(0.8711)의 차이가
**0.009**라, 2종 소프트보팅이면 대부분을 회수합니다.

## 출력물

- `checkpoints/best.pt` — `state_dict` + 백본명 + 라벨 + `img_size` + 클래스별 F1 임계값.
  `load_checkpoint()` 한 줄로 복원됩니다.
- `checkpoints/*_ts.pt` — TorchScript. 소스 코드 없이 `torch.jit.load`로 로드.
- `checkpoints/*.onnx` — onnxruntime / onnxruntime-web 배포용. `pip install onnx` 필요.

ConvNeXtV2-Base는 fp32 약 350 MB, int8 양자화 시 약 88 MB로 브라우저 온디바이스 추론 범위에 들어옵니다.

## 라이선스 주의

- **데이터셋** RFMiD — CC BY 4.0
- **모델 가중치** ConvNeXt V2 — 코드는 MIT지만 Meta가 배포한 ImageNet 사전학습 가중치는
  [CC BY-NC 4.0](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE)입니다.
  timm 모델 카드에는 apache-2.0으로 적혀 있어 **불일치**하니, 상업적 사용 시 상류 저장소 기준으로 판단하세요.
  교육용/해커톤은 NC 조건에 걸리지 않지만 데이터셋 라이선스 옆에 **모델 라이선스도 함께 표기**해야 합니다.
- 상업 배포가 전제라면 `--backbone effnetv2_m`(Apache 2.0)으로 학습하세요.

## 검증 상태

이 환경(Windows / torch 2.8.0+cpu / GPU 없음)에서 실제로 돌려 확인한 것:

- `scripts/smoke_test.py` — 전 단계 통과. 체크포인트 재로드 오차 0, TorchScript 오차 0.
- `src/train.py` — RFMiD 실제 이미지 24장 subset으로 1 epoch 완주, `best.pt`/`last.pt` 생성.
- `src/predict.py` — 실제 안저 이미지 추론 확인.
- `src/export.py` — pt · TorchScript · ONNX 3종 생성, onnxruntime 파리티 `1.25e-06`.

`--amp`와 `channels_last`는 CUDA에서만 켜집니다. 이 머신에는 GPU가 없어 **AMP 경로는 미검증**입니다.
전체 학습(1,920장 × 20 epoch)은 CPU로는 비현실적이니 GPU 환경에서 돌리세요.

## 참고

[RIADD 챌린지](https://riadd.grand-challenge.org/) ·
[RFMiD 데이터 논문](https://www.mdpi.com/2306-5729/6/2/14) ·
[ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) ·
[Asymmetric Loss](https://arxiv.org/abs/2009.14119)
